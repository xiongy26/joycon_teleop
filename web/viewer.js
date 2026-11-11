// EL-A3 ACT simulation viewer — MuJoCo compiled to WebAssembly (mujoco-js),
// rendered with three.js following the official wasm demo_app pattern:
// mjv_updateScene() fills an MjvScene, then every MjvGeom is mirrored into a
// THREE.Mesh. Adds mesh-geom support built from model.mesh_vert/mesh_face.

import * as THREE from "./vendor/three.module.js";
import loadMujoco from "./vendor/mujoco_wasm.js";

const XML_URL = "../act_sim/el_a3_pick_scene.xml";
// Keep the virtual filenames identical to the names used by the XML.  The
// browser has no access to the repository filesystem, so every mesh referenced
// by the scene must be downloaded explicitly into MuJoCo's MEMFS first.
const MESH_ASSETS = [
  ...["l1.stl", "l2.stl", "l3.stl", "l4.stl", "l5.stl", "l6.stl", "el05_l6.stl", "jaw.stl", "yam_gripper_rail.stl"]
    .map((name) => ({ name, url: `../robot/el_a3_description/meshes/${name}` })),
  { name: "tip_left.stl", url: "../yam_arm_mujoco_standalone/yam_arm_standalone/yam/assets/tip_left.stl" },
  { name: "tip_right.stl", url: "../yam_arm_mujoco_standalone/yam_arm_standalone/yam/assets/tip_right.stl" },
];
const EPISODES_URL = "./data/episodes.json";
const ROLLOUTS_URL = "./data/rollouts.json";

const ARM_JOINTS = ["L1_joint", "L2_joint", "L3_joint", "L4_joint", "L5_joint", "L6_joint", "L7_joint"];
const HOME = [1.04, 1.355, -1.15, 0.243, -1.096, 0.0, 0.0];
const OBJ_JOINT = 3, OBJ_BODY = 1, OBJ_SITE = 6, OBJ_CAMERA = 7;

const $ = (id) => document.getElementById(id);
const statusEl = $("status");

function setStatus(msg) {
  statusEl.textContent = msg;
}

function errorMessage(error) {
  if (error === undefined || error === null) {
    return "未知错误（错误对象为空，请打开浏览器控制台查看详情）";
  }
  // MuJoCo's embind wrapper throws mjCError/CppException objects without a
  // `message` field. Interpolating err.message used to render "undefined".
  if (error.name === "mjCError" || error.name === "CppException") {
    return "MuJoCo 编译场景失败（请检查 XML 和 mesh 文件路径；详细信息见控制台）";
  }
  if (error instanceof Error && error.message) return error.message;
  if (typeof error === "string" && error.trim()) return error;
  if (error && typeof error.message === "string" && error.message.trim()) return error.message;
  try {
    const text = JSON.stringify(error);
    if (text && text !== "{}") return text;
  } catch (e) { /* ignore non-serializable errors */ }
  return "未知错误（请打开浏览器控制台查看详情）";
}

// Emscripten/embind array-like -> JS array (supports both view and handle).
function vecToArray(v) {
  if (v === undefined || v === null) return [];
  if (typeof v.length === "number" && v.length !== undefined) {
    try { return Array.from(v); } catch (e) { /* fall through */ }
  }
  try {
    const n = Number(v.size());
    const out = new Array(n);
    for (let i = 0; i < n; i++) out[i] = v.get(i);
    return out;
  } catch (e) {
    return [];
  }
}

class Viewer {
  constructor(mujoco) {
    this.mujoco = mujoco;
    this.maxGeoms = 2 ** 15;
    this.meshes = [];
    this.geomCache = new Map();
    this.paused = false;
    this.showContacts = false;
    this.viewMode = "free"; // free | cam_high | cam_wrist
    this.freeCamState = { lookat: [0, -0.35, 0.12], distance: 1.1, azimuth: 150, elevation: -28 };
    this.playback = null; // {kind, list, idx, frame, playing}
    this.jointSliders = [];
  }

  async loadModel() {
    const m = this.mujoco;
    const fs = m.FS;
    try { fs.mkdir("/working"); } catch (e) { /* exists */ }
    try { fs.mkdir("/working/meshes"); } catch (e) { /* exists */ }
    setStatus("下载 mesh …");
    for (const asset of MESH_ASSETS) {
      const resp = await fetch(asset.url);
      if (!resp.ok) throw new Error(`下载 ${asset.name} 失败（HTTP ${resp.status}）`);
      fs.writeFile(`/working/meshes/${asset.name}`, new Uint8Array(await resp.arrayBuffer()));
    }
    setStatus("加载模型 …");
    const xmlResp = await fetch(XML_URL);
    if (!xmlResp.ok) throw new Error(`下载场景 XML 失败（HTTP ${xmlResp.status}）`);
    let xml = await xmlResp.text();
    // All assets are placed in one MEMFS directory.  Flatten paths from the
    // source XML (including the yam tip meshes) so MuJoCo does not resolve
    // them outside /working/meshes.
    xml = xml.replace(/meshdir="[^"]*"/, 'meshdir="meshes"');
    xml = xml.replace(/(<mesh\b[^>]*\bfile=")[^"]+(")/g, (match, prefix, suffix) => {
      const file = match.match(/\bfile="([^"]+)"/)[1];
      return `${prefix}${file.split(/[\\/]/).pop()}${suffix}`;
    });
    fs.writeFile("/working/scene.xml", xml);

    try {
      this.model = m.MjModel.loadFromXML("/working/scene.xml");
    } catch (err) {
      // Keep the original embind exception in the console while exposing a
      // useful status message in the page. mjCError.message is undefined.
      console.error("MuJoCo loadFromXML failed", err);
      throw new Error(errorMessage(err), { cause: err });
    }
    if (!this.model) throw new Error("MuJoCo 没有返回模型对象");
    this.data = new m.MjData(this.model);

    const id2 = (type, name) => m.mj_name2id(this.model, type, name);
    this.siteId = id2(OBJ_SITE, "gripper_tip_site");
    this.camIds = { cam_high: id2(OBJ_CAMERA, "cam_high"), cam_wrist: id2(OBJ_CAMERA, "cam_wrist") };
    this.cubeBodyId = id2(OBJ_BODY, "cube");
    this.jointQadr = ARM_JOINTS.map((n) => Number(this.model.jnt_qposadr[id2(OBJ_JOINT, n)]));
    this.cubeQadr = Number(this.model.jnt_qposadr[id2(OBJ_JOINT, "cube_joint")]);
    this.jointRange = ARM_JOINTS.map((n, i) => {
      const jid = id2(OBJ_JOINT, n);
      return [Number(this.model.jnt_range[2 * jid]), Number(this.model.jnt_range[2 * jid + 1])];
    });
    this.buildSceneGraph();
    this.buildUI();
    this.resetSim();
  }

  // -- three.js scene --------------------------------------------------
  buildSceneGraph() {
    this.threeScene = new THREE.Scene();
    this.threeScene.background = new THREE.Color(0x10141a);

    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    const canvasWrap = $("canvas-wrap");
    canvasWrap.appendChild(this.renderer.domElement);
    this.renderer.domElement.addEventListener("pointerdown", (e) => this.onPointerDown(e));
    window.addEventListener("pointermove", (e) => this.onPointerMove(e));
    window.addEventListener("pointerup", () => { this.drag = null; });
    this.renderer.domElement.addEventListener("contextmenu", (e) => e.preventDefault());
    this.renderer.domElement.addEventListener("wheel", (e) => {
      if (this.viewMode !== "free") return;
      e.preventDefault();
      this.freeCamState.distance *= Math.exp(e.deltaY * 0.001);
      this.freeCamState.distance = Math.min(5, Math.max(0.15, this.freeCamState.distance));
    }, { passive: false });
    window.addEventListener("resize", () => this.resize());

    this.camera = new THREE.PerspectiveCamera(45, 1, 0.005, 50);
    this.camera.up.set(0, 0, 1);

    this.threeScene.add(new THREE.AmbientLight(0xffffff, 0.45));
    const key = new THREE.DirectionalLight(0xffffff, 1.6);
    key.position.set(1.0, -1.2, 1.8);
    key.castShadow = true;
    key.shadow.mapSize.set(2048, 2048);
    key.shadow.camera.near = 0.1;
    key.shadow.camera.far = 6;
    key.shadow.camera.left = key.shadow.camera.bottom = -1;
    key.shadow.camera.right = key.shadow.camera.top = 1;
    key.shadow.normalBias = 0.02;
    this.threeScene.add(key);
    const fill = new THREE.DirectionalLight(0x88aaff, 0.35);
    fill.position.set(-1.5, -0.5, 1.0);
    this.threeScene.add(fill);

    const m = this.mujoco;
    this.mjvOption = new m.MjvOption();
    this.mjvPerturb = new m.MjvPerturb();
    this.mjvCamera = new m.MjvCamera();
    this.mjvScene = new m.MjvScene(this.model, this.maxGeoms);
    this.resize();
  }

  resize() {
    const wrap = $("canvas-wrap");
    const w = wrap.clientWidth, h = wrap.clientHeight;
    this.renderer.setSize(w, h);
    this.camera.aspect = w / Math.max(h, 1);
    this.camera.updateProjectionMatrix();
  }

  checkerTexture(half) {
    const cacheKey = `checker:${half}`;
    if (this.geomCache.has(cacheKey)) return this.geomCache.get(cacheKey);
    const canvas = document.createElement("canvas");
    canvas.width = canvas.height = 256;
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "#8d99a8";
    ctx.fillRect(0, 0, 256, 256);
    ctx.fillStyle = "#7c8896";
    ctx.fillRect(0, 0, 128, 128);
    ctx.fillRect(128, 128, 128, 128);
    const tex = new THREE.CanvasTexture(canvas);
    tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
    tex.repeat.set(half, half); // one 2x2-cell tile per meter
    tex.colorSpace = THREE.SRGBColorSpace;
    this.geomCache.set(cacheKey, tex);
    return tex;
  }

  meshGeometry(sceneDataId) {
    // mjv scene renumbers mesh geoms: 2*meshid = original, 2*meshid+1 = convex hull
    const meshId = Math.floor(Number(sceneDataId) / 2);
    const key = `mesh:${meshId}`;
    if (this.geomCache.has(key)) return this.geomCache.get(key);
    const model = this.model;
    const vadr = Number(model.mesh_vertadr[meshId]);
    const vnum = Number(model.mesh_vertnum[meshId]);
    const fadr = Number(model.mesh_faceadr[meshId]);
    const fnum = Number(model.mesh_facenum[meshId]);
    const verts = new Float32Array(vecToArray(model.mesh_vert).slice(vadr * 3, (vadr + vnum) * 3));
    // model.mesh_vert is the compilation frame; MuJoCo folds mesh_pos/quat into
    // geom_pos/quat at compile time (verified against mjr_uploadMesh), so the
    // mjv scene pose applies to these vertices directly — no extra transform.
    const idx = vecToArray(model.mesh_face).slice(fadr * 3, (fadr + fnum) * 3);
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(verts, 3));
    geo.setIndex(idx);
    geo.computeVertexNormals();
    this.geomCache.set(key, geo);
    return geo;
  }

  primitiveGeometry(mjvGeom) {
    const gtype = Number(mjvGeom.type);
    const dataid = Number(mjvGeom.dataid);
    const size = vecToArray(mjvGeom.size).map(Number);
    if (gtype === this.mujoco.mjtGeom.mjGEOM_MESH.value) return this.meshGeometry(dataid);
    const key = `prim:${gtype}:${size.join(",")}`;
    if (this.geomCache.has(key)) return this.geomCache.get(key);
    const T = this.mujoco.mjtGeom;
    let geo;
    if (gtype === T.mjGEOM_PLANE.value) {
      // PlaneGeometry already faces +z, matching MuJoCo's plane convention
      // (do NOT rotate it — rotating stood the floor up like a wall).
      const half = size[0] || 50;
      geo = new THREE.PlaneGeometry(2 * half, 2 * (size[1] || half));
      geo.userData.map = this.checkerTexture(half);
    } else if (gtype === T.mjGEOM_SPHERE.value) geo = new THREE.SphereGeometry(size[0], 24, 16);
    else if (gtype === T.mjGEOM_CAPSULE.value) {
      geo = new THREE.CapsuleGeometry(size[0], 2 * size[2], 8, 16);
      geo.rotateX(Math.PI / 2);
    } else if (gtype === T.mjGEOM_BOX.value) geo = new THREE.BoxGeometry(2 * size[0], 2 * size[1], 2 * size[2]);
    else if (gtype === T.mjGEOM_CYLINDER.value) {
      geo = new THREE.CylinderGeometry(size[0], size[1], 2 * size[2], 24);
      geo.rotateX(Math.PI / 2);
    } else geo = new THREE.BufferGeometry();
    this.geomCache.set(key, geo);
    return geo;
  }

  syncScene() {
    const m = this.mujoco;
    m.mjv_updateScene(
      this.model, this.data, this.mjvOption, this.mjvPerturb,
      this.mjvCamera, m.mjtCatBit.mjCAT_ALL.value, this.mjvScene
    );
    const geoms = this.mjvScene.geoms;
    const count = Number(geoms.size());
    // Drop surplus meshes (e.g. contacts toggled off).
    while (this.meshes.length > count) {
      const mesh = this.meshes.pop();
      this.threeScene.remove(mesh);
      if (mesh.material) mesh.material.dispose();
    }
    for (let i = 0; i < count; i++) {
      const g = geoms.get(i);
      const rgba = vecToArray(g.rgba).map(Number);
      const mat = vecToArray(g.mat).map(Number);
      const pos = vecToArray(g.pos).map(Number);
      let mesh = this.meshes[i];
      if (!mesh) {
        const geo = this.primitiveGeometry(g);
        const material = new THREE.MeshPhongMaterial({ flatShading: Number(g.type) === m.mjtGeom.mjGEOM_MESH.value });
        if (geo.userData && geo.userData.map) {
          material.map = geo.userData.map;
          material.color.setRGB(1, 1, 1); // texture provides the shade
        }
        mesh = new THREE.Mesh(geo, material);
        mesh.castShadow = true;
        mesh.receiveShadow = true;
        mesh.matrixAutoUpdate = false;
        this.meshes.push(mesh);
        this.threeScene.add(mesh);
      }
      if (!mesh.material.map) {
        mesh.material.color.setRGB(rgba[0], rgba[1], rgba[2]);
      }
      mesh.material.opacity = rgba[3];
      mesh.material.transparent = rgba[3] < 1.0;
      mesh.material.side = THREE.DoubleSide;
      mesh.matrix.set(
        mat[0], mat[1], mat[2], pos[0],
        mat[3], mat[4], mat[5], pos[1],
        mat[6], mat[7], mat[8], pos[2],
        0, 0, 0, 1
      );
      mesh.matrixWorldNeedsUpdate = true;
      g.delete();
    }
    geoms.delete();
  }

  updateCamera() {
    if (this.viewMode === "free") {
      const st = this.freeCamState;
      this.mjvCamera.type = 0;
      const look = st.lookat;
      this.mjvCamera.lookat[0] = look[0]; this.mjvCamera.lookat[1] = look[1]; this.mjvCamera.lookat[2] = look[2];
      this.mjvCamera.distance = st.distance;
      this.mjvCamera.azimuth = st.azimuth;
      this.mjvCamera.elevation = st.elevation;
      const az = (st.azimuth * Math.PI) / 180, el = (st.elevation * Math.PI) / 180;
      const off = new THREE.Vector3(Math.cos(az) * Math.cos(el), Math.sin(az) * Math.cos(el), -Math.sin(el)).multiplyScalar(st.distance);
      this.camera.matrixAutoUpdate = true;
      this.camera.position.set(look[0] + off.x, look[1] + off.y, look[2] + off.z);
      this.camera.lookAt(look[0], look[1], look[2]);
      this.camera.updateMatrix();
      this.camRight = new THREE.Vector3().setFromMatrixColumn(this.camera.matrix, 0).setZ(0).normalize();
      this.camUpPan = new THREE.Vector3().setFromMatrixColumn(this.camera.matrix, 1);
    } else {
      const camid = this.camIds[this.viewMode];
      this.mjvCamera.type = 2; // mjCAMERA_FIXED
      this.mjvCamera.fixedcamid = camid;
      const mat = vecToArray(this.data.cam_xmat).slice(camid * 9, camid * 9 + 9);
      const pos = vecToArray(this.data.cam_xpos).slice(camid * 3, camid * 3 + 3);
      this.camera.matrixAutoUpdate = false;
      this.camera.matrix.set(
        mat[0], mat[1], mat[2], pos[0],
        mat[3], mat[4], mat[5], pos[1],
        mat[6], mat[7], mat[8], pos[2],
        0, 0, 0, 1
      );
      this.camera.matrixWorldNeedsUpdate = true;
    }
  }

  onPointerDown(e) {
    if (this.viewMode !== "free") return;
    this.drag = { x: e.clientX, y: e.clientY, pan: e.button === 2 || e.shiftKey };
    e.preventDefault();
  }

  onPointerMove(e) {
    if (!this.drag || this.viewMode !== "free") return;
    const dx = e.clientX - this.drag.x, dy = e.clientY - this.drag.y;
    this.drag.x = e.clientX; this.drag.y = e.clientY;
    const st = this.freeCamState;
    if (this.drag.pan) {
      const k = st.distance * 0.0016;
      const d = this.camRight.clone().multiplyScalar(-dx * k).add(this.camUpPan.clone().multiplyScalar(dy * k));
      st.lookat[0] += d.x; st.lookat[1] += d.y; st.lookat[2] += d.z;
    } else {
      st.azimuth -= dx * 0.25;
      st.elevation = Math.min(-2, Math.max(-89, st.elevation - dy * 0.25));
    }
  }

  // -- simulation control ------------------------------------------------
  resetSim() {
    const m = this.mujoco;
    m.mj_resetData(this.model, this.data);
    this.jointQadr.forEach((adr, i) => { this.data.qpos[adr] = HOME[i]; });
    const cx = parseFloat($("cube-x").value), cy = parseFloat($("cube-y").value);
    this.data.qpos[this.cubeQadr + 0] = cx;
    this.data.qpos[this.cubeQadr + 1] = cy;
    this.data.qpos[this.cubeQadr + 2] = 0.102;
    this.data.qpos[this.cubeQadr + 3] = 1.0;
    ARM_JOINTS.forEach((_, i) => { this.data.ctrl[i] = HOME[i]; });
    if (this.jointSliders.length) this.jointSliders.forEach((s, i) => { s.value = HOME[i]; });
    m.mj_forward(this.model, this.data);
  }

  setCube() {
    const cx = parseFloat($("cube-x").value), cy = parseFloat($("cube-y").value);
    this.data.qpos[this.cubeQadr + 0] = cx;
    this.data.qpos[this.cubeQadr + 1] = cy;
    this.data.qpos[this.cubeQadr + 2] = 0.102;
    this.mujoco.mj_forward(this.model, this.data);
  }

  tipCubeDistance() {
    const site = vecToArray(this.data.site_xpos).slice(this.siteId * 3, this.siteId * 3 + 3);
    const cube = vecToArray(this.data.xpos).slice(this.cubeBodyId * 3, this.cubeBodyId * 3 + 3);
    return Math.hypot(site[0] - cube[0], site[1] - cube[1], site[2] - cube[2]);
  }

  stepFrame() {
    // advance sim time by ~1/60 s per frame (timestep = 1 ms)
    const start = Number(this.data.time);
    while (Number(this.data.time) - start < 1 / 60) {
      this.mujoco.mj_step(this.model, this.data);
    }
  }

  // -- playback -----------------------------------------------------------
  async loadPlaybackData() {
    this.datasets = { episodes: null, rollouts: null };
    try {
      const r = await fetch(EPISODES_URL);
      this.datasets.episodes = r.ok ? await r.json() : null;
    } catch (e) { /* optional */ }
    try {
      const r = await fetch(ROLLOUTS_URL);
      this.datasets.rollouts = r.ok ? await r.json() : null;
    } catch (e) { /* optional */ }
    this.refreshPlaybackUI();
  }

  refreshPlaybackUI() {
    const epSel = $("episode-select"), roSel = $("rollout-select");
    const eps = this.datasets.episodes?.episodes || [];
    const ros = this.datasets.rollouts?.rollouts || [];
    epSel.innerHTML = eps.length
      ? eps.map((e, i) => `<option value="${i}">episode ${e.id}（cube ${e.cube.join(", ")}）</option>`).join("")
      : "<option>（无数据：先跑管线）</option>";
    roSel.innerHTML = ros.length
      ? ros.map((r, i) => `<option value="${i}">rollout ${r.i}（${r.success ? "成功" : "失败"}，${r.T}步）</option>`).join("")
      : "<option>（无数据：先跑管线评估）</option>";
    $("ep-timeline").max = eps.length ? eps[0].T - 1 : 0;
    $("ro-timeline").max = ros.length ? ros[0].T - 1 : 0;
  }

  startPlayback(kind, idx) {
    const list = kind === "episode"
      ? this.datasets.episodes?.episodes || []
      : this.datasets.rollouts?.rollouts || [];
    if (!list.length) return;
    this.playback = { kind, list, idx, frame: 0, playing: true };
    this.setButtonLabel();
  }

  applyPlaybackFrame(scrub = false) {
    const pb = this.playback;
    if (!pb) return;
    const item = pb.list[pb.idx];
    const T = item.T;
    if (pb.frame >= T) pb.frame = scrub ? pb.frame : T - 1;
    const t = Math.min(pb.frame, T - 1);
    const objectTrace = item.object_qpos || [];
    const objectQpos = objectTrace.length ? objectTrace[t] : null;
    const setObjectQpos = () => {
      if (objectQpos && objectQpos.length >= 7) {
        for (let i = 0; i < 7; i++) this.data.qpos[this.cubeQadr + i] = objectQpos[i];
      } else {
        this.data.qpos[this.cubeQadr + 0] = item.cube[0];
        this.data.qpos[this.cubeQadr + 1] = item.cube[1];
        this.data.qpos[this.cubeQadr + 2] = 0.102;
        this.data.qpos[this.cubeQadr + 3] = 1.0;
      }
    };
    if (scrub) {
      // jump: set state directly
      this.jointQadr.forEach((adr, i) => { this.data.qpos[adr] = item.qpos[t][i]; });
      for (let i = 0; i < Number(this.model.nv); i++) this.data.qvel[i] = 0.0;
      ARM_JOINTS.forEach((_, i) => { this.data.ctrl[i] = item.ctrl[t][i]; });
      setObjectQpos();
      this.mujoco.mj_forward(this.model, this.data);
    } else {
      // real-time-ish: hold ctrl for one control period (20 Hz -> 50 substeps)
      ARM_JOINTS.forEach((_, i) => { this.data.ctrl[i] = item.ctrl[t][i]; });
      setObjectQpos();
      const start = Number(this.data.time);
      while (Number(this.data.time) - start < 1 / 20) this.mujoco.mj_step(this.model, this.data);
    }
    const timeline = pb.kind === "episode" ? $("ep-timeline") : $("ro-timeline");
    timeline.value = t;
    if (pb.kind === "rollout" && item.dist) {
      const lift = objectQpos ? ((objectQpos[2] - 0.102) * 100).toFixed(1) : null;
      $("hud-extra").textContent = `tip-cube 距离: ${(item.dist[t] * 100).toFixed(1)} cm / 最终 ${(item.dist[item.dist.length - 1] * 100).toFixed(1)} cm` + (lift ? ` / 抬升 ${lift} cm` : "");
    } else if (objectQpos) {
      $("hud-extra").textContent = `方块抬升: ${((objectQpos[2] - 0.102) * 100).toFixed(1)} cm`;
    } else {
      $("hud-extra").textContent = "";
    }
  }

  setButtonLabel() {
    const pb = this.playback;
    const btns = { episode: $("ep-play"), rollout: $("ro-play") };
    for (const [k, b] of Object.entries(btns)) {
      b.textContent = pb && pb.kind === k && pb.playing ? "暂停" : "播放";
    }
  }

  buildUI() {
    // joint sliders
    const wrap = $("joint-sliders");
    ARM_JOINTS.forEach((name, i) => {
      const [lo, hi] = this.jointRange[i];
      const row = document.createElement("div");
      row.className = "slider-row";
      row.innerHTML = `<label>${name.replace("_joint", "")}</label>`;
      const s = document.createElement("input");
      s.type = "range"; s.min = lo; s.max = hi; s.step = 0.001; s.value = HOME[i];
      s.addEventListener("input", () => { this.data.ctrl[i] = parseFloat(s.value); this.stopPlayback(); });
      row.appendChild(s);
      const val = document.createElement("span");
      val.className = "slider-val";
      row.appendChild(val);
      s.addEventListener("input", () => { val.textContent = parseFloat(s.value).toFixed(2); });
      val.textContent = HOME[i].toFixed(2);
      wrap.appendChild(row);
      this.jointSliders.push(s);
    });
    $("cube-x").addEventListener("input", () => { this.stopPlayback(); this.setCube(); $("cube-x").parentElement.querySelector(".slider-val").textContent = parseFloat($("cube-x").value).toFixed(2); });
    $("cube-y").addEventListener("input", () => { this.stopPlayback(); this.setCube(); $("cube-y").parentElement.querySelector(".slider-val").textContent = parseFloat($("cube-y").value).toFixed(2); });
    document.querySelectorAll("[data-mode]").forEach((b) => {
      b.onclick = () => this.setModeTab(b.dataset.mode);
    });
    $("btn-home").onclick = () => { this.stopPlayback(); this.resetSim(); };
    $("btn-pause").onclick = () => { this.paused = !this.paused; $("btn-pause").textContent = this.paused ? "继续" : "暂停"; };
    document.querySelectorAll("[data-view]").forEach((b) => {
      b.onclick = () => {
        this.viewMode = b.dataset.view;
        document.querySelectorAll("[data-view]").forEach((x) => x.classList.toggle("active", x === b));
      };
    });
    $("btn-contacts").onclick = () => {
      this.showContacts = !this.showContacts;
      const idx = this.mujoco.mjtVisFlag.mjVIS_CONTACTPOINT.value;
      this.mjvOption.flags[idx] = this.showContacts ? 1 : 0;
      $("btn-contacts").textContent = this.showContacts ? "隐藏接触点" : "显示接触点";
    };
    // playback controls
    $("ep-play").onclick = () => this.togglePlay("episode");
    $("ro-play").onclick = () => this.togglePlay("rollout");
    $("ep-select-btn").onclick = () => { this.startPlayback("episode", parseInt($("episode-select").value)); this.setModeTab("episode"); };
    $("ro-select-btn").onclick = () => { this.startPlayback("rollout", parseInt($("rollout-select").value)); this.setModeTab("rollout"); };
    $("episode-select").onchange = () => { $("ep-timeline").max = (this.datasets.episodes?.episodes || [])[$("episode-select").value].T - 1; };
    $("rollout-select").onchange = () => { $("ro-timeline").max = (this.datasets.rollouts?.rollouts || [])[$("rollout-select").value].T - 1; };
    $("ep-timeline").oninput = () => { if (this.playback?.kind === "episode") { this.playback.frame = parseInt($("ep-timeline").value); this.applyPlaybackFrame(true); } };
    $("ro-timeline").oninput = () => { if (this.playback?.kind === "rollout") { this.playback.frame = parseInt($("ro-timeline").value); this.applyPlaybackFrame(true); } };
    this.setModeTab("interactive");
  }

  setModeTab(mode) {
    document.querySelectorAll("[data-mode-panel]").forEach((p) => { p.style.display = p.dataset.modePanel === mode ? "block" : "none"; });
    document.querySelectorAll("[data-mode]").forEach((b) => b.classList.toggle("active", b.dataset.mode === mode));
    if (mode === "interactive") this.stopPlayback();
  }

  stopPlayback() {
    if (this.playback) { this.playback = null; this.setButtonLabel(); }
  }

  togglePlay(kind) {
    if (this.playback && this.playback.kind === kind) {
      this.playback.playing = !this.playback.playing;
    } else {
      const sel = kind === "episode" ? $("episode-select") : $("rollout-select");
      this.startPlayback(kind, parseInt(sel.value) || 0);
    }
    this.setButtonLabel();
  }

  // -- main loop -----------------------------------------------------------
  run() {
    const loop = () => {
      try {
        if (this.playback && this.playback.playing) {
          this.playback.frame += 1;
          this.applyPlaybackFrame(false);
          if (this.playback.frame >= this.playback.list[this.playback.idx].T - 1) {
            this.playback.playing = false;
            this.setButtonLabel();
          }
        } else if (!this.paused && !this.playback) {
          this.stepFrame();
        }
        this.updateCamera();
        this.syncScene();
        this.renderer.render(this.threeScene, this.camera);
        const d = this.tipCubeDistance();
        $("hud").textContent =
          `t = ${Number(this.data.time).toFixed(2)} s | tip-cube: ${(d * 100).toFixed(1)} cm` +
          (this.playback ? ` | 回放 ${this.playback.kind === "episode" ? "演示" : "策略"} #${this.playback.idx} 帧 ${this.playback.frame}` : "") +
          (this.paused && !this.playback ? " | 已暂停" : "");
      } catch (err) {
        console.error(err);
        window.__lastLoopErr = String(err && err.stack ? err.stack : err);
      }
      requestAnimationFrame(loop);
    };
    requestAnimationFrame(loop);
  }
}

async function main() {
  let viewer;
  try {
    setStatus("加载 MuJoCo WASM …");
    const mujoco = await loadMujoco();
    setStatus("初始化 …");
    viewer = new Viewer(mujoco);
    await viewer.loadModel();
  } catch (err) {
    console.error(err);
    setStatus(`模型加载失败：${errorMessage(err)}`);
    return;
  }
  viewer.loadPlaybackData();
  viewer.run();
  setStatus("就绪");
  document.querySelectorAll("button, input, select").forEach((el) => (el.disabled = false));
}

main();
