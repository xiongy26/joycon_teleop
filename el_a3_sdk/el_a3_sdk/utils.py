"""
EL-A3 SDK 工具函数

数据映射转换、单位换算等。
"""

import math


def float_to_uint16(x: float, x_min: float, x_max: float) -> int:
    """浮点数线性映射到 uint16 (0~65535)"""
    x = max(x_min, min(x_max, x))
    return int((x - x_min) * 65535.0 / (x_max - x_min))


def uint16_to_float(x_int: int, x_min: float, x_max: float) -> float:
    """uint16 (0~65535) 线性映射到浮点数"""
    return x_int * (x_max - x_min) / 65535.0 + x_min


def rad_to_deg(rad: float) -> float:
    return rad * 180.0 / math.pi


def deg_to_rad(deg: float) -> float:
    return deg * math.pi / 180.0


def clamp(value: float, min_val: float, max_val: float) -> float:
    return max(min_val, min(max_val, value))


def euler_to_quat(rx: float, ry: float, rz: float):
    """XYZ intrinsic Euler angles (rad) -> quaternion (w, x, y, z)"""
    cx, sx = math.cos(rx / 2), math.sin(rx / 2)
    cy, sy = math.cos(ry / 2), math.sin(ry / 2)
    cz, sz = math.cos(rz / 2), math.sin(rz / 2)
    w = cx * cy * cz - sx * sy * sz
    x = sx * cy * cz + cx * sy * sz
    y = cx * sy * cz - sx * cy * sz
    z = cx * cy * sz + sx * sy * cz
    return (w, x, y, z)


def quat_to_euler(w: float, x: float, y: float, z: float):
    """Quaternion (w, x, y, z) -> XYZ intrinsic Euler angles (rad)"""
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    rx = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    sinp = clamp(sinp, -1.0, 1.0)
    ry = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    rz = math.atan2(siny_cosp, cosy_cosp)

    return (rx, ry, rz)


def slerp_euler(rx0: float, ry0: float, rz0: float,
                rx1: float, ry1: float, rz1: float,
                t: float):
    """Spherical linear interpolation between two sets of Euler angles via quaternion SLERP."""
    q0 = euler_to_quat(rx0, ry0, rz0)
    q1 = euler_to_quat(rx1, ry1, rz1)

    dot = q0[0] * q1[0] + q0[1] * q1[1] + q0[2] * q1[2] + q0[3] * q1[3]

    if dot < 0.0:
        q1 = (-q1[0], -q1[1], -q1[2], -q1[3])
        dot = -dot

    dot = clamp(dot, -1.0, 1.0)

    if dot > 0.9995:
        result = tuple(q0[i] + t * (q1[i] - q0[i]) for i in range(4))
    else:
        theta = math.acos(dot)
        sin_theta = math.sin(theta)
        s0 = math.sin((1.0 - t) * theta) / sin_theta
        s1 = math.sin(t * theta) / sin_theta
        result = tuple(s0 * q0[i] + s1 * q1[i] for i in range(4))

    norm = math.sqrt(sum(c * c for c in result))
    if norm > 1e-10:
        result = tuple(c / norm for c in result)

    return quat_to_euler(*result)
