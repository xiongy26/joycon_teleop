# EL-A3 Motion Deployment

`ela3_motion` is the reusable motion deployment layer between planners
and `el_a3_sdk`.

It accepts a 6-axis joint trajectory from MuJoCo or any other planner, prepares
it for real EL-A3 execution, and optionally submits the resulting SDK queue.
It does not parse text/SVG, run IK, or implement CAN motor drivers.

```text
planner / MotionPlan / joint_plan
  -> ela3_motion
  -> start bridge + densify + validate + SDK queue
  -> el_a3_sdk
  -> CAN / EL-A3
```

## Module Boundaries

- `types.py`: public data types such as `ELA3Trajectory`, `PreparedTrajectory`,
  `QueueBuildResult`, and `MotionExecutionResult`.
- `adapter.py`: adapters from `MotionPlan` or raw `joint_plan` to
  `ELA3Trajectory`.
- `bridge.py`: start-state bridging from real feedback to the first target.
- `safety.py`: trajectory densification and joint-limit/step validation.
- `queue.py`: conversion from dense joint plans to EL-A3 SDK queue points.
- `deployment.py`: pure preparation pipeline from `ELA3Trajectory` to
  `PreparedTrajectory`.
- `backend.py`: EL-A3 SDK connection, enable, feedback, queue submission, and
  hold behavior.
- `realtime.py`: realtime joint target deployment for teleop and GUI adapters.
- `manager.py`: high-level orchestration that combines planner output,
  deployment preparation, backend execution, and finish policy.
- `trajectory.py`: compatibility aggregate for older imports.

## Minimal Use

```python
import numpy as np

from ela3_motion import ELA3MotionManager, ELA3MotionOptions

joint_plan = [
    np.zeros(6),
    np.array([0.02, 0.0, 0.0, 0.0, 0.0, 0.0]),
]

manager = ELA3MotionManager(ELA3MotionOptions(dry_run=True))
result = manager.execute_joint_plan(joint_plan, dt=0.02, dry_run=True)
```

For offline queue preparation without connecting hardware, call
`prepare_trajectory_for_deployment()` directly.

## Realtime Joint Targets

Realtime callers such as teleop loops should generate a 6-axis `q_target` and
let `ela3_motion` handle feedback, safety checks, densification, queue building,
and backend submission:

```python
import numpy as np

from ela3_motion import ELA3MotionOptions, ELA3RealtimeController

controller = ELA3RealtimeController(ELA3MotionOptions(command_rate_hz=50.0))
controller.connect()
controller.submit_joint_target(np.zeros(6), dt=0.02, block=False)
```

The realtime API still sits below Joy-Con, GUI, Cartesian IK, and MuJoCo. Those
layers produce joint targets; `ela3_motion` deploys them safely; `el_a3_sdk`
owns motor and CAN communication.
