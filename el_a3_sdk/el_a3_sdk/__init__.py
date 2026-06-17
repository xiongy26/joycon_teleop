from el_a3_sdk.interface import ELA3Interface
from el_a3_sdk.arm_manager import ArmManager
from el_a3_sdk.data_types import (
    MotorFeedback,
    ArmStatus,
    ArmJointStates,
    ArmEndPose,
    MotorHighSpdInfo,
    MotorLowSpdInfo,
    MotorAngleLimitMaxVel,
    DynamicsInfo,
    TrajectoryResult,
)
from el_a3_sdk.protocol import (
    MotorType,
    RunMode,
    ControlMode,
    MoveMode,
    ArmState,
    LogLevel,
)

__version__ = "1.0.0"


def get_kinematics():
    """延迟导入 ELA3Kinematics（避免无 pinocchio 环境下 import 失败）"""
    from el_a3_sdk.kinematics import ELA3Kinematics
    return ELA3Kinematics


def get_slcan_driver():
    """延迟导入 SlcanCanDriver（避免无 pyserial 环境下 import 失败）"""
    from el_a3_sdk.slcan_can_driver import SlcanCanDriver
    return SlcanCanDriver


__all__ = [
    "ELA3Interface",
    "ArmManager",
    "get_kinematics",
    "get_slcan_driver",
    "MotorFeedback",
    "ArmStatus",
    "ArmJointStates",
    "ArmEndPose",
    "MotorHighSpdInfo",
    "MotorLowSpdInfo",
    "MotorAngleLimitMaxVel",
    "DynamicsInfo",
    "TrajectoryResult",
    "MotorType",
    "RunMode",
    "ControlMode",
    "MoveMode",
    "ArmState",
    "LogLevel",
]
