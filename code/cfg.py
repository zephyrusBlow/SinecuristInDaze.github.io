import os
from dataclasses import dataclass, field

from motrix_envs import registry
from motrix_envs.base import EnvCfg

# 指向复杂地形的 XML 文件
# 确保该路径下存在 scene_rough_terrain.xml，通常在 locomotion/go1/xmls 目录下可以找到参考
model_file = os.path.join(os.path.dirname(__file__), "xmls", "scene_rough_terrain.xml")

@dataclass
class NoiseConfig:
    level: float = 0.2
    scale_joint_angle: float = 0.03
    scale_joint_vel: float = 1.5
    scale_gyro: float = 0.2
    scale_gravity: float = 0.05
    scale_linvel: float = 0.1
    # 新增：地形感知噪声
    scale_height: float = 0.005

@dataclass
class ControlConfig:
    # 复杂地形可能需要稍大的 action_scale 来提供足够的力量，保持 0.06 或微调至 0.08
    action_scale: float = 0.06

@dataclass
class InitState:
    # 复杂地形初始高度建议稍微调高一点（0.56 -> 0.65），防止出生时卡入地面
    pos: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.65])
    pos_randomization_range: list[float] = field(default_factory=lambda: [-5.0, -5.0, 5.0, 5.0])

    default_joint_angles: dict[str, float] = field(
        default_factory=lambda: {
            "LF_HAA": 0.0, "RF_HAA": 0.0, "LH_HAA": 0.0, "RH_HAA": 0.0,
            "LF_HFE": 0.4, "RF_HFE": 0.4, "LH_HFE": -0.4, "RH_HFE": -0.4,
            "LF_KFE": -0.8, "RF_KFE": -0.8, "LH_KFE": 0.8, "RH_KFE": 0.8,
        }
    )

@dataclass
class Commands:
    pose_command_range: list[float] = field(default_factory=lambda: [-5.0, -5.0, -3.14, 5.0, 5.0, 3.14])
    position_gain: float = 1.0
    yaw_gain: float = 1.0
    yaw_deadband_deg: float = 8.0
    max_command: float = 1.0

@dataclass
class Normalization:
    lin_vel: float = 2.0
    ang_vel: float = 0.25
    dof_pos: float = 1.0
    dof_vel: float = 0.05
    # 新增：地形高度归一化系数
    height_map: float = 5.0

@dataclass
class Asset:
    body_name: str = "base"
    foot_names: list[str] = field(default_factory=lambda: ["LF_FOOT", "RF_FOOT", "LH_FOOT", "RH_FOOT"])
    terminate_after_contacts_on: list[str] = field(default_factory=lambda: ["base", "hatch", "shell"])
    # 确保此处地磁平面名称与 XML 对应
    ground_name: str = "floor"

@dataclass
class Sensor:
    base_linvel: str = "base_linvel"
    base_gyro: str = "base_gyro"

@dataclass
class RewardConfig:
    # 复杂地形惩罚需更严厉
    termination_penalty: float = -50.0 

    tracking_lin_weight: float = 1.5
    tracking_ang_weight: float = 0.5 # 增加旋转跟踪权重
    approach_scale: float = 4.0
    approach_clip: float = 1.0

    stop_bonus_scale: float = 2.0
    zero_ang_bonus: float = 6.0
    arrival_bonus: float = 15.0 # 提高到达奖励

    # 惩罚项
    lin_vel_z_penalty: float = -2.0
    ang_vel_xy_penalty: float = -0.05
    orientation_penalty: float = -0.1 # 增加姿态惩罚，鼓励背部水平
    torque_penalty: float = -2e-5    # 增加功耗惩罚，防止关节抖动
    dof_vel_penalty: float = -1e-4
    action_rate_penalty: float = -0.01 # 显著增加动作平滑惩罚

# 注册为 rough 任务
@registry.envcfg("anymal_c_navigation_rough-v0")
@dataclass
class AnymalCEnvCfg(EnvCfg):
    model_file: str = model_file
    reset_noise_scale: float = 0.02 # 增加初始扰动，提高鲁棒性
    max_episode_seconds: float = 10.0 # 复杂地形跑得慢，增加时长
    sim_dt: float = 0.01
    ctrl_dt: float = 0.02 # 建议将控制周期设为 50Hz (0.02)，有助于策略学习

    max_dof_vel: float = 80.0 # 稍微降低阈值，保护电机

    noise_config: NoiseConfig = field(default_factory=NoiseConfig)
    control_config: ControlConfig = field(default_factory=ControlConfig)
    reward_config: RewardConfig = field(default_factory=RewardConfig)
    init_state: InitState = field(default_factory=InitState)
    commands: Commands = field(default_factory=Commands)
    normalization: Normalization = field(default_factory=Normalization)
    asset: Asset = field(default_factory=Asset)
    sensor: Sensor = field(default_factory=Sensor)


# --- RL 相关配置 ---

from motrix_rl.registry import rlcfg
from motrix_rl.base import BaseRLCfg

@rlcfg("anymal_c_navigation_rough-v0", backend="torch")
class AnymalCRLCfg(BaseRLCfg):
    def __init__(self):
        super().__init__()
        self.seed = 42
        self.num_envs = 2048 # 复杂地形建议保持大规模并行
        
        # PPO 算法参数微调
        self.max_iterations = 30000 
        self.rollout_steps = 24 
        self.mini_batches = 4
        self.epochs = 5
        self.learning_rate = 3e-4
        self.grad_norm_clip = 1.0
        # 复杂地形可以考虑加入更长的 discount factor (gamma)