from dataclasses import dataclass, field
from .cfg import AnymalCEnvCfg # 继承你之前的平地配置

@dataclass
class AnymalCRoughCfg(AnymalCEnvCfg):
    
    xml_path: str = "MotrixLab/motrix_envs/src/motrix_envs/locomotion/go1/xmls/scene_rough_terrain.xml"
    
    # 模仿 Go1，机器人需要感知周围 1.6m x 1.0m 范围内的地面高度
    @dataclass
    class TerrainCfg:
        num_rows: int = 10
        num_cols: int = 20
        selected: bool = True

        measured_points_x: list = field(default_factory=lambda: [x * 0.1 for x in range(-8, 9)]) 
        measured_points_y: list = field(default_factory=lambda: [y * 0.1 for y in range(-5, 6)])

    # 奖励函数增加惩罚、适应rough路面
    @dataclass
    class RewardCfg(AnymalCEnvCfg.RewardCfg):
        # 踩空或踢到障碍物
        feet_air_time_weight: float = 0.5    # 奖励正常的步态摆动
        foot_collision_penalty: float = -1.0 # 惩罚脚部侧向撞击地形
        hip_pos_penalty: float = -1.0        # 惩罚大腿根部过低（防止卡住）

    terrain = TerrainCfg()
    reward_config = RewardCfg()