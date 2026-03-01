# IssacLab & MotrixLab
> Isaac-Navigation-Flat-Anymal-C-v0

## 镜像 mirror
选择镜像的时候，选择xbotics-full-v2.5。这个镜像为大家下载好了isaaclab，motrixlab项目以及相关的环境依赖，还安装了vs code。

## 第 1 周：熟悉平台 & 跑通基础任务
> 目标：搞懂 IsaacLab / MotrixLab 的基本使用方式，跑通示例任务
### IsaacLab 
- IsaacLab仓库目录结构
IsaacLab/
    1. source/isaaclab/ 放核心代码
    - tasks/ 
    改 reward、obs、termination 的地方
    - envs/ 
    定义机器人关节、初始姿态、物理参数、DOF 顺序
    - robots/ 
        环境封装
    - controllers/ 
    强化学习输出 action 后，controller决定怎么作用到物理引擎上，包含PD 控制、Torque 控制、Position 控制
    - sensors/
    传感器模拟出observation
    - utils/
    常用的模块放这里
    - algorithms/
    放algorithms
    2. scripts/ 真正运行的地方
    - train.py
    play.py
    evaluate.py
    读取 cfg、创建 env、启动 RL 算法
    3. cfg/ 不改代码就能改训练参数
    - task/
    - robot/
    - train/
    4. assets/
    5. logs/
    6. extensions/
    7. docs/
- Navigation-Flat-Anymal-C-v0任务相关的文件结构


### MotrixLab
- motrixlab仓库目录结构

- 了解 GO1 环境的结构（因为后面迁移 Anymal C）

### 实操任务
- 在 IsaacLab 训练和推理
  - Navigation-Flat-Anymal-C-v0
      - train:
          ./isaaclab.sh-pscripts/reinforcement_learning/skrl/train.py--taskIsaac-Navigation-Flat-Anymal-C-v0--headless
      - test:
          ./isaaclab.sh-pscripts/reinforcement_learning/skrl/play.py--taskIsaac-Navigation-Flat-Anymal-C-v0--num_envs64
  ![isaac_navigation_test](images/isaac_navigation_test)

- 在 MotrixLab 训练和推理
  - Unitree GO1 示例任务
      - flat
    1. 环境预览
        uv run scripts/view.py --env go1-flat-terrain-walk
    2. 开始训练
        uv run scripts/train.py --env go1-flat-terrain-walk
    3. 查看训练进度
        uv run tensorboard --logdir runs/go1-flat-terrain-walk
    4. 测试训练结果
        uv run scripts/play.py --env go1-flat-terrain-walk
![images/motrix_unitree_go1_test_plain](images/motrix_unitree_go1_test_plain)
    - rough & stairs
    0. 环境设置
flat:
```
<worldbody>
    <light pos="0 0 1.5" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="0 0 0.01" type="plane" material="groundplane" contype="1" conaffinity="0" priority="1"
      friction="0.6" condim="3"/>
</worldbody>
```
rough:
```<geom name="floor" type="hfield" pos="0 0 0" size="10 10 1" hfield="rough_heightmap.png" material="groundplane" contype="1" conaffinity="0" friction="0.6"/>```
box:
```<geom type="box" pos="0.5 0 0.05" size="0.1 0.1 0.05" material="groundplane"/>
<geom type="box" pos="-0.5 0.2 0.05" size="0.2 0.2 0.05" material="groundplane"/>
```
stairs:
```<geom type="box" pos="0.0 0.0 0.05" size="0.5 1 0.05"/>
<geom type="box" pos="0.5 0.0 0.15" size="0.5 1 0.05"/>
<geom type="box" pos="1.0 0.0 0.25" size="0.5 1 0.05"/>
```
   1. 环境预览
    uv run scripts/view.py --env go1-rough-terrain-walk
    uv run scripts/view.py --env go1-stairs-terrain-walk
    2. 开始训练
    uv run scripts/train.py --env go1-rough-terrain-walk
    uv run scripts/train.py --env go1-stairs-terrain-walk
    3. 查看训练进度
    uv run tensorboard --logdir runs/go1-rough-terrain-walk
    4. 测试训练结果
    由于粗糙地形场景中同时生成了了一个无限大平面和一个崎岖地形高度场，测试训练结果时会仿照训练过程先将智能体生成在平面上，完成一轮行走后再生成到崎岖地形上。用户需要主动调整相机视角和位置来观察智能体的状态。
    uv run scripts/play.py --env go1-rough-terrain-walk
    uv run scripts/play.py --env go1-stairs-terrain-walk
    - Go1WalkNpEnvCfg 配置文件
```# Copyright (C) 2020-2025 Motphys Technology Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os
from dataclasses import dataclass, field

from motrix_envs import registry
from motrix_envs.base import EnvCfg

model_file = os.path.dirname(__file__) + "/xmls/scene_motor_actuator.xml"
#📌指定机器人模型xml文件

@dataclass
class NoiseConfig: #📌控制传感器/关节数据的噪声去模拟现实中的噪声
    level: float = 1.0 #📌 总噪声强度
    scale_joint_angle: float = 0.03
    scale_joint_vel: float = 1.5
    scale_gyro: float = 0.2
    scale_gravity: float = 0.05
    scale_linvel: float = 0.1


@dataclass
class ControlConfig:
    stiffness = 80  # [N*m/rad]
    damping = 1  # [N*m*s/rad]
    # action scale: target angle = actionScale * action + defaultAngle
    action_scale = 0.05


@dataclass
class InitState:
    # the initial position of the robot in the world frame
    pos = [0.0, 0.0, 0.42]

    # the default angles for all joints. key = joint name, value = target angle [rad]
    default_joint_angles = {
        "FL_hip": 0.0,  # [rad]
        "RL_hip": 0.0,  # [rad]
        "FR_hip": -0.0,  # [rad]
        "RR_hip": -0.0,  # [rad]
        "FL_thigh": 0.9,  # [rad]
        "RL_thigh": 0.9,  # [rad]
        "FR_thigh": 0.9,  # [rad]
        "RR_thigh": 0.9,  # [rad]
        "FL_calf": -1.8,  # [rad]
        "RL_calf": -1.8,  # [rad]
        "FR_calf": -1.8,  # [rad]
        "RR_calf": -1.8,  # [rad]
    }


@dataclass
class Commands: #📌 指定机器人可以接受的线速度和角速度的上下限
    vel_limit = [
        [0.0, -1.0, -1.0],  # min: vel_x [m/s], vel_y [m/s], ang_vel [rad/s]
        [2.0, 1.0, 1.0],  # max
    ]


@dataclass
class Normalization: #📌 状态信息（速度、角度等）进行归一化，使训练更稳定
    lin_vel = 2
    ang_vel = 0.25
    dof_pos = 1
    dof_vel = 0.05


@dataclass
class Asset:
    body_name = "trunk"
    foot_name = "foot"
    penalize_contacts_on = ["thigh", "calf"]
    terminate_after_contacts_on = [
        "trunk",
    ]
    ground = "floor"


@dataclass
class Sensor:
    local_linvel = "local_linvel"
    gyro = "gyro"


@dataclass
class RewardConfig:
    scales: dict[str, float] = field(
        default_factory=lambda: {
            "termination": -0.0,
            "tracking_lin_vel": 1.0,
            "tracking_ang_vel": 0.5,
            "lin_vel_z": -2.0,
            "ang_vel_xy": -0.05,
            "orientation": -0.0,
            "torques": -0.00001,
            "dof_vel": -0.0,
            "dof_acc": -2.5e-7,
            "base_height": -0.0,
            "feet_air_time": 1.0,
            "collision": -1.0 * 0,
            "feet_stumble": -0.0,
            "action_rate": -0.001,
            "stand_still": -0.0,
            "hip_pos": -1,
            "calf_pos": -0.3 * 0,
        }
    )

    tracking_sigma: float = 0.25
    max_foot_height: float = 0.1


@registry.envcfg("go1-flat-terrain-walk")
@dataclass
class Go1WalkNpEnvCfg(EnvCfg):
    max_episode_seconds: float = 20.0
    model_file: str = model_file
    noise_config: NoiseConfig = field(default_factory=NoiseConfig)
    control_config: ControlConfig = field(default_factory=ControlConfig)
    reward_config: RewardConfig = field(default_factory=RewardConfig)
    init_state: InitState = field(default_factory=InitState)
    commands: Commands = field(default_factory=Commands)
    normalization: Normalization = field(default_factory=Normalization)
    asset: Asset = field(default_factory=Asset)
    sensor: Sensor = field(default_factory=Sensor)
    sim_dt: float = 0.01
    ctrl_dt: float = 0.01
```
- Go1WalkTask 环境实现
```
# Copyright (C) 2020-2025 Motphys Technology Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import gymnasium as gym
import motrixsim as mtx
import numpy as np

from motrix_envs import registry
from motrix_envs.locomotion.go1.cfg import Go1WalkNpEnvCfg
from motrix_envs.np.env import NpEnv, NpEnvState


## provide quat math utility from motrixsim.
def quat_rotate_inverse(quats, v):
    """
    Rotate a fixed vector v by a list of quaternions using a vectorized approach.

    Parameters:
        quats (np.ndarray): Array of quaternions of shape (N, 4). Each quaternion is in [w, x, y, z] format.
        v (np.ndarray): Fixed vector of shape (3,) to be rotated.

    Returns:
        np.ndarray: Array of rotated vectors of shape (N, 3).
    """
    # Normalize the quaternions to ensure they are unit quaternions

    # Extract the scalar (w) and vector (x, y, z) parts of the quaternions
    w = quats[:, -1]  # Shape (N,)
    im = quats[:, :3]  # Shape (N, 3)

    # Compute the cross product between the imaginary part of each quaternion and the fixed vector v.
    # np.cross broadcasts v to match each row in im, resulting in an array of shape (N, 3)
    cross_im_v = np.cross(im, v)

    # Compute the intermediate terms for the rotation formula:
    term1 = w[:, np.newaxis] * cross_im_v  # w * cross(im, v)
    term2 = np.cross(im, cross_im_v)  # cross(im, cross(im, v))

    # Apply the rotation formula: v_rot = v + 2 * (term1 + term2)
    v_rotated = v + 2 * (term1 + term2)

    return v_rotated


@registry.env("go1-flat-terrain-walk", sim_backend="np")
class Go1WalkTask(NpEnv):
    _init_dof_pos: np.ndarray
    _init_dof_vel: np.ndarray

    def __init__(self, cfg: Go1WalkNpEnvCfg, num_envs=1): 
        super().__init__(cfg, num_envs)
        self._init_action_space()
        self._init_obs_space()
        self._body = self._model.get_body(self.cfg.asset.body_name)
        self._num_action = self._action_space.shape[0]
        self._num_observation = self._observation_space.shape[0]
        self._num_dof_pos = self._model.num_dof_pos
        self._num_dof_vel = self._model.num_dof_vel

        self._init_dof_vel = np.zeros(
            (self._num_dof_vel,),
            dtype=np.float32,
        )
        self._init_dof_pos = self._model.compute_init_dof_pos()
        self._init_buffer()

    def _init_obs_space(self):
        model = self.model
        num_dof_vel = model.num_dof_vel  # linvel + gyro + joint_vel
        num_joint_angle = model.num_dof_pos - 7
        num_gravity = 3
        num_actions = model.num_actuators
        num_command = 3

        num_obs = num_dof_vel + num_joint_angle + num_gravity + num_actions + num_command
        assert num_obs == 48

        self._observation_space = gym.spaces.Box(-np.inf, np.inf, (num_obs,), dtype=np.float32)

    def _init_action_space(self):
        model = self.model
        self._action_space = gym.spaces.Box(
            np.array(model.actuator_ctrl_limits[0, :]),
            np.array(model.actuator_ctrl_limits[1, :]),
            (model.num_actuators,),
            dtype=np.float32,
        )

    @property
    def action_space(self) -> gym.spaces.Box:
        return self._action_space

    @property
    def observation_space(self) -> gym.spaces.Box:
        return self._observation_space

    def get_dof_pos(self, data: mtx.SceneModel):
        return self._body.get_joint_dof_pos(data)

    def get_dof_vel(self, data: mtx.SceneModel):
        return self._body.get_joint_dof_vel(data)

    def _init_buffer(self):
        cfg = self._cfg
        assert isinstance(cfg, Go1WalkNpEnvCfg)
        # init buffers

        self.reset_buf = np.ones(self._num_envs, dtype=np.bool)
        self.kps = np.ones(self._num_action, dtype=np.float32) * cfg.control_config.stiffness
        self.kds = np.ones(self._num_action, dtype=np.float32) * cfg.control_config.damping
        self.gravity_vec = np.array([0, 0, -1], dtype=np.float32)
        self.commands_scale = np.array(
            (
                [
                    cfg.normalization.lin_vel,
                    cfg.normalization.lin_vel,
                    cfg.normalization.ang_vel,
                ]
            ),
            dtype=np.float32,
        )

        self.default_angles = np.zeros(self._num_action, dtype=np.float32)
        self.hip_indices = []
        self.calf_indices = []
        for i in range(self._model.num_actuators):
            for name in cfg.init_state.default_joint_angles.keys():
                if name in self._model.actuator_names[i]:
                    self.default_angles[i] = cfg.init_state.default_joint_angles[name]
            if "hip" in self._model.actuator_names[i]:
                self.hip_indices.append(i)
            if "calf" in self._model.actuator_names[i]:
                self.calf_indices.append(i)

        self._init_dof_pos[-self._num_action :] = self.default_angles

        self.ground = self._model.get_geom_index(cfg.asset.ground)
        self.termination_contact = None
        self.foot = []
        for name in cfg.asset.terminate_after_contacts_on:
            if self.termination_contact is None:
                self.termination_contact = np.array([[self._model.get_geom_index(name), self.ground]], dtype=np.uint32)
            else:
                self.termination_contact = np.append(
                    self.termination_contact,
                    np.array(
                        [[self._model.get_geom_index(name), self.ground]],
                        dtype=np.uint32,
                    ),
                    axis=0,
                )
        for name in cfg.asset.foot_name:
            self.foot.append([self._model.get_geom_index(name), self.ground])
        self.num_check = self.termination_contact.shape[0]

        self.foot = None
        for i in self._model.geom_names:
            if i is not None and cfg.asset.foot_name in i:
                if self.foot is None:
                    self.foot = np.array([[self._model.get_geom_index(i), self.ground]], dtype=np.uint32)
                else:
                    self.foot = np.append(
                        self.foot,
                        np.array(
                            [[self._model.get_geom_index(i), self.ground]],
                            dtype=np.uint32,
                        ),
                        axis=0,
                    )
        self.foot_check_num = self.foot.shape[0]
        self.foot_check = self.foot

        self.termination_check = self.termination_contact

    def apply_action(self, actions, state):
        state.info["last_dof_vel"] = self.get_dof_vel(state.data)
        state.info["last_actions"] = state.info["current_actions"]
        state.info["current_actions"] = actions
        state.data.actuator_ctrls = self._compute_torques(actions, state.data)
        return state

    def _compute_torques(self, actions, data):
        # Compute torques from actions.
        # pd controller
        actions_scaled = actions * self.cfg.control_config.action_scale
        torques = self.kps * (
            actions_scaled + self.default_angles - self.get_dof_pos(data)
        ) - self.kds * self.get_dof_vel(data)
        return torques

    def get_local_linvel(self, data: mtx.SceneData) -> np.ndarray:
        return self._model.get_sensor_value(self.cfg.sensor.local_linvel, data)

    def get_gyro(self, data: mtx.SceneData) -> np.ndarray:
        return self._model.get_sensor_value(self.cfg.sensor.gyro, data)

    def update_state(self, state):
        state = self.update_observation(state) 
        state = self.update_terminated(state) #📌 检查机器人是否碰撞、跌倒，决定 episode 是否结束
        state = self.update_reward(state) #📌 计算奖励
        return state

    def _get_obs(self, data: mtx.SceneData, info: dict) -> np.ndarray:
        linear_vel = self.get_local_linvel(data)
        gyro = self.get_gyro(data)
        pose = self._body.get_pose(data)
        base_quat = pose[:, 3:7]
        local_gravity = quat_rotate_inverse(base_quat, self.gravity_vec)
        diff = self.get_dof_pos(data) - self.default_angles
        noisy_linvel = linear_vel * self.cfg.normalization.lin_vel
        noisy_gyro = gyro * self.cfg.normalization.ang_vel
        noisy_joint_angle = diff * self.cfg.normalization.dof_pos
        noisy_joint_vel = self.get_dof_vel(data) * self.cfg.normalization.dof_vel
        command = info["commands"] * self.commands_scale
        last_actions = info["current_actions"]

        obs = np.hstack(
            [
                noisy_linvel,
                noisy_gyro,
                local_gravity,
                noisy_joint_angle,
                noisy_joint_vel,
                last_actions,
                command,
            ]
        )
        return obs

    def update_observation(self, state: NpEnvState):
        data = state.data
        obs = self._get_obs(data, state.info)
        cquerys = self._model.get_contact_query(data)
        foot_contact = cquerys.is_colliding(self.foot_check)
        state.info["contacts"] = foot_contact.reshape((self._num_envs, self.foot_check_num))
        state.info["feet_air_time"] = self.update_feet_air_time(state.info)
        return state.replace(obs=obs)

    def update_terminated(self, state: NpEnvState) -> NpEnvState:
        data = state.data
        cquerys = self._model.get_contact_query(data)
        termination_check = cquerys.is_colliding(self.termination_check)
        termination_check.reshape((self._num_envs, self.num_check))
        terminated = termination_check.any(axis=1)

        return state.replace(
            terminated=terminated,
        )

    def update_feet_air_time(self, info: dict):
        feet_air_time = info["feet_air_time"]
        feet_air_time += self.cfg.ctrl_dt
        feet_air_time *= ~info["contacts"]
        return feet_air_time

    def resample_commands(self, num_envs: int):
        commands = np.random.uniform(
            low=self.cfg.commands.vel_limit[0],
            high=self.cfg.commands.vel_limit[1],
            size=(num_envs, 3),
        )
        return commands

    def update_reward(self, state: NpEnvState) -> NpEnvState:
        data = state.data
        terminated = state.terminated

        reward_dict = self._get_reward(data, state.info)

        rewards = {k: v * self.cfg.reward_config.scales[k] for k, v in reward_dict.items()}
        rwd = sum(rewards.values())
        rwd = np.clip(rwd, 0.0, 10000.0)
        if "termination" in self.cfg.reward_config.scales:
            termination = self._reward_termination(terminated) * self.cfg.reward_config.scales["termination"]
            rwd += termination

        rwd = np.where(terminated, np.array(0.0), rwd)

        return state.replace(reward=rwd)

    def reset(self, data) -> tuple[np.ndarray, dict]:
        num_reset = data.shape[0]

        dof_pos = np.tile(self._init_dof_pos, (num_reset, 1))
        dof_vel = np.tile(self._init_dof_vel, (num_reset, 1))

        data.reset(self._model)
        data.set_dof_vel(dof_vel)
        data.set_dof_pos(dof_pos, self._model)
        self._model.forward_kinematic(data)

        info = {
            "current_actions": np.zeros((num_reset, self._num_action), dtype=np.float32),
            "last_actions": np.zeros((num_reset, self._num_action), dtype=np.float32),
            "commands": self.resample_commands(num_reset),
            "last_dof_vel": np.zeros((num_reset, self._num_action), dtype=np.float32),
            "feet_air_time": np.zeros((num_reset, self.foot_check_num), dtype=np.float32),
            "contacts": np.zeros((num_reset, self.foot_check_num), dtype=np.bool),
        }
        obs = self._get_obs(data, info)
        return obs, info

    def _get_reward(
        self,
        data: mtx.SceneData,
        info: dict,
    ) -> dict[str, np.ndarray]:
        commands = info["commands"]
        return {
            "lin_vel_z": self._reward_lin_vel_z(data),
            "ang_vel_xy": self._reward_ang_vel_xy(data),
            "orientation": self._reward_orientation(data),
            "torques": self._reward_torques(data),
            "dof_vel": self._reward_dof_vel(data),
            "dof_acc": self._reward_dof_acc(data, info),
            "action_rate": self._reward_action_rate(info),
            "tracking_lin_vel": self._reward_tracking_lin_vel(data, commands),
            "tracking_ang_vel": self._reward_tracking_ang_vel(data, commands),
            "stand_still": self._reward_stand_still(data, commands),
            "hip_pos": self._reward_hip_pos(data, commands),
            "calf_pos": self._reward_calf_pos(data, commands),
            "feet_air_time": self._reward_feet_air_time(commands, info),
        }

    # ------------ reward functions----------------
    def _reward_lin_vel_z(self, data):
        # Penalize z axis base linear velocity
        return np.square(self.get_local_linvel(data)[:, 2])

    def _reward_ang_vel_xy(self, data):
        # Penalize xy axes base angular velocity
        return np.sum(np.square(self.get_gyro(data)[:, :2]), axis=1)

    def _reward_orientation(self, data):
        # Penalize non flat base orientation
        pose = self._body.get_pose(data)
        base_quat = pose[:, 3:7]
        gravity = quat_rotate_inverse(base_quat, self.gravity_vec)
        return np.sum(np.square(gravity[:, :2]), axis=1)

    def _reward_torques(self, data: mtx.SceneData):
        # Penalize torques
        return np.sum(np.square(data.actuator_ctrls), axis=1)

    def _reward_dof_vel(self, data):
        # Penalize dof velocities
        return np.sum(np.square(self.get_dof_vel(data)), axis=1)

    def _reward_dof_acc(self, data, info):
        # Penalize dof accelerations
        return np.sum(
            np.square((info["last_dof_vel"] - self.get_dof_vel(data)) / self.cfg.ctrl_dt),
            axis=1,
        )

    def _reward_action_rate(self, info: dict):
        # Penalize changes in actions
        action_diff = info["current_actions"] - info["last_actions"]
        return np.sum(np.square(action_diff), axis=1)

    def _reward_termination(self, done):
        # Terminal reward / penalty
        return done

    def _reward_feet_air_time(self, commands: np.ndarray, info: dict):
        # Reward long steps
        feet_air_time = info["feet_air_time"]
        first_contact = (feet_air_time > 0.0) * info["contacts"]
        # reward only on first contact with the ground
        rew_airTime = np.sum((feet_air_time - 0.5) * first_contact, axis=1)
        # no reward for zero command
        rew_airTime *= np.linalg.norm(commands[:, :2], axis=1) > 0.1
        return rew_airTime

    def _reward_tracking_lin_vel(self, data, commands: np.ndarray):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = np.sum(np.square(commands[:, :2] - self.get_local_linvel(data)[:, :2]), axis=1)
        return np.exp(-lin_vel_error / self.cfg.reward_config.tracking_sigma)

    def _reward_tracking_ang_vel(self, data, commands: np.ndarray):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = np.square(commands[:, 2] - self.get_gyro(data)[:, 2])
        return np.exp(-ang_vel_error / self.cfg.reward_config.tracking_sigma)

    def _reward_stand_still(self, data, commands: np.ndarray):
        # Penalize motion at zero commands
        return np.sum(np.abs(self.get_dof_pos(data) - self.default_angles), axis=1) * (
            np.linalg.norm(commands, axis=1) < 0.1
        )

    def _reward_hip_pos(self, data, commands: np.ndarray):
        return (0.8 - np.abs(commands[:, 1])) * np.sum(
            np.square(self.get_dof_pos(data)[:, self.hip_indices] - self.default_angles[self.hip_indices]),
            axis=1,
        )

    def _reward_calf_pos(self, data, commands: np.ndarray):
        return (0.8 - np.abs(commands[:, 1])) * np.sum(
            np.square(self.get_dof_pos(data)[:, self.calf_indices] - self.default_angles[self.calf_indices]),
            axis=1,
        )
```
