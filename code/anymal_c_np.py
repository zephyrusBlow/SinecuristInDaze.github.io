from dataclasses import dataclass
import json
import pprint

import gymnasium as gym
import motrixsim as mtx
import numpy as np

from motrix_envs import registry
from motrix_envs.math import quaternion
from motrix_envs.np.env import NpEnv, NpEnvState

from .cfg import AnymalCEnvCfg, AnymalCRLCfg

_ENV_NAME = "anymal_c_navigation_rough-v0"

def _wrap_to_pi(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi

@registry.env(_ENV_NAME, "np")
class AnymalCEnv(NpEnv):
    _cfg: AnymalCEnvCfg

    # 基础 54 + 高度采样 187 = 241
    _OBS_DIM = 241 
    _ACT_DIM = 12

    _GOAL_POS_THRESHOLD = 0.3
    _GOAL_YAW_THRESHOLD = np.deg2rad(15.0)
    _STOP_ANGULAR_THRESHOLD = 0.05
    _TILT_TERMINATION_ANGLE = np.deg2rad(75.0)

    def __init__(self, cfg: AnymalCEnvCfg, num_envs: int = 1):
        super().__init__(cfg, num_envs=num_envs)

        self._body = self._model.get_body(cfg.asset.body_name)
        self._target_marker_body = self._safe_get_body("target_marker")
        self._robot_heading_arrow_body = self._safe_get_body("robot_heading_arrow")
        self._desired_heading_arrow_body = self._safe_get_body("desired_heading_arrow")

        self._num_dof_pos = self._model.num_dof_pos
        self._num_dof_vel = self._model.num_dof_vel
        self._num_action = self._model.num_actuators

        self._action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self._ACT_DIM,), dtype=np.float32)
        self._observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self._OBS_DIM,), dtype=np.float32)

        # 初始化采样点网格 (相对于机身坐标系)
        self._init_height_points()
        self._init_buffer()
        self._init_contact_geometry()

    def _init_height_points(self):
        """ 初始化 17x11 的高度采样点矩阵 """
        # x: -0.8 to 0.8 (17 points), y: -0.5 to 0.5 (11 points)
        x = np.linspace(-0.8, 0.8, 17)
        y = np.linspace(-0.5, 0.5, 11)
        grid_x, grid_y = np.meshgrid(x, y)
        
        # 形状为 (187, 3), z 默认为 0
        self.height_points = np.zeros((187, 3), dtype=np.float32)
        self.height_points[:, 0] = grid_x.flatten()
        self.height_points[:, 1] = grid_y.flatten()

    def _get_heights(self, data: mtx.SceneData):
        """ 获取机器人下方的地形高度采样 """
        # 1. 获取机器人在世界系下的位置和偏航角
        pose = self._body.get_pose(data) # (num_envs, 7)
        root_pos = pose[:, :3]
        root_quat = pose[:, 3:7]
        yaw = quaternion.get_yaw(root_quat)

        # 2. 构造旋转矩阵 (仅绕Z轴旋转)
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        
        # 3. 将相对采样点转换到世界坐标系
        # points_world = R * points_local + base_pos
        num_envs = data.shape[0]
        # (num_envs, 187, 3)
        sampled_points = np.zeros((num_envs, 187, 3), dtype=np.float32)
        
        for i in range(num_envs):
            # 简单的 2D 旋转
            rot_matrix = np.array([
                [cos_yaw[i], -sin_yaw[i]],
                [sin_yaw[i],  cos_yaw[i]]
            ])
            sampled_points[i, :, :2] = self.height_points[:, :2] @ rot_matrix.T
            sampled_points[i, :, :2] += root_pos[i, :2]

        # 4. 从场景中获取地形高度 (需根据 MotrixSim API 调用)
        # 假设接口为 self._model.get_height(data, points) 或类似
        # 如果 self._model 没有直接接口，通常通过 raycast 获取
        heights = self._model.get_height(data, sampled_points[:, :, :2]) 
        
        # 5. 返回相对高度 (采样高度 - 机器人基座高度)
        # 限制范围在 [-10, 10] 防止数值爆炸
        rel_heights = np.clip(heights - root_pos[:, 2:3], -10.0, 10.0)
        return rel_heights.astype(np.float32)

    def _safe_get_body(self, body_name: str):
        try:
            return self._model.get_body(body_name)
        except Exception:
            return None

    def _init_buffer(self):
        cfg = self._cfg
        self.default_angles = np.zeros(self._num_action, dtype=np.float32)
        for i in range(self._num_action):
            actuator_name = self._model.actuator_names[i]
            for name, angle in cfg.init_state.default_joint_angles.items():
                if name in actuator_name:
                    self.default_angles[i] = angle

        self._init_dof_pos = self._model.compute_init_dof_pos()
        self._init_dof_pos[-self._num_action :] = self.default_angles
        self._init_dof_vel = np.zeros((self._num_dof_vel,), dtype=np.float32)

        self._gravity = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        self._action_scale = float(cfg.control_config.action_scale)
        self._commands_scale = np.array(
            [cfg.normalization.lin_vel, 
             cfg.normalization.lin_vel, 
             cfg.normalization.ang_vel], dtype=np.float32
        )

        ctrl_limits = self._model.actuator_ctrl_limits
        self._actuator_ctrl_low = np.asarray(ctrl_limits[0], dtype=np.float32)
        self._actuator_ctrl_high = np.asarray(ctrl_limits[1], dtype=np.float32)

        # 注意：这里的 noise_scales 长度也需要对齐 _OBS_DIM
        self._obs_noise_scales = np.zeros((self._OBS_DIM,), dtype=np.float32)
        self._obs_noise_scales[0:3] = cfg.noise_config.scale_linvel
        self._obs_noise_scales[3:6] = cfg.noise_config.scale_gyro
        self._obs_noise_scales[6:9] = cfg.noise_config.scale_gravity
        self._obs_noise_scales[9:21] = cfg.noise_config.scale_joint_angle
        self._obs_noise_scales[21:33] = cfg.noise_config.scale_joint_vel
        # 地形高度采样的噪声通常设为很小或 0
        self._obs_noise_scales[54:] = 0.005 

    def _init_contact_geometry(self):
        ground_idx = None
        for name in [self._cfg.asset.ground_name, "floor", "ground"]:
            try:
                ground_idx = self._model.get_geom_index(name)
                break
            except Exception:
                continue

        if ground_idx is None:
            self._termination_contact = None
            return

        candidate_tokens = [token.lower() for token in self._cfg.asset.terminate_after_contacts_on]
        candidate_tokens.extend(["base", "shell", "battery", "hatch"])

        base_indices = []
        seen = set()
        for geom_name in self._model.geom_names:
            if geom_name is None: continue
            lname = geom_name.lower()
            if not any(token in lname for token in candidate_tokens): continue
            try:
                geom_idx = self._model.get_geom_index(geom_name)
            except Exception: continue
            if geom_idx in seen: continue
            seen.add(geom_idx)
            base_indices.append(geom_idx)

        if not base_indices:
            self._termination_contact = None
            return

        self._termination_contact = np.array([[idx, ground_idx] for idx in base_indices], dtype=np.uint32)

    def _check_base_contact(self, data: mtx.SceneData) -> np.ndarray:
        if self._termination_contact is None:
            return np.zeros((self._num_envs,), dtype=bool)
        cquery = self._model.get_contact_query(data)
        contacts = cquery.is_colliding(self._termination_contact)
        contacts = contacts.reshape((self._num_envs, -1))
        return contacts.any(axis=1)

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    def get_dof_pos(self, data: mtx.SceneData) -> np.ndarray:
        return self._body.get_joint_dof_pos(data)

    def get_dof_vel(self, data: mtx.SceneData) -> np.ndarray:
        return self._body.get_joint_dof_vel(data)

    def _compute_projected_gravity(self, quat: np.ndarray) -> np.ndarray:
        return quaternion.rotate_vector(quat, self._gravity)

    def _compute_navigation_state(self, root_pos: np.ndarray, root_quat: np.ndarray, info: dict) -> dict[str, np.ndarray]:
        cmd_cfg = self._cfg.commands
        target_pos = info["target_pos"]
        target_yaw = info["target_yaw"]

        robot_pos = root_pos[:, :2]
        robot_heading = quaternion.get_yaw(root_quat)

        position_error = target_pos - robot_pos
        distance = np.linalg.norm(position_error, axis=1)
        reached_position = distance < self._GOAL_POS_THRESHOLD
        heading_error = _wrap_to_pi(target_yaw - robot_heading)
        reached_heading = np.abs(heading_error) < self._GOAL_YAW_THRESHOLD
        reached_pose = np.logical_and(reached_position, reached_heading)

        desired_vel_xy = np.clip(position_error * float(cmd_cfg.position_gain), -float(cmd_cfg.max_command), float(cmd_cfg.max_command))
        desired_vel_xy = np.where(reached_position[:, np.newaxis], 0.0, desired_vel_xy)

        desired_yaw_rate = np.clip(heading_error * float(cmd_cfg.yaw_gain), -float(cmd_cfg.max_command), float(cmd_cfg.max_command))
        deadband = np.deg2rad(float(cmd_cfg.yaw_deadband_deg))
        desired_yaw_rate = np.where(np.abs(heading_error) < deadband, 0.0, desired_yaw_rate)
        desired_yaw_rate = np.where(reached_pose, 0.0, desired_yaw_rate)
        desired_vel_xy = np.where(reached_pose[:, np.newaxis], 0.0, desired_vel_xy)

        commands = np.concatenate([desired_vel_xy, desired_yaw_rate[:, np.newaxis]], axis=-1).astype(np.float32)

        return {
            "position_error": position_error.astype(np.float32),
            "distance": distance.astype(np.float32),
            "heading_error": heading_error.astype(np.float32),
            "reached_position": reached_position,
            "reached_heading": reached_heading,
            "reached_pose": reached_pose,
            "commands": commands,
            "desired_vel_xy": desired_vel_xy.astype(np.float32),
        }

    def _build_observation(
        self,
        base_lin_vel: np.ndarray,
        base_ang_vel: np.ndarray,
        root_quat: np.ndarray,
        joint_pos: np.ndarray,
        joint_vel: np.ndarray,
        info: dict,
        nav_state: dict[str, np.ndarray],
        data: mtx.SceneData, # 新增 data 参数
    ) -> tuple[np.ndarray, np.ndarray]:
        joint_pos_rel = joint_pos - self.default_angles
        projected_gravity = self._compute_projected_gravity(root_quat)

        command_obs = nav_state["commands"] * self._commands_scale
        position_error_obs = nav_state["position_error"] / 5.0
        heading_error_obs = (nav_state["heading_error"] / np.pi).reshape(-1, 1)
        distance_obs = np.clip(nav_state["distance"] / 5.0, 0.0, 1.0).reshape(-1, 1)

        stop_ready = np.logical_and(nav_state["reached_pose"], np.abs(base_ang_vel[:, 2]) < self._STOP_ANGULAR_THRESHOLD)

        # 核心改动：加入高度采样
        height_obs = self._get_heights(data)

        obs = np.concatenate(
            [
                base_lin_vel * self._cfg.normalization.lin_vel,
                base_ang_vel * self._cfg.normalization.ang_vel,
                projected_gravity,
                joint_pos_rel * self._cfg.normalization.dof_pos,
                joint_vel * self._cfg.normalization.dof_vel,
                info["current_actions"],
                command_obs,
                position_error_obs,
                heading_error_obs,
                distance_obs,
                nav_state["reached_pose"].astype(np.float32).reshape(-1, 1),
                stop_ready.astype(np.float32).reshape(-1, 1),
                height_obs, # 拼接高度图
            ],
            axis=-1,
        ).astype(np.float32)

        noise_level = float(self._cfg.noise_config.level)
        if noise_level > 0.0:
            obs_noise = np.random.uniform(low=-1.0, high=1.0, size=obs.shape).astype(np.float32)
            obs = obs + obs_noise * self._obs_noise_scales * noise_level

        assert obs.shape[1] == self._OBS_DIM, f"Obs dim mismatch: {obs.shape[1]} vs {self._OBS_DIM}"
        return obs, stop_ready

    # ... [保持原来的 _set_body_mocap_pose, _update_target_marker, _update_heading_arrows] ...
    def _set_body_mocap_pose(self, body, data: mtx.SceneData, pose: np.ndarray):
        if body is None: return
        mocap = getattr(body, "mocap", None)
        if mocap is None: return
        mocap.set_pose(data, pose)

    def _update_target_marker(self, data: mtx.SceneData, target_pos: np.ndarray, target_yaw: np.ndarray):
        num_envs = data.shape[0]
        marker_pos = np.column_stack([target_pos[:, 0], target_pos[:, 1], np.full((num_envs,), 0.5, dtype=np.float32)])
        marker_quat = quaternion.from_euler(0, 0, target_yaw)
        self._set_body_mocap_pose(self._target_marker_body, data, np.concatenate([marker_pos, marker_quat], axis=1))

    def _update_heading_arrows(self, data: mtx.SceneData, root_pos: np.ndarray, desired_vel_xy: np.ndarray, base_lin_vel_xy: np.ndarray):
        arrow_pos = root_pos.copy()
        arrow_pos[:, 2] = root_pos[:, 2] + 0.2
        current_yaw = np.where(np.linalg.norm(base_lin_vel_xy, axis=1) > 1e-3, np.arctan2(base_lin_vel_xy[:, 1], base_lin_vel_xy[:, 0]), 0.0)
        robot_arrow_quat = quaternion.from_euler(0, 0, current_yaw)
        self._set_body_mocap_pose(self._robot_heading_arrow_body, data, np.concatenate([arrow_pos, robot_arrow_quat], axis=1))
        desired_yaw = np.where(np.linalg.norm(desired_vel_xy, axis=1) > 1e-6, np.arctan2(desired_vel_xy[:, 1], desired_vel_xy[:, 0]), 0.0)
        desired_arrow_quat = quaternion.from_euler(0, 0, desired_yaw)
        self._set_body_mocap_pose(self._desired_heading_arrow_body, data, np.concatenate([arrow_pos, desired_arrow_quat], axis=1))

    # ... [保持原来的 _compute_reward 和 _compute_terminated] ...
    def _compute_reward(self, data: mtx.SceneData, info: dict, nav_state: dict[str, np.ndarray], base_lin_vel: np.ndarray, base_ang_vel: np.ndarray, root_quat: np.ndarray, joint_vel: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        cfg_reward = self._cfg.reward_config
        termination_penalty = np.zeros((self._num_envs,), dtype=np.float32)
        vel_max = np.abs(joint_vel).max(axis=1)
        vel_overflow = vel_max > float(self._cfg.max_dof_vel)
        vel_extreme = (np.isnan(joint_vel).any(axis=1)) | (np.isinf(joint_vel).any(axis=1)) | (vel_max > 1e6)
        termination_penalty = np.where(vel_overflow | vel_extreme, cfg_reward.termination_penalty, termination_penalty)
        base_contact = self._check_base_contact(data)
        termination_penalty = np.where(base_contact, cfg_reward.termination_penalty, termination_penalty)
        projected_gravity = self._compute_projected_gravity(root_quat)
        gxy = np.linalg.norm(projected_gravity[:, :2], axis=1)
        tilt_angle = np.arctan2(gxy, np.abs(projected_gravity[:, 2]))
        termination_penalty = np.where(tilt_angle > self._TILT_TERMINATION_ANGLE, cfg_reward.termination_penalty, termination_penalty)
        
        lin_vel_error = np.sum(np.square(nav_state["commands"][:, :2] - base_lin_vel[:, :2]), axis=1)
        tracking_lin = np.exp(-lin_vel_error / 0.25)
        ang_vel_error = np.square(nav_state["commands"][:, 2] - base_ang_vel[:, 2])
        tracking_ang = np.exp(-ang_vel_error / 0.25)
        
        distance_to_target = nav_state["distance"]
        if "min_distance" not in info: info["min_distance"] = distance_to_target.copy()
        distance_improvement = info["min_distance"] - distance_to_target
        approach_reward = np.clip(distance_improvement * float(cfg_reward.approach_scale), -float(cfg_reward.approach_clip), float(cfg_reward.approach_clip))
        info["min_distance"] = np.minimum(info["min_distance"], distance_to_target)
        
        reached_pose = nav_state["reached_pose"]
        info["ever_reached"] = info.get("ever_reached", np.zeros((self._num_envs,), dtype=bool))
        first_time_reach = np.logical_and(reached_pose, ~info["ever_reached"])
        info["ever_reached"] = np.logical_or(info["ever_reached"], reached_pose)
        arrival_bonus = np.where(first_time_reach, cfg_reward.arrival_bonus, 0.0)
        
        stop_bonus = np.where(reached_pose, float(cfg_reward.stop_bonus_scale) * (0.8 * np.exp(-((np.linalg.norm(base_lin_vel[:, :2], axis=1) / 0.2) ** 2))), 0.0)
        regularization = cfg_reward.lin_vel_z_penalty * np.square(base_lin_vel[:, 2]) + cfg_reward.torque_penalty * np.sum(np.square(data.actuator_ctrls), axis=1)
        
        reward = np.where(reached_pose, stop_bonus + arrival_bonus + regularization + termination_penalty, tracking_lin + tracking_ang + approach_reward + regularization + termination_penalty).astype(np.float32)
        return reward, {"total": reward}

    def _compute_terminated(self, obs: np.ndarray, joint_vel: np.ndarray, root_quat: np.ndarray, data: mtx.SceneData) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        finite_ok = np.isfinite(obs).all(axis=1)
        dof_vel_exceeded = np.max(np.abs(joint_vel), axis=1) > float(self._cfg.max_dof_vel)
        base_contact = self._check_base_contact(data)
        terminated = np.logical_or.reduce([~finite_ok, dof_vel_exceeded, base_contact])
        return terminated, {"base_contact": base_contact}

    def apply_action(self, actions: np.ndarray, state: NpEnvState):
        if "current_actions" not in state.info: state.info["current_actions"] = np.zeros((self._num_envs, self._num_action), dtype=np.float32)
        state.info["last_actions"] = state.info["current_actions"]
        state.info["current_actions"] = actions
        actions_scaled = actions * self._action_scale
        target_pos = np.clip(self.default_angles + actions_scaled, self._actuator_ctrl_low, self._actuator_ctrl_high)
        state.data.actuator_ctrls = target_pos
        return state

    def update_state(self, state: NpEnvState):
        data = state.data
        pose = self._body.get_pose(data)
        root_pos, root_quat = pose[:, :3], pose[:, 3:7]
        base_lin_vel = self._model.get_sensor_value(self._cfg.sensor.base_linvel, data).astype(np.float32)
        base_ang_vel = self._model.get_sensor_value(self._cfg.sensor.base_gyro, data).astype(np.float32)
        joint_pos, joint_vel = self.get_dof_pos(data).astype(np.float32), self.get_dof_vel(data).astype(np.float32)

        nav_state = self._compute_navigation_state(root_pos, root_quat, state.info)
        state.info.update({"commands": nav_state["commands"], "distance_to_target": nav_state["distance"], "has_reached_target": nav_state["reached_pose"], "desired_vel_xy": nav_state["desired_vel_xy"]})

        self._update_heading_arrows(data, root_pos, nav_state["desired_vel_xy"], base_lin_vel[:, :2])

        obs, stop_ready = self._build_observation(base_lin_vel, base_ang_vel, root_quat, joint_pos, joint_vel, state.info, nav_state, data)
        reward, _ = self._compute_reward(data, state.info, nav_state, base_lin_vel, base_ang_vel, root_quat, joint_vel)
        terminated, _ = self._compute_terminated(obs, joint_vel, root_quat, data)

        state.obs, state.reward, state.terminated = obs, reward, terminated
        return state

    def reset(self, data: mtx.SceneData, done: np.ndarray = None) -> tuple[np.ndarray, dict]:
        cfg = self._cfg
        num_envs = data.shape[0]
        pos_range = cfg.init_state.pos_randomization_range
        robot_init_pos = np.stack([np.random.uniform(pos_range[0], pos_range[2], num_envs), np.random.uniform(pos_range[1], pos_range[3], num_envs)], axis=1).astype(np.float32)
        
        cmd_range = cfg.commands.pose_command_range
        target_offset = np.random.uniform(low=np.array(cmd_range[:2]), high=np.array(cmd_range[3:5]), size=(num_envs, 2)).astype(np.float32)
        target_pos = robot_init_pos + target_offset
        target_yaw = np.random.uniform(low=cmd_range[2], high=cmd_range[5], size=(num_envs,)).astype(np.float32)

        init_dof_pos = np.tile(self._init_dof_pos, (num_envs, 1)).astype(np.float32)
        init_dof_pos[:, 0] += robot_init_pos[:, 0] - float(cfg.init_state.pos[0])
        init_dof_pos[:, 1] += robot_init_pos[:, 1] - float(cfg.init_state.pos[1])

        data.reset(self._model)
        data.set_dof_pos(init_dof_pos, self._model)
        self._model.forward_kinematic(data)

        info = {"target_pos": target_pos, "target_yaw": target_yaw, "current_actions": np.zeros((num_envs, self._num_action), dtype=np.float32)}
        nav_state = self._compute_navigation_state(robot_init_pos, np.tile([0,0,0,1], (num_envs, 1)), info)
        info.update({"commands": nav_state["commands"], "distance_to_target": nav_state["distance"], "min_distance": nav_state["distance"].copy()})

        # 获取传感器数据
        base_lin_vel = self._model.get_sensor_value(self._cfg.sensor.base_linvel, data).astype(np.float32)
        base_ang_vel = self._model.get_sensor_value(self._cfg.sensor.base_gyro, data).astype(np.float32)
        joint_pos = self.get_dof_pos(data).astype(np.float32)
        joint_vel = self.get_dof_vel(data).astype(np.float32)
        root_quat = self._body.get_pose(data)[:, 3:7]

        obs, _ = self._build_observation(base_lin_vel, base_ang_vel, root_quat, joint_pos, joint_vel, info, nav_state, data)
        return obs, info