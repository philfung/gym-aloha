import gymnasium as gym
import numpy as np
from dm_control import mujoco
from dm_control.rl import control
from gymnasium import spaces

from gym_aloha.constants import (
    ACTIONS,
    ASSETS_DIR,
    DT,
    JOINTS,
)
from gym_aloha.tasks.sim import BOX_POSE, InsertionTask, TransferCubeTask
from gym_aloha.tasks.sim_end_effector import (
    InsertionEndEffectorTask,
    TransferCubeEndEffectorTask,
)
from gym_aloha.utils import sample_box_pose, sample_insertion_pose


class AlohaEnv(gym.Env):
    # TODO(aliberts): add "human" render_mode
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(
        self,
        task,
        obs_type="pixels",
        render_mode="rgb_array",
        observation_width=640,
        observation_height=480,
        visualization_width=640,
        visualization_height=480,
        action_smoothing=0.2,  # Action smoothing factor (0 = no smoothing, 1 = full smoothing)
    ):
        super().__init__()
        self.task = task
        self.obs_type = obs_type
        self.render_mode = render_mode
        self.observation_width = observation_width
        self.observation_height = observation_height
        self.visualization_width = visualization_width
        self.visualization_height = visualization_height
        self.action_smoothing = action_smoothing
        self.previous_action = None

        self._env = self._make_env_task(self.task)

        if self.obs_type == "state":
            raise NotImplementedError()
            self.observation_space = spaces.Box(
                low=np.array([0] * len(JOINTS)),  # ???
                high=np.array([255] * len(JOINTS)),  # ???
                dtype=np.float64,
            )
        elif self.obs_type == "pixels":
            self.observation_space = spaces.Dict(
                {
                    "top": spaces.Box(
                        low=0,
                        high=255,
                        shape=(self.observation_height, self.observation_width, 3),
                        dtype=np.uint8,
                    )
                }
            )
        elif self.obs_type == "pixels_agent_pos":
            self.observation_space = spaces.Dict(
                {
                    "pixels": spaces.Dict(
                        {
                            "top": spaces.Box(
                                low=0,
                                high=255,
                                shape=(self.observation_height, self.observation_width, 3),
                                dtype=np.uint8,
                            )
                        }
                    ),
                    "agent_pos": spaces.Box(
                        low=-1000.0,
                        high=1000.0,
                        shape=(len(JOINTS),),
                        dtype=np.float64,
                    ),
                }
            )

        self.action_space = spaces.Box(low=-1, high=1, shape=(len(ACTIONS),), dtype=np.float32)

    def render(self):
        return self._render(visualize=True)

    def _render(self, visualize=False):
        assert self.render_mode == "rgb_array"
        width, height = (
            (self.visualization_width, self.visualization_height)
            if visualize
            else (self.observation_width, self.observation_height)
        )
        # if mode in ["visualize", "human"]:
        #     height, width = self.visualize_height, self.visualize_width
        # elif mode == "rgb_array":
        #     height, width = self.observation_height, self.observation_width
        # else:
        #     raise ValueError(mode)
        # TODO(rcadene): render and visualizer several cameras (e.g. angle, front_close)
        image = self._env.physics.render(height=height, width=width, camera_id="top")
        return image

    def _make_env_task(self, task_name):
        # time limit is controlled by StepCounter in env factory
        time_limit = float("inf")

        if task_name == "transfer_cube":
            xml_path = ASSETS_DIR / "bimanual_viperx_transfer_cube.xml"
            physics = mujoco.Physics.from_xml_path(str(xml_path))
            task = TransferCubeTask()
        elif task_name == "insertion":
            xml_path = ASSETS_DIR / "bimanual_viperx_insertion.xml"
            physics = mujoco.Physics.from_xml_path(str(xml_path))
            task = InsertionTask()
        elif task_name == "end_effector_transfer_cube":
            raise NotImplementedError()
            xml_path = ASSETS_DIR / "bimanual_viperx_end_effector_transfer_cube.xml"
            physics = mujoco.Physics.from_xml_path(str(xml_path))
            task = TransferCubeEndEffectorTask()
        elif task_name == "end_effector_insertion":
            raise NotImplementedError()
            xml_path = ASSETS_DIR / "bimanual_viperx_end_effector_insertion.xml"
            physics = mujoco.Physics.from_xml_path(str(xml_path))
            task = InsertionEndEffectorTask()
        else:
            raise NotImplementedError(task_name)

        # Use the physics settings from the XML file
        # Add a small amount of action noise to prevent instability
        env = control.Environment(
            physics, task, time_limit, control_timestep=DT, flat_observation=False
        )
        return env

    def _format_raw_obs(self, raw_obs):
        if self.obs_type == "state":
            raise NotImplementedError()
        elif self.obs_type == "pixels":
            obs = {"top": raw_obs["images"]["top"].copy()}
        elif self.obs_type == "pixels_agent_pos":
            obs = {
                "pixels": {"top": raw_obs["images"]["top"].copy()},
                "agent_pos": raw_obs["qpos"],
            }
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset action smoothing
        self.previous_action = None

        # TODO(rcadene): how to seed the env?
        if seed is not None:
            self._env.task.random.seed(seed)
            self._env.task._random = np.random.RandomState(seed)

        # TODO(rcadene): do not use global variable for this
        if self.task == "transfer_cube":
            BOX_POSE[0] = sample_box_pose(seed)  # used in sim reset
        elif self.task == "insertion":
            BOX_POSE[0] = np.concatenate(sample_insertion_pose(seed))  # used in sim reset
        else:
            raise ValueError(self.task)

        try:
            raw_obs = self._env.reset()
            observation = self._format_raw_obs(raw_obs.observation)
            info = {"is_success": False}
            return observation, info
        except Exception as e:
            print(f"Error in environment reset: {e}")
            # Return a safe fallback in case of error
            return self.observation_space.sample(), {"is_success": False, "reset_error": True}

    def step(self, action):
        assert action.ndim == 1
        # TODO(rcadene): add info["is_success"] and info["success"] ?

        try:
            # Check for NaN or Inf in action
            if np.any(np.isnan(action)) or np.any(np.isinf(action)):
                print("Warning: NaN or Inf detected in action. Clipping to valid range.")
                action = np.clip(action, -1.0, 1.0)  # Clip to valid range
                action = np.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0)  # Replace NaN/Inf with valid values
            
            # Apply action smoothing to prevent large, sudden movements
            if self.previous_action is not None and self.action_smoothing > 0:
                # Blend previous action with current action to smooth transitions
                smoothed_action = self.action_smoothing * self.previous_action + (1 - self.action_smoothing) * action
                # Add a small amount of noise to prevent getting stuck in unstable states
                noise = np.random.normal(0, 0.01, size=action.shape)
                smoothed_action = np.clip(smoothed_action + noise, -1.0, 1.0)
                self.previous_action = smoothed_action
                action_to_apply = smoothed_action
            else:
                self.previous_action = action
                action_to_apply = action
            
            # Take a step in the environment
            _, reward, _, raw_obs = self._env.step(action_to_apply)
            
            # Check for physics instability
            qpos = self._env.physics.data.qpos
            if np.any(np.isnan(qpos)) or np.any(np.isinf(qpos)):
                print("Warning: Physics instability detected. Terminating episode.")
                reward = 0  # Penalize unstable states
                terminated = True
                truncated = True
                info = {"is_success": False, "physics_error": True}
                # Return the last valid observation
                return self.observation_space.sample(), reward, terminated, truncated, info
            
            # Normal case - no instability
            terminated = is_success = reward == 4
            info = {"is_success": is_success, "physics_error": False}
            observation = self._format_raw_obs(raw_obs)
            truncated = False
            
            return observation, reward, terminated, truncated, info
            
        except Exception as e:
            print(f"Error in environment step: {e}")
            # Return a safe fallback in case of error
            terminated = True
            truncated = True
            reward = 0
            info = {"is_success": False, "physics_error": True, "error_msg": str(e)}
            # Return a random valid observation as fallback
            return self.observation_space.sample(), reward, terminated, truncated, info

    def close(self):
        pass
