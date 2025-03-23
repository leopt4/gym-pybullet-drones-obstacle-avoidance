import os
import numpy as np
import pybullet as p
from gymnasium import spaces

from gym_pybullet_drones.envs.BaseAviary import BaseAviary, ImageType
from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl

import random
import math
import json

class VisionOAAviary(BaseRLAviary):
    """Multi-drone environment class for control applications using vision."""

    ################################################################################
    
    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 num_drones: int=1,
                 neighbourhood_radius: float=np.inf,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int=30,
                 gui=False,
                 record=False,
                 obs: ObservationType=ObservationType.OB,
                 act: ActionType=ActionType.VEL,
                #  obstacles=False,
                #  user_debug_gui=True,
                 vision_attributes=True
                #  output_folder='results'
                 ):
        """Initialization of an aviary environment for control applications using vision.

        Attribute `vision_attributes` is automatically set to True when calling
        the superclass `__init__()` method.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
        num_drones : int, optional
            The desired number of drones in the aviary.
        neighbourhood_radius : float, optional
            Radius used to compute the drones' adjacency matrix, in meters.
        initial_xyzs: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial XYZ position of the drones.
        initial_rpys: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial orientations of the drones (in radians).
        physics : Physics, optional
            The desired implementation of PyBullet physics/custom dynamics.
        freq : int, optional
            The frequency (Hz) at which the physics engine steps.
        pyb_freq : int, optional
            The frequency at which PyBullet steps (a multiple of ctrl_freq).
        ctrl_freq : int, optional
            The frequency at which the environment steps.
        gui : bool, optional
            Whether to use PyBullet's GUI.
        record : bool, optional
            Whether to save a video of the simulation in folder `files/videos/`.
        obs : ObservationType, optional
            The type of observation space (kinematic information, vision or combination)
        act : ActionType, optional
            The type of action space (1 or 3D; RPMS, thurst and torques, or waypoint with PID control)
        obstacles : bool, optional
            Whether to add obstacles to the simulation.
        user_debug_gui : bool, optional
            Whether to draw the drones' axes and the GUI RPMs sliders.

        """
        os.environ['KMP_DUPLICATE_LIB_OK']='True'
        if drone_model in [DroneModel.CF2X, DroneModel.CF2P]:
            self.ctrl = [DSLPIDControl(drone_model=DroneModel.CF2X) for i in range(num_drones)]

        ## Obstacle 
        filename = "gym_pybullet_drones/obstacles/env_empty.json"
        try:
            with open(filename, "r") as f:
                self.OBSTACLES_POSITIONS = json.load(f)
            print(f"Loaded {len(self.OBSTACLES_POSITIONS)} positions from {filename}")
        except FileNotFoundError:
            print(f"File {filename} not found.")
            return []
        
        self.NUM_DRONES = 1
        self.ACTION_BUFFER_SIZE = 0

        # net_arch = dict(pi=[16, 16], vf=[16, 16])

        # Obstacle parameters
        self.OBSTACLES_RADIUS = 0.2
 
        # Observation parameters
        self.NAVIGATION_3D          = False     # cfg.getboolean('OPTIONS', 'NAVIGATION_3D')
        self.SPLIT_ROW              = 1         # cfg.getint('OPTIONS', 'SPLIT_ROW')
        self.SPLIT_COL              = 5         # cfg.getint('OPTIONS', 'SPLIT_COL')
        self.USING_VELOCITY_STATE   = False     # cfg.getboolean('OPTIONS', 'USING_VELOCITY_STATE')

        self.PERCEPTION_TYPE        = 'vector'  # cfg.get('OPTIONS', 'PERCEPTION_TYPE')
        self.MIN_DEPTH_M            = 0.07      # cfg.getfloat('OPTIONS', 'MIN_DEPTH_M')
        self.MAX_DEPTH_M            = 3.00      # cfg.getfloat('OPTIONS', 'MAX_DEPTH_M')
        
        if self.NAVIGATION_3D:
            if self.USING_VELOCITY_STATE:
                self.STATE_FEATURE_LENGTH = 6
            else:
                self.STATE_FEATURE_LENGTH = 3
        else:
            if self.USING_VELOCITY_STATE:
                self.STATE_FEATURE_LENGTH = 4
            else:
                self.STATE_FEATURE_LENGTH = 2

        self.CNN_FEATURE_LENGTH = self.SPLIT_ROW * self.SPLIT_COL

        # Start, Goal position and Workspace
        INIT_XYZS = np.array([[ 0, 0, 1.0+i*0.2] for i in range(self.NUM_DRONES)]).reshape(self.NUM_DRONES,3)
        INIT_RPYS = np.array([[0, 0, i*np.pi/4] for i in range(self.NUM_DRONES)]).reshape(self.NUM_DRONES,3)
        self.GOAL_DISTANCE = 3.0
        self.goal_pos = np.array([[self.GOAL_DISTANCE * np.cos(i*np.pi/4), self.GOAL_DISTANCE * np.sin(i*np.pi/4), 1.0+i*0.2] for i in range(self.NUM_DRONES)]).reshape(self.NUM_DRONES,3)

        self.work_space_x = [INIT_XYZS[0,0] - self.GOAL_DISTANCE*2, INIT_XYZS[0,0] + self.GOAL_DISTANCE*2]
        self.work_space_y = [INIT_XYZS[0,1] - self.GOAL_DISTANCE*2, INIT_XYZS[0,1] + self.GOAL_DISTANCE*2]
        self.work_space_z = [0.5, 2]
        # Action
        self.V_XY_MAX_MS        = 0.5           # cfg.getfloat('Drone', 'V_XY_MAX_MS')
        self.V_XY_MIN_MS        = 0.0           # cfg.getfloat('Drone', 'V_XY_MIN_MS')
        self.V_Z_MAX_MS         = 0.1           # cfg.getfloat('Drone', 'V_Z_MAX_MS')

        self.YAW_RATE_MAX_DEG   = 180            # cfg.getfloat('Drone', 'YAW_RATE_MAX_DEG')
        self.YAW_RATE_MAX_RAD   = math.radians(self.YAW_RATE_MAX_DEG)
        self.MAX_VERTICAL_DIFFERENCE = 1.0

        ## Reward parameters
        # Sparse rewards
        self.GOAL_REACHING_REWARD   = 30
        self.CRASH_REWARD           = -20
        self.OUTMAP_REWARD          = -30
        # Collision Proximity Penalty
        self.SAFETY_DISTANCE    = 0.25
        self.CRASH_DISTANCE     = 0.07
        # Scale
        self.XY_SCALE           = 1.0
        self.Z_SCALE            = 1.0
        self.CRASH_SCALE        = 5
        # 
        self.GOAL_ACCEPT_RADIUS = 0.25
        self.EPISODE_LEN_SEC = 10

        ## Variables
        self.previous_distance_from_des_point = self.GOAL_DISTANCE
        self.min_distance_to_obstacles = np.array([[np.inf] for i in range(self.NUM_DRONES)])   
    
        super().__init__(drone_model=drone_model,
                         num_drones=num_drones,
                         neighbourhood_radius=neighbourhood_radius,
                         initial_xyzs=INIT_XYZS,
                         initial_rpys=INIT_RPYS,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record,
                        #  obstacles=obstacles,
                        #  user_debug_gui=user_debug_gui,
                        #  output_folder=output_folder,
                         vision_attributes=vision_attributes
                         )

    ################################################################################

    def _addObstacles(self):
        """Add obstacles to the environment.

        These obstacles are loaded from standard URDF files included in Bullet.

        """
        
        
        # p.loadURDF("cylinders_map.urdf",
        #            physicsClientId=self.CLIENT
        #            )
        
        for i in range(len(self.OBSTACLES_POSITIONS)):
            p.loadURDF(
                "cylinder.urdf",
                self.OBSTACLES_POSITIONS[i],
                p.getQuaternionFromEuler([0, 0, 0]),
                physicsClientId=self.CLIENT,
                useFixedBase=True,
                globalScaling=1,
            )
        # p.loadURDF(
        #     "cube.urdf",
        #     [4.5,0,0],
        #     p.getQuaternionFromEuler([0, 0, 0]),
        #     physicsClientId=self.CLIENT,
        #     useFixedBase=True,
        #     globalScaling=1,
        # )

# ! -------------------------------- ACTION ---------------------------------------------
    ################################################################################
    def _actionSpace(self):
        """Returns the action space of the environment.

        Returns
        -------
        spaces.Box
            A Box of size NUM_DRONES x 4, 3, or 1, depending on the action type.

        """

        if self.NAVIGATION_3D:
            action_size = 3
            act_lower_bound = np.array([[self.V_XY_MIN_MS, -self.V_Z_MAX_MS, -self.YAW_RATE_MAX_RAD] for i in range(self.NUM_DRONES)])
            act_upper_bound = np.array([[self.V_XY_MAX_MS, self.V_Z_MAX_MS, self.YAW_RATE_MAX_RAD] for i in range(self.NUM_DRONES)])

        else:
            action_size = 2
            act_lower_bound = np.array([[self.V_XY_MIN_MS, -self.YAW_RATE_MAX_RAD] for i in range(self.NUM_DRONES)])
            act_upper_bound = np.array([[self.V_XY_MAX_MS, self.YAW_RATE_MAX_RAD] for i in range(self.NUM_DRONES)])

        # Init action buffer
        for i in range(self.ACTION_BUFFER_SIZE):
            self.action_buffer.append(np.zeros((self.NUM_DRONES,action_size)))
        
        return spaces.Box(low=act_lower_bound, high=act_upper_bound, dtype=np.float32)

    def _preprocessAction(self,
                          action
                          ):
        """Pre-processes the action passed to `.step()` into motors' RPMs.

        Uses PID control to target a desired velocity vector.

        Parameters
        ----------
        action : ndarray
            The desired velocity input for each drone, to be translated into RPMs.

        Returns
        -------
        ndarray
            (NUM_DRONES, 4)-shaped array of ints containing to clipped RPMs
            commanded to the 4 motors of each drone.

        """
        rpm = np.zeros((self.NUM_DRONES, 4))
        for i in range(action.shape[0]):
            target_v = action[i, :]

            # ! Note scale actions
            # v_xy_sp = action[0] * 0.7
            # yaw_rate_sp = action[-1] * 2

            if self.NAVIGATION_3D:
                v_z_sp = float(target_v[1])
            else:
                v_z_sp = 0.0

            yaw_sp = self.rpy[i,2] + target_v[-1] * self.CTRL_TIMESTEP
            
            v_x_sp = target_v[0] * math.cos(yaw_sp)
            v_y_sp = target_v[0] * math.sin(yaw_sp)

            pos_x_sp = self.pos[0,0] + v_x_sp * self.CTRL_TIMESTEP
            pos_y_sp = self.pos[0,1] + v_y_sp * self.CTRL_TIMESTEP
            pos_z_sp = self.pos[0,2]

            temp, _, _ = self.ctrl[i].computeControl(control_timestep=self.CTRL_TIMESTEP,
                                                    cur_pos=self.pos[i,:],
                                                    cur_quat=self.quat[i,:],
                                                    cur_vel=self.vel[i,:],
                                                    cur_ang_vel=self.ang_v[i,:],
                                                    target_pos=np.array([pos_x_sp, pos_y_sp, pos_z_sp]),                       # same as the current position
                                                    target_vel=np.array([v_x_sp, v_y_sp, v_z_sp]),  # target the desired velocity vector
                                                    target_rpy=np.array([0,0,yaw_sp]),
                                                    target_rpy_rates=np.array([0,0,target_v[-1]])
                                                    )
            rpm[i,:] = temp

        return rpm
    
# ! ------------------------------- OBSERVATION -----------------------------------------
    ################################################################################
    def _observationSpace(self):
        """Returns the observation space of the environment.

        Returns
        -------
        spaces.Dict
            A dictionary containing:
            - "image": A Box space for depth images (NUM_DRONES, H, W).
            - "states": A Box space for [xyz position, xyz velocity, yaw rate] (NUM_DRONES, 7).
        """

        
        if self.PERCEPTION_TYPE == 'vector':
            #### OBS SPACE OF SIZE CNN_FEATURE_LENGTH + STATE_LENGTH
            lo = 0
            hi = 1
            # Observation length
            obs_length = self.CNN_FEATURE_LENGTH*0 + self.STATE_FEATURE_LENGTH

            obs_lower_bound = np.array([[lo]*obs_length for i in range(self.NUM_DRONES)])
            obs_upper_bound = np.array([[hi]*obs_length for i in range(self.NUM_DRONES)])

            observation_space = spaces.Box(low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32)

        elif self.PERCEPTION_TYPE == "image_states":
            observation_space = spaces.Box(low=0, high=255,
                                            shape=(self.screen_height,
                                                    self.screen_width, 2),
                                            dtype=np.uint8)

        return observation_space

    def _computeObs(self):
        """Returns the current observation of the environment as a dictionary.

        Returns
        -------
        dict[str, np.ndarray]
            - "image": A NumPy array of shape (NUM_DRONES, H, W) containing depth images.
            - "states": A NumPy array of shape (NUM_DRONES, 7) containing [xyz position, xyz velocity, yaw rate].
        """
        if self.PERCEPTION_TYPE == 'vector':
            obs = self._getObsVector()

        # elif self.perception_type == 'image_states':
        #     obs = self.get_obs_lgmd()
        # else:
        #     obs = self.get_obs_image()

        return obs

    def _getObsVector(self):
        
        # obs_length = self.CNN_FEATURE_LENGTH + self.STATE_FEATURE_LENGTH
        obs_length = self.STATE_FEATURE_LENGTH
        obs = np.zeros((self.NUM_DRONES, obs_length))
        if self.step_counter % self.IMG_CAPTURE_FREQ == 0:
            for i in range(self.NUM_DRONES):
                ## GET IMAGE VECTOR SIZE OF CNN_FEATURE_LENGTH (SPLIT_ROW * SPLIT_COL)
                img_vector_norm_obs = self._getImageVectorObs(i)
                
                ## GET STATES VECTOR SIZE OF STATE_LENGTH
                state_norm_obs = self._getStateObs(i) / 255

                ## CONCATENATE OBSERVATION FEATURE
                # obs[i,:] = np.hstack([img_vector_norm_obs, state_norm_obs]).reshape(1,obs_length)
                obs[i,:] = np.hstack([state_norm_obs]).reshape(1,obs_length)
            ret = np.array([obs[i, :] for i in range(self.NUM_DRONES)]).astype('float32')
            #### Add action buffer to observation #######################
            for i in range(self.ACTION_BUFFER_SIZE):
                ret = np.hstack([ret, np.array([self.action_buffer[i][j, :] for j in range(self.NUM_DRONES)])])

        return ret
    
    def _getImageVectorObs(self, nth_drone):
        
        self.rgb[nth_drone], self.dep[nth_drone], self.seg[nth_drone] = self._getDroneImages(nth_drone, segmentation=False)

        # Save images if recording is enabled
        if self.RECORD:
            self._exportImage(
                img_type=ImageType.DEP,
                img_input=self.dep[nth_drone],
                path=self.ONBOARD_IMG_PATH + f"/drone_{nth_drone}",
                frame_num=int(self.step_counter / self.IMG_CAPTURE_FREQ)
            )

        # filtered_values = self.dep[i][(self.dep[i] >= self.min_depth_meters) & (self.dep[i] <= self.max_depth_meters)]
        
        # Vectorize depth image to size 1 x (cnn_feature_length)
        image_scaled = np.clip(self.dep[nth_drone], self.MIN_DEPTH_M, self.MAX_DEPTH_M) \
                            / (self.MAX_DEPTH_M - self.MIN_DEPTH_M)
        
        self.min_distance_to_obstacles[nth_drone] = image_scaled.min()

        image_scaled = image_scaled * 255
        image_scaled = 255 - image_scaled
        image_uint8 = image_scaled.astype(np.uint8)

        v_split_list = np.vsplit(image_uint8, self.SPLIT_ROW)

        split_final = []
        for i in range(self.SPLIT_ROW):
            h_split_list = np.hsplit(v_split_list[i], self.SPLIT_COL)
            for j in range(self.SPLIT_COL):
                split_final.append(h_split_list[j].max())

        img_feature_norm = np.array(split_final).reshape(1,self.SPLIT_ROW*self.SPLIT_COL) / 255.0

        return img_feature_norm
    
    def _getStateObs(self, nth_drone):
        distance = self.get_distance_to_goal_2d(nth_drone)

        relative_yaw = self.get_relative_yaw(nth_drone)                    # return relative yaw -pi to pi 
        
        relative_pose_z = self.pos[nth_drone,2] - self.goal_pos[nth_drone,2]   # current position z is positive

        vertical_distance_norm = (relative_pose_z / self.MAX_VERTICAL_DIFFERENCE / 2 + 0.5) * 255

        distance_norm = distance / self.GOAL_DISTANCE / 2 * 255

        relative_yaw_norm = (relative_yaw / np.pi / 2 + 0.5) * 255

        # current speed and angular speed
        velocity = self.vel[nth_drone,:]
        linear_velocity_xy = np.sqrt(pow(velocity[0], 2) + pow(velocity[1], 2))
        linear_velocity_norm = (linear_velocity_xy - self.V_XY_MIN_MS) / (self.V_XY_MAX_MS - self.V_XY_MIN_MS) * 255
        linear_velocity_z = velocity[2]
        linear_velocity_z_norm = (linear_velocity_z / self.V_Z_MAX_MS / 2 + 0.5) * 255

        angular_velocity = self.ang_v[nth_drone,2]
        angular_velocity_norm = (angular_velocity / self.YAW_RATE_MAX_RAD / 2 + 0.5) * 255

        # state: distance_h, distance_v, relative yaw, velocity_x, velocity_z, velocity_yaw
        self.state_raw = np.array([distance, relative_pose_z,  math.degrees(
            relative_yaw), linear_velocity_xy, linear_velocity_z,  math.degrees(angular_velocity)])
        state_norm = np.array([distance_norm, vertical_distance_norm, relative_yaw_norm,
                            linear_velocity_norm, linear_velocity_z_norm, angular_velocity_norm])
        state_norm = np.clip(state_norm, 0, 255)
        
        if self.NAVIGATION_3D:
            if self.USING_VELOCITY_STATE == False:
                state_norm = state_norm[:3]
        else:
            state_norm = np.array(
                [state_norm[0], state_norm[2], state_norm[3], state_norm[5]])
            if self.USING_VELOCITY_STATE == False:
                state_norm = state_norm[:2]

        state_norm = state_norm.reshape(1,len(state_norm))
        return state_norm
    
# ! --------------------------------- REWARD ---------------------------------------------
    ################################################################################
    def _computeReward(self, action, terminated=False, truncated=False):
        """Computes the current reward value(s).

        Unused as this subclass is not meant for reinforcement learning.

        Returns
        -------
        float
            The reward.

        """
        """Computes the current reward value based on distance, collision proximity, and penalties."""

        reward = 0

        distance_reward_coef = 10

        if not (terminated or truncated):
            # 1 - goal reward
            distance_now = self.get_distance_to_goal_3d(0)
            reward_distance = distance_reward_coef * (self.previous_distance_from_des_point - distance_now) / \
                self.GOAL_DISTANCE   # normalized to 100 according to goal_distance
            
            self.previous_distance_from_des_point = distance_now

            # 2 - Position punishment
            current_pose = self.pos[0,:]
            goal_pose = self.goal_pos[0,:]
            x = current_pose[0]
            y = current_pose[1]
            z = current_pose[2]
            x_g = goal_pose[0]
            y_g = goal_pose[1]
            z_g = goal_pose[2]

            punishment_xy = np.clip(self.getDis(
                x, y, 0, 0, x_g, y_g) / self.XY_SCALE, 0, 1)
            punishment_z = 0.5 * np.clip(abs(z - z_g)/self.Z_SCALE, 0, 1)

            punishment_pose = punishment_xy + punishment_z

            min_distance_to_obstacles = self.min_distance_to_obstacles[0].astype(float)
            if min_distance_to_obstacles[0] < self.SAFETY_DISTANCE:
                punishment_obs = 1 - np.clip((min_distance_to_obstacles[0] - self.CRASH_DISTANCE) / self.CRASH_SCALE, 0, 1)
            else:
                punishment_obs = 0

            punishment_action = 0

            # add yaw_rate cost
            yaw_speed_cost = abs(action[0,-1]) / self.YAW_RATE_MAX_RAD

            if self.NAVIGATION_3D:
                # add action and z error cost
                v_z_cost = ((abs(action[0,1]) / self.V_Z_MAX_MS)**2)
                z_err_cost = (
                    (abs(self.state_raw[1]) / self.MAX_VERTICAL_DIFFERENCE)**2)
                punishment_action += (v_z_cost + z_err_cost)

            punishment_action += yaw_speed_cost

            yaw_error = self.state_raw[2]
            yaw_error_cost = abs(yaw_error / 90)

            reward = reward_distance - 0.1 * punishment_pose - 0.2 * \
                punishment_obs - 0.1 * punishment_action - 0.5 * yaw_error_cost
        else:
            if self.has_reached_des_pose:
                reward = self.GOAL_REACHING_REWARD
            if self.too_close_to_obstable:
                reward = self.CRASH_REWARD
            if self.is_not_inside_workspace_now:
                reward = self.OUTMAP_REWARD
        # print("Distance: ", round(reward_distance,5), "Pose: ", round(- 0.1 * punishment_pose, 5), "Action: ", round(- 0.1 * punishment_action, 5), "Yaw: ", round(- 0.5 * yaw_error_cost, 5))
        return reward

        ################################################################################

# ! --------------------------------- TERMINATED -----------------------------------------
    ################################################################################
    def _computeTerminated(self):
        """Computes the current done value.

        Returns
        -------
        bool
            Whether the current episode is done.

        """

        return self.reachDesiredPose()

# ! --------------------------------- TRUNCATED ------------------------------------------
    ################################################################################
    def _computeTruncated(self):
        """Computes the current truncated value(s).

        Unused as this subclass is not meant for reinforcement learning.

        Returns
        -------
        bool
            Dummy value.

        """
        truncated = False


        self.is_not_inside_workspace_now = self.isNotInsideWorkspace()
        self.has_reached_des_pose = self.reachDesiredPose()
        self.too_close_to_obstable = self.isCrashed()

        truncated = self.is_not_inside_workspace_now or\
            self.has_reached_des_pose or\
            self.too_close_to_obstable or\
            self.step_counter/self.PYB_FREQ > self.EPISODE_LEN_SEC
        
        return truncated

# ! ---------------------------------- INFORMATION ---------------------------------------
    ################################################################################
    def _computeInfo(self):
        """Computes the current info dict(s).

        Returns
        -------
        dict[str, bool]
            info.

        """
        info = {
            'is_success': self.reachDesiredPose(),
            'is_crash': self.isCrashed(),
            'is_not_in_workspace': self.isNotInsideWorkspace(),
            'step_num': self.step_counter/self.PYB_FREQ
        }
        return info

# ! -------------------------- CONDITION FUNCTION------------------------------------------
    ################################################################################

    def isNotInsideWorkspace(self):
        """
        Check if the Drones is inside the Workspace defined
        """
        is_not_inside = False
        for i in range(self.NUM_DRONES):
            current_position = self.pos[i,:]

            if current_position[0] < self.work_space_x[0] or current_position[0] > self.work_space_x[1] or \
                current_position[1] < self.work_space_y[0] or current_position[1] > self.work_space_y[1] or \
                    current_position[2] < self.work_space_z[0] or current_position[2] > self.work_space_z[1]:
                is_not_inside = True
                # print("------------------------------------OUTMAP---------------------------------------")

        return is_not_inside

    def reachDesiredPose(self):
        reach_desired_pose = False
        for i in range(self.NUM_DRONES):
            if self.get_distance_to_goal_3d(i) < self.GOAL_ACCEPT_RADIUS:
                reach_desired_pose = True
                print("------------------------------------GOAL-----------------------------------------")

        return reach_desired_pose

    def isCrashed(self):
        is_crashed_truncated = False

        for i in range(self.NUM_DRONES):
            for obs_center in self.OBSTACLES_POSITIONS:  # List of (x, y, z) obstacle centers
                obs_x, obs_y, obs_z = obs_center

                # Compute 2D distance (ignoring height for cylinder collision)
                distance = np.sqrt(pow(self.pos[i,0] - obs_x, 2) + pow(self.pos[i,1] - obs_y, 2))
                
                if distance-self.OBSTACLES_RADIUS <= self.CRASH_DISTANCE:  # Collision if within the radius
                    is_crashed_truncated = True
                    # print("------------------------------------CRASH-----------------------------------------")
                    break

        return is_crashed_truncated
    
    # ! ------------------ USEFULL FUNCTION ---------------------------------------------
    #################################################################################

    def get_relative_yaw(self, nth_drone):
        """Returns relative yaw from current pose to goal in radian of the n-th drone.
        Parameters
        ----------
        nth_drone : int
            The ordinal number/position of the desired drone in list self.DRONE_IDS.

        Returns
        -------
        ndarray 
            Relative yaw from current pose to goal in radian of the n-th drone.

        """
        current_position = self.pos[nth_drone, :]
        goal_pos = self.goal_pos[nth_drone, :]
        # get relative angle
        relative_pose_x = goal_pos[0] - current_position[0]
        relative_pose_y = goal_pos[1] - current_position[1]
        angle = np.arctan2(relative_pose_y, relative_pose_x)

        # get current yaw
        yaw_current = self.rpy[nth_drone, 2]
        
        # get yaw error
        yaw_error = angle - yaw_current
        yaw_error = angle_norm(yaw_error)

        return yaw_error
    
    def get_distance_to_goal_2d(self, nth_drone):
        """ 2D Returns distance from current pose to goal in meters of the n-th drone.
        Parameters
        ----------
        nth_drone : int
            The ordinal number/position of the desired drone in list self.DRONE_IDS.

        Returns
        -------
        ndarray 
            2D Distance from current pose to goal in meters of the n-th drone.

        """
        return np.sqrt(pow(self.pos[nth_drone, 0] - self.goal_pos[nth_drone, 0], 2) \
                       + pow(self.pos[nth_drone, 1] - self.goal_pos[nth_drone, 1], 2))
    
    def get_distance_to_goal_3d(self, nth_drone):
        """Returns 3D distance from current pose to goal in meters of the n-th drone.
        Parameters
        ----------
        nth_drone : int
            The ordinal number/position of the desired drone in list self.DRONE_IDS.

        Returns
        -------
        ndarray 
            3D Distance from current pose to goal in meters of the n-th drone.

        """
        current_pose = self.pos[nth_drone, :]
        goal_pose = self.goal_pos[nth_drone, :]
        relative_pose_x = current_pose[0] - goal_pose[0]
        relative_pose_y = current_pose[1] - goal_pose[1]
        relative_pose_z = current_pose[2] - goal_pose[2]

        return np.sqrt(pow(relative_pose_x, 2) + pow(relative_pose_y, 2) + pow(relative_pose_z, 2))
    
    def getDis(self, pointX, pointY, lineX1, lineY1, lineX2, lineY2):
        '''
        Get distance between Point and Line
        Used to calculate position punishment
        '''
        a = lineY2-lineY1
        b = lineX1-lineX2
        c = lineX2*lineY1-lineX1*lineY2
        dis = (math.fabs(a*pointX+b*pointY+c))/(math.pow(a*a+b*b, 0.5))

        return dis

    def _getClosestObstacleDistance(self):
        """Computes the closest distance from the drone to any obstacle.

        Returns
        -------
        float
            drone's closest distance to the surface of an obstacle

        """
        pos = self._getDroneStateVector(0)[0:2]     # Drone position (x, y, z)
        if not self.OBSTACLES_POSITIONS:
            return float('inf')  # No obstacles loaded
        
        distances = [np.linalg.norm(np.array(obstacle[0:2]) - pos) - self.OBSTACLES_RADIUS 
                    for obstacle in self.OBSTACLES_POSITIONS]
        return min(distances) if distances else float('inf')

        """Computes the intersection point between the closest obstacle surface 
        and the line connecting the drone to the obstacle center.

        Returns
        -------
        tuple
            (intersection_position, closest_distance)
        """
        pos = self._getDroneStateVector(0)[0:2]  # Drone position (x, y, z)
        heading = self._getDroneStateVector(0)[9] # Drone yaw
        if not self.OBSTACLES_POSITIONS:
            return None, float('inf')  # No obstacles loaded

        min_distance = float('inf')
        closest_intersection = None

        for obstacle in self.OBSTACLES_POSITIONS:
            obs = np.array([obstacle[0], obstacle[1]])
            direction = np.array(pos) - np.array(obs)
            distance = np.linalg.norm(direction) - self.OBSTACLES_RADIUS

            if distance < min_distance:
                min_distance = distance
                unit_direction = direction / np.linalg.norm(direction)  # Normalize vector
                closest_intersection = np.array(obs) + self.OBSTACLES_RADIUS * unit_direction

        if closest_intersection is not None:
            # Compute relative position
            relative_intersection = closest_intersection - pos

            # Rotation matrix to transform into drone frame
            rotation_matrix = np.array([
                [np.cos(-heading), -np.sin(-heading)],
                [np.sin(-heading), np.cos(-heading)]
            ])
            
            # Apply rotation
            intersection_in_drone_frame = rotation_matrix @ relative_intersection
            return relative_intersection, min_distance

        return None, float('inf')  # No valid intersection found


def angle_norm(angle_rad):
    if angle_rad > math.pi:
        angle_rad -= 2*math.pi
    elif angle_rad < -math.pi:
        angle_rad += 2*math.pi
    return angle_rad