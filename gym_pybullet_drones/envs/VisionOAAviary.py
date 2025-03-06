import os
import numpy as np
import pybullet as p
from gym import spaces

from gym_pybullet_drones.envs.BaseAviary import BaseAviary, ImageType
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl

import json

class VisionOAAviary(BaseAviary):
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
                 ctrl_freq: int=240,
                 gui=False,
                 record=False,
                 obstacles=False,
                 user_debug_gui=True,
                 vision_attributes=True,
                 output_folder='results'
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
        obstacles : bool, optional
            Whether to add obstacles to the simulation.
        user_debug_gui : bool, optional
            Whether to draw the drones' axes and the GUI RPMs sliders.

        """
        os.environ['KMP_DUPLICATE_LIB_OK']='True'
        if drone_model in [DroneModel.CF2X, DroneModel.CF2P]:
            self.ctrl = [DSLPIDControl(drone_model=DroneModel.CF2X) for i in range(num_drones)]

        super().__init__(drone_model=drone_model,
                         num_drones=num_drones,
                         neighbourhood_radius=neighbourhood_radius,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record,
                         obstacles=obstacles,
                         user_debug_gui=user_debug_gui,
                         output_folder=output_folder,
                         vision_attributes=vision_attributes
                         )
        
        #### Set a limit on the maximum target speed ###############
        self.SPEED_LIMIT = 0.03 * self.MAX_SPEED_KMH * (1000/3600)


    ################################################################################

    # def reset(self):
    #     self.truncated = False
    #     self.done = False
    #     for rwd in self.reward_components:
    #         rwd.reset()
    #     for term in self.term_components:
    #         term.reset()
    #     obs = super().reset()
    #     # setting this to false here to allow one time allocations of reward and term values to not be repeated
    #     self.init = False

    #     return obs

    ################################################################################

    def _addObstacles(self):
        """Add obstacles to the environment.

        These obstacles are loaded from standard URDF files included in Bullet.

        """
        # filename = "../../tests/env_create/env_1.0_5_0.7_63.json"
        # try:
        #     with open(filename, "r") as f:
        #         positions = json.load(f)
        #     print(f"Loaded {len(positions)} positions from {filename}")
        # except FileNotFoundError:
        #     print(f"File {filename} not found.")
        #     return []

        p.loadURDF("cylinders_map.urdf",
                   physicsClientId=self.CLIENT
                   )
        
        # for i in range(len(positions)):
        #     p.loadURDF(
        #         "cylinder.urdf",
        #         positions[i],
        #         p.getQuaternionFromEuler([0, 0, 0]),
        #         physicsClientId=self.CLIENT,
        #         useFixedBase=True,
        #         globalScaling=1,
        #     )

    ################################################################################
    
    def _actionSpace(self):
        """Returns the action space of the environment.

        Returns
        -------
        spaces.Box
            An ndarray of shape (NUM_DRONES, 4) for the commanded RPMs.

        """
        #### Action vector ######## P0            P1            P2            P3
        act_lower_bound = np.array([[0.,           0.,           0.,           0.] for i in range(self.NUM_DRONES)])
        act_upper_bound = np.array([[self.MAX_RPM, self.MAX_RPM, self.MAX_RPM, self.MAX_RPM] for i in range(self.NUM_DRONES)])
        return spaces.Box(low=act_lower_bound, high=act_upper_bound, dtype=np.float32)
    
    ################################################################################
    
    # def _observationSpace(self):
    #     """Returns the observation space of the environment.

    #     Returns
    #     -------
    #     dict[str, dict[str, ndarray]]
    #         A Dict with NUM_DRONES entries indexed by Id in string format,
    #         each a Dict in the form {Box(20,), MultiBinary(NUM_DRONES), Box(H,W,4), Box(H,W), Box(H,W)}.

    #     """
    #     #### Observation vector ### X        Y        Z       Q1   Q2   Q3   Q4   R       P       Y       VX       VY       VZ       WX       WY       WZ       P0            P1            P2            P3
    #     obs_lower_bound = np.array([-np.inf, -np.inf, 0.,     -1., -1., -1., -1., -np.pi, -np.pi, -np.pi, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, 0.,           0.,           0.,           0.])
    #     obs_upper_bound = np.array([np.inf,  np.inf,  np.inf, 1.,  1.,  1.,  1.,  np.pi,  np.pi,  np.pi,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  self.MAX_RPM, self.MAX_RPM, self.MAX_RPM, self.MAX_RPM])
    #     return spaces.Dict({str(i): spaces.Dict({"state": spaces.Box(low=obs_lower_bound,
    #                                                                  high=obs_upper_bound,
    #                                                                  dtype=np.float32
    #                                                                  ),
    #                                              "neighbors": spaces.MultiBinary(self.NUM_DRONES),
    #                                              "rgb": spaces.Box(low=0,
    #                                                                high=255,
    #                                                                shape=(self.IMG_RES[1], self.IMG_RES[0], 4),
    #                                                                dtype=np.uint8
    #                                                                ),
    #                                              "dep": spaces.Box(low=.01,
    #                                                                high=1000.,
    #                                                                shape=(self.IMG_RES[1],
    #                                                                 self.IMG_RES[0]),
    #                                                                dtype=np.float32
    #                                                                ),
    #                                              "seg": spaces.Box(low=0,
    #                                                                high=100,
    #                                                                shape=(self.IMG_RES[1],
    #                                                                self.IMG_RES[0]),
    #                                                                dtype=int
    #                                                                )
    #                                              }) for i in range(self.NUM_DRONES)})
    def _observationSpace(self):
        """Returns the observation space of the environment.

        Returns
        -------
        spaces.Box
            The observation space, i.e., an ndarray of shape (NUM_DRONES, 20).

        """
        #### Observation vector ### X        Y        Z       Q1   Q2   Q3   Q4   R       P       Y       VX       VY       VZ       WX       WY       WZ       P0            P1            P2            P3
        obs_lower_bound = np.array([[-np.inf, -np.inf, 0.,     -1., -1., -1., -1., -np.pi, -np.pi, -np.pi, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, 0.,           0.,           0.,           0.] for i in range(self.NUM_DRONES)])
        obs_upper_bound = np.array([[np.inf,  np.inf,  np.inf, 1.,  1.,  1.,  1.,  np.pi,  np.pi,  np.pi,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf,  self.MAX_RPM, self.MAX_RPM, self.MAX_RPM, self.MAX_RPM] for i in range(self.NUM_DRONES)])
        return spaces.Box(low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32)

    ################################################################################
    
    # def _computeObs(self):
    #     """Returns the current observation of the environment.

    #     For the value of key "state", see the implementation of `_getDroneStateVector()`,
    #     the value of key "neighbors" is the drone's own row of the adjacency matrix,
    #     "rgb", "dep", and "seg" are matrices containing POV camera captures.

    #     Returns
    #     -------
    #     dict[str, dict[str, ndarray]]
    #         A Dict with NUM_DRONES entries indexed by Id in string format,
    #         each a Dict in the form {Box(20,), MultiBinary(NUM_DRONES), Box(H,W,4), Box(H,W), Box(H,W)}.

    #     """
    #     adjacency_mat = self._getAdjacencyMatrix()
    #     obs = {}
    #     for i in range(self.NUM_DRONES):
    #         if self.step_counter%self.IMG_CAPTURE_FREQ == 0:
    #             self.rgb[i], self.dep[i], self.seg[i] = self._getDroneImages(i)
    #             #### Printing observation to PNG frames example ############
    #             if self.RECORD:
    #                 # self._exportImage(img_type=ImageType.RGB, # ImageType.BW, ImageType.DEP, ImageType.SEG
    #                 #                   img_input=self.rgb[i],
    #                 #                   path=self.ONBOARD_IMG_PATH+"drone_"+str(i),
    #                 #                   frame_num=int(self.step_counter/self.IMG_CAPTURE_FREQ)
    #                 #                   )
    #                 self._exportImage(img_type=ImageType.SEG, # ImageType.BW, ImageType.DEP, ImageType.SEG
    #                                   img_input=self.seg[i],
    #                                   path=self.ONBOARD_IMG_PATH+"drone_"+str(i),
    #                                   frame_num=int(self.step_counter/self.IMG_CAPTURE_FREQ)
    #                                   )
    #         obs[str(i)] = {"state": self._getDroneStateVector(i), \
    #                        "neighbors": adjacency_mat[i,:], \
    #                        "rgb": self.rgb[i], \
    #                        "dep": self.dep[i], \
    #                        "seg": self.seg[i] \
    #                        }
    #     return obs

    def _computeObs(self):
        """Returns the current observation of the environment.

        For the value of the state, see the implementation of `_getDroneStateVector()`.

        Returns
        -------
        ndarray
            An ndarray of shape (NUM_DRONES, 20) with the state of each drone.

        """
        return np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])

    ################################################################################
    
    # def _preprocessAction(self,
    #                       action
    #                       ):
    #     """Pre-processes the action passed to `.step()` into motors' RPMs.

    #     Clips and converts a dictionary into a 2D array.

    #     Parameters
    #     ----------
    #     action : ndarray
    #         The (unbounded) input action for each drone, to be translated into feasible RPMs.

    #     Returns
    #     -------
    #     ndarray
    #         (NUM_DRONES, 4)-shaped array of ints containing to clipped RPMs
    #         commanded to the 4 motors of each drone.

    #     """
    #     return np.array([np.clip(action[i, :], 0, self.MAX_RPM) for i in range(self.NUM_DRONES)])
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
        for k in range(action.shape[0]):
            #### Get the current state of the drone  ###################
            state = self._getDroneStateVector(k)
            target_v = action[k, :]
            #### Normalize the first 3 components of the target velocity
            if np.linalg.norm(target_v[0:3]) != 0:
                v_unit_vector = target_v[0:3] / np.linalg.norm(target_v[0:3])
            else:
                v_unit_vector = np.zeros(3)
            temp, _, _ = self.ctrl[k].computeControl(control_timestep=self.CTRL_TIMESTEP,
                                                    cur_pos=state[0:3],
                                                    cur_quat=state[3:7],
                                                    cur_vel=state[10:13],
                                                    cur_ang_vel=state[13:16],
                                                    target_pos=state[0:3], # same as the current position
                                                    target_rpy=np.array([0,0,state[9]]), # keep current yaw
                                                    target_vel=self.SPEED_LIMIT * np.abs(target_v[3]) * v_unit_vector # target the desired velocity vector
                                                    )
            rpm[k,:] = temp
        return rpm

    ################################################################################

    def _computeReward(self):
        """Computes the current reward value(s).

        Unused as this subclass is not meant for reinforcement learning.

        Returns
        -------
        int
            Dummy value.

        """
        return -1

        ################################################################################
    
    def _computeTerminated(self):
        """Computes the current terminated value(s).

        Unused as this subclass is not meant for reinforcement learning.

        Returns
        -------
        bool
            Dummy value.

        """
        return False
    
    ################################################################################
    
    def _computeTruncated(self):
        """Computes the current truncated value(s).

        Unused as this subclass is not meant for reinforcement learning.

        Returns
        -------
        bool
            Dummy value.

        """
        return False

    ################################################################################
    
    # def _computeDone(self):
    #     """Computes the current done value(s).

    #     Unused as this subclass is not meant for reinforcement learning.

    #     Returns
    #     -------
    #     bool
    #         Dummy value.

    #     """
    #     return False

    ################################################################################
    
    def _computeInfo(self):
        """Computes the current info dict(s).

        Unused as this subclass is not meant for reinforcement learning.

        Returns
        -------
        dict[str, int]
            Dummy value.

        """
        return {"answer": 42} #### Calculated by the Deep Thought supercomputer in 7.5M years
