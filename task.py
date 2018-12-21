import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 
        

    def get_reward(self, done):
        """Uses current pose of sim to return reward."""
        # reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()
        
        """
        Description
        --------------
        Take-off: this reward function is trying to represent the take-off task.
        It rewards the quadricopter if it stays in the area inside a "conic coordinates", with its peak in (0, 0, 0)
            and its base at z=20. The higher the elevation, more points it gets based on a reward multiplier

        Parameters
        --------------    
            self.sim.pose[0:2]: x and y axis represents the ground
            self.sim.pose[2]: z axis represents the elevation
            radius: represents the radius of the conic base
            height: represents the height of the conic
        """
        # input parameters
        best_radius = 4
        max_radius = best_radius * 2
        height = self.target_pos[2]
        reward_mult = 3
        achievement = False

        # check if it is on a reasonable height
        check_height = self.sim.pose[2] <= height * 1.5
        
        # penalty if it went to far from the axis z
        dz = self.sim.pose[0]**2 + self.sim.pose[1]**2
        check_radius = (dz <= best_radius**2)
        if check_radius:
            z_penalty = 0
        else:
            z_penalty = max(0, (max_radius**2 - dz) / (max_radius**2 - best_radius**2))
        radius_penalty = 1 - z_penalty
        
        # reward function
        reward = min(self.sim.pose[2], height)**reward_mult * radius_penalty * check_height

        # if it is stable for the hole episode (3 poses), it passes and episode ends
        self.stability_score = self.stability_score[:-1]
        if reward > 0 and self.sim.pose[2] >= 0.8 * height:
            self.stability_score = np.concatenate([[1], self.stability_score])
            if np.mean(self.stability_score) == 1:
                done = True
                achievement = True
        else:
            self.stability_score = np.concatenate([[0], self.stability_score])

        return reward, done, achievement


    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        reward_aux = 0
        pose_all = []
        self.stability_score = [0, 0, 0]

        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward_aux, done, achievement = self.get_reward(done)
            reward += reward_aux
            pose_all.append(self.sim.pose)

        next_state = np.concatenate(pose_all)
        return next_state, reward, done, achievement

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state