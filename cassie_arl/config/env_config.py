# config for test, play.py
config_play = {
    "max_timesteps": 2500,  # NOTE: max timestep per episode
    "is_visual": True,  # NOTE: set true to visualize robot
    "cam_track_robot": False,  # NOTE: set true to let camera track the robot
    "step_zerotorque": False,  # NOTE: set true to step without torque
    "minimal_rand": False,  # NOTE: set true to use minimal rand range -> only rand floor friction
    "is_noisy": True,  # NOTE: set true to add noise on observation
    "add_perturbation": False,  # NOTE: set true to add external perturbation
    "add_standing": True,  # NOTE: set true to include standing skill
    "fixed_gait": True,  # NOTE: set true to only test one cmd of 'fixed_gait_cmd'
    "add_rotation": False,  # NOTE: set true to add rotation (vyaw) command, only useful if not fixed gait
    "fixed_gait_cmd": [
        1.0,  # vx, m/s
        0.0,  # vy, m/s
        0.98,  # walking height, m
        0.0,  # turning rate, deg/s
    ],  # fixed gait cmd
}

# config for trainig, train.py
config_train = {
    "max_timesteps": 2500,  # NOTE: max timestep per episode
    "is_visual": False,  # NOTE: set true to visualize robot
    "cam_track_robot": False,  # NOTE: set true to let camera track the robot
    "step_zerotorque": False,  # NOTE: set true to step without torque
    "minimal_rand": False,  # NOTE: set true to use minimal rand range -> only rand floor friction
    "is_noisy": False,  # NOTE: set true to add noise on observation
    "add_perturbation": False,  # NOTE: set true to add external perturbation
    "add_standing": True,  # NOTE: set true to include standing skill
    "fixed_gait": True,  # NOTE: set true to only test one cmd of 'fixed_gait_cmd'
    "add_rotation": True,  # NOTE: set true to add rotation (vyaw) command, only useful if not fixed gait
    "fixed_gait_cmd": [
        0.0,
        0.0,
        0.98,
        0.0,
    ],  # fixed gait cmd: vx, vy, walking height, vyaw
}

# config for gaitlibary_test.py
config_static = {
    "max_timesteps": 2500,  # NOTE: max timestep per episode
    "is_visual": True,  # NOTE: set true to visualize robot
    "cam_track_robot": False,  # NOTE: set true to let camera track the robot
    "step_zerotorque": True,  # NOTE: set true to step without torque
    "minimal_rand": True,  # NOTE: set true to use minimal rand range -> only rand floor friction
    "is_noisy": False,  # NOTE: set true to add noise on observation
    "add_perturbation": False,  # NOTE: set true to add external perturbation
    "add_standing": False,  # NOTE: set true to include standing skill
    "fixed_gait": True,  # NOTE: set true to only test one cmd of 'fixed_gait_cmd'
    "add_rotation": False,  # NOTE: set true to add rotation (vyaw) command, only useful if not fixed gait
    "fixed_gait_cmd": [
        0.0,
        0.0,
        0.98,
        0.0,
    ],  # fixed gait cmd: vx, vy, walking height, vyaw
}
