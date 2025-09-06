import numpy as np
import os

ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

ROBOT_MODEL_FILE = "../assets/cassie.xml"

ACTUATED_JOINT_RANGE = np.array(
    [
        [
            -0.2618,
            -0.3927,
            -0.8727,
            -2.8623,
            -2.4435,
            -0.3927,
            -0.3927,
            -0.8727,
            -2.8623,
            -2.4435,
        ],
        [
            0.3927,
            0.3927,
            1.3963,
            -0.6458,
            -0.5236,
            0.2618,
            0.3927,
            1.3963,
            -0.6458,
            -0.5236,
        ],
    ]
)


FALLING_THRESHOLD = 0.55
TARSUS_HITGROUND_THRESHOLD = 0.15

DEFAULT_PGAIN = np.array([400, 200, 200, 500, 20, 400, 200, 200, 500, 20])
DEFAULT_DGAIN = np.array([4, 4, 10, 20, 4, 4, 4, 10, 20, 4])

# NOTE: this safe torque range is used in the cassiemujoco_ctypes, hardcoded, this is not used in the python env
TORQUE_LB = np.array([-80.0, -60.0, -80.0, -190.0, -45.0, -80.0, -60.0, -80.0, -190.0, -45.0])
TORQUE_UB = np.array([80.0, 60.0, 80.0, 190.0, 45.0, 80.0, 60.0, 80.0, 190.0, 45.0])


STANDING_POSE = np.array(
    [
        0.0,
        0.0,
        0.95,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.4544,
        -1.21,
        -1.643,
        0.0,
        0.0,
        0.4544,
        -1.21,
        -1.643,
    ]
)
