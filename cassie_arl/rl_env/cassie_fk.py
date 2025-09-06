import numpy as np
import kinpy as kp
from kinpy.transform import Transform
from configs.defaults import ROOT_PATH

'''
Forward kinematics library for Cassie
'''


class CassieFK:
    def __init__(self):
        self.cassie_tf_tree = kp.build_chain_from_urdf(
            open(ROOT_PATH + "/assets/cassie.urdf").read()
        )

    def get_foot_pos(self, motor_pos, pelvis_pos, pelvis_rot):
        # input rot can be either quat or euler
        root = Transform(rot=pelvis_rot, pos=pelvis_pos)
        th = {
            "hip_abduction_left": motor_pos[0],
            "hip_rotation_left": motor_pos[1],
            "hip_flexion_left": motor_pos[2],
            "knee_joint_left": motor_pos[3],
            "knee_to_shin_left": 0,
            "ankle_joint_left": np.radians(13) - motor_pos[3],
            "toe_joint_left": motor_pos[4],
            "hip_abduction_right": motor_pos[5],
            "hip_rotation_right": motor_pos[6],
            "hip_flexion_right": motor_pos[7],
            "knee_joint_right": motor_pos[8],
            "knee_to_shin_right": 0,
            "ankle_joint_right": np.radians(13) - motor_pos[8],
            "toe_joint_right": motor_pos[9],
        }
        ret = self.cassie_tf_tree.forward_kinematics(th, world=root)
        foot_pos = np.concatenate([ret["left_toe"].pos, ret["right_toe"].pos])
        # foot_height = np.array([ret['left_toe'].pos[2], ret['right_toe'].pos[2]])
        return foot_pos

    def get_tarsus_pos(self, motor_pos, pelvis_pos, pelvis_rot):
        root = Transform(rot=pelvis_rot, pos=pelvis_pos)
        th = {
            "hip_abduction_left": motor_pos[0],
            "hip_rotation_left": motor_pos[1],
            "hip_flexion_left": motor_pos[2],
            "knee_joint_left": motor_pos[3],
            "knee_to_shin_left": 0,
            "ankle_joint_left": np.radians(13) - motor_pos[3],
            "toe_joint_left": motor_pos[4],
            "hip_abduction_right": motor_pos[5],
            "hip_rotation_right": motor_pos[6],
            "hip_flexion_right": motor_pos[7],
            "knee_joint_right": motor_pos[8],
            "knee_to_shin_right": 0,
            "ankle_joint_right": np.radians(13) - motor_pos[8],
            "toe_joint_right": motor_pos[9],
        }
        ret = self.cassie_tf_tree.forward_kinematics(th, world=root)
        tarsus_pos = np.concatenate([ret["left_tarsus"].pos, ret["right_tarsus"].pos])
        return tarsus_pos

    def get_foot_and_tarsus_pos(self, motor_pos, pelvis_pos, pelvis_rot):
        root = Transform(rot=pelvis_rot, pos=pelvis_pos)
        th = {
            "hip_abduction_left": motor_pos[0],
            "hip_rotation_left": motor_pos[1],
            "hip_flexion_left": motor_pos[2],
            "knee_joint_left": motor_pos[3],
            "knee_to_shin_left": 0,
            "ankle_joint_left": np.radians(13) - motor_pos[3],
            "toe_joint_left": motor_pos[4],
            "hip_abduction_right": motor_pos[5],
            "hip_rotation_right": motor_pos[6],
            "hip_flexion_right": motor_pos[7],
            "knee_joint_right": motor_pos[8],
            "knee_to_shin_right": 0,
            "ankle_joint_right": np.radians(13) - motor_pos[8],
            "toe_joint_right": motor_pos[9],
        }
        ret = self.cassie_tf_tree.forward_kinematics(th, world=root)
        tarsus_pos = np.concatenate([ret["left_tarsus"].pos, ret["right_tarsus"].pos])
        foot_pos = np.concatenate([ret["left_toe"].pos, ret["right_toe"].pos])
        return foot_pos, tarsus_pos
    