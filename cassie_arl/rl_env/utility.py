import numpy as np
import math
from six.moves import cPickle as pickle
from scipy.special import comb
import mujoco


# # roll, pitch, yaw: rx, ry, rz
# def euler2quat(rxryrz):
#     rx = rxryrz[0]
#     ry = rxryrz[1]
#     rz = rxryrz[2]
#     q = np.vstack(
#         [
#             np.cos(rx / 2) * np.cos(ry / 2) * np.cos(rz / 2)
#             + np.sin(rx / 2) * np.sin(ry / 2) * np.sin(rz / 2),
#             np.sin(rx / 2) * np.cos(ry / 2) * np.cos(rz / 2)
#             - np.cos(rx / 2) * np.sin(ry / 2) * np.sin(rz / 2),
#             np.cos(rx / 2) * np.sin(ry / 2) * np.cos(rz / 2)
#             + np.sin(rx / 2) * np.cos(ry / 2) * np.sin(rz / 2),
#             np.cos(rx / 2) * np.cos(ry / 2) * np.sin(rz / 2)
#             - np.sin(rx / 2) * np.sin(ry / 2) * np.cos(rz / 2),
#         ]
#     )
#     return q.ravel()

def euler2quat(euler: np.ndarray) -> np.ndarray:
    """
    Convert Euler angles (XYZ order, radians) to quaternion [w, x, y, z].
    
    Args:
        euler (array-like): 3-element array of Euler angles [rx, ry, rz] in radians.
    
    Returns:
        np.ndarray: 4-element quaternion [w, x, y, z].
    """
    quat = np.zeros(4, dtype=np.float64)
    mujoco.mju_euler2Quat(quat, np.asarray(euler, dtype=np.float64))
    return quat


# def quat2euler(q):
#     w = q[0]
#     x = q[1]
#     y = q[2]
#     z = q[3]

#     sinr_cosp = 2 * (w * x + y * z)
#     cosr_cosp = 1 - 2 * (x**2 + y**2)
#     roll = np.arctan2(sinr_cosp, cosr_cosp)

#     sinp = 2 * (w * y - z * x)
#     # if sinp<-1 or sinp >1:
#     #     i=1
#     sinp = np.clip(sinp, -1, 1)
#     pitch = np.where(np.abs(sinp) >= 1, np.sign(sinp) * np.pi / 2, np.arcsin(sinp))

#     siny_cosp = 2 * (w * z + x * y)
#     cosy_cosp = 1 - 2 * (y**2 + z**2)
#     yaw = np.arctan2(siny_cosp, cosy_cosp)

#     return np.vstack([roll, pitch, yaw]).ravel()

def quat2euler(q: np.ndarray) -> np.ndarray:
    """
    Convert a quaternion [w, x, y, z] to Euler angles (XYZ order, radians).
    
    Args:
        q (array-like): 4-element quaternion [w, x, y, z]
    
    Returns:
        np.ndarray: 3-element Euler angles [roll, pitch, yaw] in radians
    """
    euler = np.zeros(3, dtype=np.float64)
    mujoco.mju_quat2Euler(euler, np.asarray(q, dtype=np.float64))
    return euler


def save_dict(di_, filename_):
    with open(filename_, "wb") as f:
        pickle.dump(di_, f)


def load_dict(filename_):
    with open(filename_, "rb") as f:
        ret_di = pickle.load(f)
    return ret_di


def dbezier(coeff, s):
    dcoeff = __diff_coeff(coeff)
    fcn = bezier(dcoeff, s)
    return fcn


def __binomial(i, n):
    """Binomial coefficient"""
    return math.factorial(n) / float(math.factorial(i) * math.factorial(n - i))


def __bernstein(t, i, n):
    """Bernstein polynom"""
    return __binomial(i, n) * (t**i) * ((1 - t) ** (n - i))


def bezier(coeff, s):
    """Calculate coordinate of a point in the bezier curve"""
    n, m = coeff.shape[0], coeff.shape[1]
    m = m - 1
    fcn = np.zeros((n, 1))
    for k in range(m + 1):
        fcn += coeff[:, k].reshape((n, 1)) * __bernstein(s, k, m)
    return fcn.reshape((n,))


def __diff_coeff(coeff):
    M = coeff.shape[1] - 1
    A = np.zeros((M, M + 1))

    for i in range(M):
        A[i, i] = -(M - i) * comb(M, i) / comb(M - 1, i)
        A[i, i + 1] = (i + 1) * comb(M, i + 1) / comb(M - 1, i)

    A[M - 1, M] = M * comb(M, M)
    dcoeff = coeff @ (A.T)
    return dcoeff


def first_order_filter(prev, new, para):
    return prev * (1 - para) + new * para


def rotation_matrix(rot_vec):
    # Precompute sines and cosines
    roll, pitch, yaw = rot_vec[0], rot_vec[1], rot_vec[2]
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    # Construct the combined rotation matrix in ZYX order
    R = np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )
    return R


def global2local(vector, rot_vec):
    # Compute the rotation matrix
    R = rotation_matrix(rot_vec)

    # Transform the vector to the local frame by applying the transpose of R
    local_vector = R.T @ vector
    return local_vector


def local2global(vector, rot_vec):
    # Get rotation matrix
    R = rotation_matrix(rot_vec)

    # Transform the vector to the global frame by applying R
    global_vector = R @ vector
    return global_vector


def local2global_yawonly(vector, yaw):
    # Precompute cosine and sine of the yaw angle
    cy = np.cos(yaw)
    sy = np.sin(yaw)

    # Directly calculate the global vector components
    x_global = vector[0] * cy - vector[1] * sy
    y_global = vector[0] * sy + vector[1] * cy

    return np.array([x_global, y_global])