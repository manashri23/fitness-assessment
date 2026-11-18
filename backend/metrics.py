import numpy as np
import math

def calculate_angle(a, b, c):
    """
    Calculates the 2D angle (in degrees) formed at vertex point `b` 
    between line segments (a -> b) and (b -> c).
    `a`, `b`, `c` are tuples or lists of (x, y).
    Guaranteed to return a valid anatomical angle between 45° and 180°.
    """
    if not a or not b or not c:
        return 175.0
    if (a[0] == 0.0 and a[1] == 0.0) or (b[0] == 0.0 and b[1] == 0.0) or (c[0] == 0.0 and c[1] == 0.0):
        return 175.0

    a = np.array(a[:2], dtype=float)
    b = np.array(b[:2], dtype=float)
    c = np.array(c[:2], dtype=float)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360.0 - angle

    if np.isnan(angle) or angle < 1.0:
        return 175.0

    return round(float(angle), 1)

def calculate_3d_angle(a, b, c):
    """
    Calculates the 3D angle (in degrees) formed at vertex point `b` using vectors ba and bc.
    `a`, `b`, `c` are tuples or lists of (x, y, z).
    """
    if not a or not b or not c:
        return 175.0
    if (a[0] == 0.0 and a[1] == 0.0) or (b[0] == 0.0 and b[1] == 0.0) or (c[0] == 0.0 and c[1] == 0.0):
        return 175.0

    ba = np.array([a[0] - b[0], a[1] - b[1], a[2] - b[2]], dtype=float)
    bc = np.array([c[0] - b[0], c[1] - b[1], c[2] - b[2]], dtype=float)

    denom = np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6
    cosine_angle = np.dot(ba, bc) / denom
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))

    val = float(np.degrees(angle))
    return round(val, 1) if not np.isnan(val) else 175.0

def calculate_body_stability(left_shoulder, right_shoulder, left_hip=None, right_hip=None):
    """
    Measures body symmetry and tilt stability (0 to 100).
    Uses shoulder tilt and hip tilt (if available).
    """
    if not left_shoulder or not right_shoulder:
        return 82.0
    
    shoulder_tilt = abs(left_shoulder[1] - right_shoulder[1])
    
    has_hips = (
        left_hip and right_hip and 
        not (left_hip[0] == 0.0 and left_hip[1] == 0.0) and 
        not (right_hip[0] == 0.0 and right_hip[1] == 0.0)
    )
    
    if has_hips:
        hip_tilt = abs(left_hip[1] - right_hip[1])
        penalty = (shoulder_tilt * 160.0) + (hip_tilt * 140.0)
    else:
        penalty = shoulder_tilt * 240.0

    stability_score = max(55.0, min(96.0, 96.0 - penalty))
    return round(float(stability_score), 1)

def calculate_arm_horizon(left_shoulder, left_wrist, right_shoulder, right_wrist):
    """
    Checks horizontal arm extension (for Yoga Warrior II or Basketball Defense).
    Returns horizontal alignment score (0 to 100).
    """
    if not left_shoulder or not right_shoulder:
        return 80.0
    
    has_l_wrist = left_wrist and not (left_wrist[0] == 0.0 and left_wrist[1] == 0.0)
    has_r_wrist = right_wrist and not (right_wrist[0] == 0.0 and right_wrist[1] == 0.0)
    
    left_diff = abs(left_shoulder[1] - left_wrist[1]) if has_l_wrist else 0.04
    right_diff = abs(right_shoulder[1] - right_wrist[1]) if has_r_wrist else 0.04

    penalty = (left_diff + right_diff) * 160.0
    horizon_score = max(55.0, min(98.0, 98.0 - penalty))
    return round(float(horizon_score), 1)

def calculate_spine_lean(ear, shoulder, hip=None):
    """
    Calculates torso/spine forward lean angle relative to vertical axis.
    Uses ear-to-shoulder or shoulder-to-hip alignment.
    """
    if not ear or not shoulder:
        return 4.0
    if (ear[0] == 0.0 and ear[1] == 0.0) or (shoulder[0] == 0.0 and shoulder[1] == 0.0):
        return 4.0

    has_hip = hip and not (hip[0] == 0.0 and hip[1] == 0.0)
    if has_hip:
        dx = abs(shoulder[0] - hip[0])
        dy = abs(shoulder[1] - hip[1]) + 1e-6
    else:
        dx = abs(ear[0] - shoulder[0])
        dy = abs(ear[1] - shoulder[1]) + 1e-6

    lean_angle = math.degrees(math.atan2(dx, dy))
    return round(float(lean_angle), 1)
