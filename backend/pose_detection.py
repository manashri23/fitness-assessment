import os
import cv2
import numpy as np
import time

try:
    import mediapipe as mp
    from mediapipe.tasks.python import vision
    MP_AVAILABLE = True
except ImportError:
    MP_AVAILABLE = False


# MediaPipe 33 Landmark Names Map
LANDMARK_NAMES = {
    0: "NOSE",
    1: "LEFT_EYE_INNER", 2: "LEFT_EYE", 3: "LEFT_EYE_OUTER",
    4: "RIGHT_EYE_INNER", 5: "RIGHT_EYE", 6: "RIGHT_EYE_OUTER",
    7: "LEFT_EAR", 8: "RIGHT_EAR",
    9: "MOUTH_LEFT", 10: "MOUTH_RIGHT",
    11: "LEFT_SHOULDER", 12: "RIGHT_SHOULDER",
    13: "LEFT_ELBOW", 14: "RIGHT_ELBOW",
    15: "LEFT_WRIST", 16: "RIGHT_WRIST",
    17: "LEFT_PINKY", 18: "RIGHT_PINKY",
    19: "LEFT_INDEX", 20: "RIGHT_INDEX",
    21: "LEFT_THUMB", 22: "RIGHT_THUMB",
    23: "LEFT_HIP", 24: "RIGHT_HIP",
    25: "LEFT_KNEE", 26: "RIGHT_KNEE",
    27: "LEFT_ANKLE", 28: "RIGHT_ANKLE",
    29: "LEFT_HEEL", 30: "RIGHT_HEEL",
    31: "LEFT_FOOT_INDEX", 32: "RIGHT_FOOT_INDEX"
}

# Standard MediaPipe Skeleton Connections
SKELETON_CONNECTIONS = [
    # Face
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
    # Torso
    (11, 12), (11, 23), (12, 24), (23, 24),
    # Left Arm
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    # Right Arm
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    # Left Leg
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    # Right Leg
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32)
]


class PoseDetector:
    """
    Robust MediaPipe Pose Landmarker that processes camera / video frames.
    Supports MediaPipe Tasks API (vision.PoseLandmarker) and legacy solutions API.
    Performs letterbox aspect padding to ensure accurate detection on all camera resolutions.
    """
    def __init__(self, model_filename="pose_landmarker_full.task"):
        self.landmarker = None
        self.legacy_pose = None
        self.is_initialized = False

        if MP_AVAILABLE:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(base_dir, model_filename)
            if not os.path.exists(model_path):
                # Fallback to lite task if full is not found
                model_path = os.path.join(base_dir, "pose_landmarker_lite.task")

            if os.path.exists(model_path):
                try:
                    self.landmarker = vision.PoseLandmarker.create_from_model_path(model_path)
                    self.is_initialized = True
                    print(f"[PoseDetector] Initialized MediaPipe Tasks PoseLandmarker ({os.path.basename(model_path)})")
                except Exception as e:
                    print(f"[PoseDetector] Warning: Could not initialize Tasks PoseLandmarker: {e}")

            # Legacy MediaPipe solutions check
            if not self.is_initialized and hasattr(mp, 'solutions') and hasattr(mp.solutions, 'pose'):
                try:
                    self.legacy_pose = mp.solutions.pose.Pose(
                        static_image_mode=False,
                        model_complexity=1,
                        smooth_landmarks=True,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5
                    )
                    self.is_initialized = True
                    print("[PoseDetector] Initialized MediaPipe Legacy Solutions Pose")
                except Exception as e:
                    print(f"[PoseDetector] Warning: Legacy pose failed: {e}")

    def process_frame(self, frame, draw=True):
        """
        Processes a single BGR OpenCV frame.
        Returns:
            processed_frame (np.ndarray): Frame with skeleton overlay.
            landmarks_dict (dict): Dictionary mapping landmark index to {name, x, y, z, px, py, visibility}.
        """
        if frame is None or frame.size == 0:
            return frame, {}

        h, w, _ = frame.shape
        landmarks_dict = {}

        # 1. Preferred Path: MediaPipe Tasks API with Letterbox Square Padding
        if self.landmarker is not None:
            max_dim = max(h, w)
            square_frame = np.zeros((max_dim, max_dim, 3), dtype=np.uint8)
            pad_x = (max_dim - w) // 2
            pad_y = (max_dim - h) // 2
            square_frame[pad_y:pad_y + h, pad_x:pad_x + w] = frame

            img_rgb = cv2.cvtColor(square_frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

            try:
                detection_result = self.landmarker.detect(mp_img)
                if detection_result.pose_landmarks and len(detection_result.pose_landmarks) > 0:
                    raw_landmarks = detection_result.pose_landmarks[0]
                    for idx, lm in enumerate(raw_landmarks):
                        # Convert normalized square coordinate to pixel in square frame
                        px_sq = lm.x * max_dim
                        py_sq = lm.y * max_dim

                        # Unpad to original frame pixel coordinate
                        px = int(px_sq - pad_x)
                        py = int(py_sq - pad_y)

                        # Normalized coordinate in original frame (0.0 to 1.0)
                        norm_x = max(0.0, min(1.0, float(px) / float(w)))
                        norm_y = max(0.0, min(1.0, float(py) / float(h)))

                        visibility = getattr(lm, "visibility", 1.0) or 1.0

                        landmarks_dict[idx] = {
                            "name": LANDMARK_NAMES.get(idx, f"LANDMARK_{idx}"),
                            "x": norm_x,
                            "y": norm_y,
                            "z": getattr(lm, "z", 0.0),
                            "px": px,
                            "py": py,
                            "visibility": float(visibility)
                        }

                    if draw:
                        self._draw_custom_skeleton(frame, landmarks_dict)

                    return frame, landmarks_dict
                else:
                    # No human body detected in this frame
                    return frame, {}
            except Exception as e:
                print(f"[PoseDetector] Detection error: {e}")
                return frame, {}

        # 2. Legacy Solutions Path
        elif self.legacy_pose is not None:
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.legacy_pose.process(img_rgb)
            if results.pose_landmarks:
                for idx, lm in enumerate(results.pose_landmarks.landmark):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    landmarks_dict[idx] = {
                        "name": LANDMARK_NAMES.get(idx, f"LANDMARK_{idx}"),
                        "x": float(lm.x),
                        "y": float(lm.y),
                        "z": float(lm.z),
                        "px": cx,
                        "py": cy,
                        "visibility": float(lm.visibility)
                    }
                if draw:
                    self._draw_custom_skeleton(frame, landmarks_dict)
                return frame, landmarks_dict
            return frame, {}

        # 3. Fallback only if no mediapipe model available
        return frame, {}

    def _draw_custom_skeleton(self, frame, landmarks_dict):
        """Draws aesthetic neon skeleton connections and joint keypoints."""
        # Draw connection lines (Neon Cyan BGR: 254, 242, 0)
        for start_idx, end_idx in SKELETON_CONNECTIONS:
            pt1 = landmarks_dict.get(start_idx)
            pt2 = landmarks_dict.get(end_idx)
            if pt1 and pt2:
                if pt1.get("visibility", 1.0) > 0.35 and pt2.get("visibility", 1.0) > 0.35:
                    cv2.line(frame, (pt1["px"], pt1["py"]), (pt2["px"], pt2["py"]), (254, 242, 0), 3)

        # Draw joint nodes (Emerald Green BGR: 129, 185, 16)
        for idx, pt in landmarks_dict.items():
            if pt.get("visibility", 1.0) > 0.35:
                cv2.circle(frame, (pt["px"], pt["py"]), 5, (129, 185, 16), -1)
                cv2.circle(frame, (pt["px"], pt["py"]), 7, (255, 255, 255), 1)
