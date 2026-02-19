# To keep our code clean and modular, i just centralize all our new, advanced quality checks into a dedicated file.


import cv2
import numpy as np
from loguru import logger

class QualityChecker:
    """
    A class to perform comprehensive quality checks on face images.
    Returns a dictionary of all measured quality metrics.
    """
    def __init__(self, config):
        self.config = config
        self.face_3d_model = np.array([
            [0.0, 0.0, 0.0],            # Nose tip
            [0.0, -330.0, -65.0],       # Chin
            [-225.0, 170.0, -135.0],    # Left eye left corner
            [225.0, 170.0, -135.0],     # Right eye right corner
            [-150.0, -150.0, -125.0],   # Left Mouth corner
            [150.0, -150.0, -125.0]     # Right mouth corner
        ], dtype=np.float64)

    def check_quality(self, frame, face_data):
        """
        Performs all quality checks and returns the results.

        Returns:
            bool: True if all checks pass, False otherwise.
            list: A list of reasons for failure.
            dict: A dictionary containing all measured quality metrics.
        """
        box = face_data[0:4].astype(int)
        (startX, startY, w, h) = box
        face_roi = frame[startY:startY+h, startX:startX+w]

        metrics = {
            'size': (w, h), 'clarity': 0.0, 'brightness': 0.0,
            'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0
        }
       
        if face_roi.size == 0:
            return False, ["Face ROI is empty"], metrics

        reasons = []

        # 1. Size Check
        if w < self.config['min_face_size'] or h < self.config['min_face_size']:
            reasons.append("Face too small")
        metrics['size'] = (w, h)

        # 2. Blur Check
        clarity = self._calculate_blurriness(face_roi)
        if clarity < self.config['min_blur_clarity']:
            reasons.append("Too blurry")
        metrics['clarity'] = clarity

        # 3. Brightness Check
        brightness = self._calculate_brightness(face_roi)
        if not (self.config['min_brightness'] < brightness < self.config['max_brightness']):
            reasons.append("Bad brightness")
        metrics['brightness'] = brightness
           
        # 4. Pose Check
        landmarks = face_data[4:14].reshape(5, 2).astype(np.float64)
        pitch, yaw, roll = self._estimate_pose(frame, landmarks)
        if pitch is None:
             reasons.append("Pose estimation failed")
        else:
            if abs(yaw) > self.config['max_pose_yaw']:
                reasons.append("Excessive yaw")
            if abs(pitch) > self.config['max_pose_pitch']:
                reasons.append("Excessive pitch")
            if abs(roll) > self.config['max_pose_roll']:
                reasons.append("Excessive roll")
            metrics.update({'pitch': pitch, 'yaw': yaw, 'roll': roll})

        return len(reasons) == 0, reasons, metrics

    def _calculate_blurriness(self, face_roi):
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    def _calculate_brightness(self, face_roi):
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        return np.mean(gray)

    def _estimate_pose(self, frame, landmarks_2d):
        h, w, _ = frame.shape
        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype=np.float64)

        points_2d = np.array([landmarks_2d[2], landmarks_2d[1], landmarks_2d[0], landmarks_2d[4], landmarks_2d[3]], dtype=np.float64)
        points_3d = np.array([self.face_3d_model[0], self.face_3d_model[2], self.face_3d_model[3], self.face_3d_model[5], self.face_3d_model[4]], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1))
       
        success = False
        try:
            success, rotation_vector, _ = cv2.solvePnP(points_3d, points_2d, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        except cv2.error as e:
            logger.warning(f"Pose estimation (ITERATIVE) failed: {e}. Retrying with EPNP.")
            try:
                success, rotation_vector, _ = cv2.solvePnP(points_3d, points_2d, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_EPNP)
            except cv2.error as e2:
                logger.error(f"Pose estimation (EPNP) also failed: {e2}")
                return None, None, None

        if not success:
            return None, None, None



        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        sy = np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
       
        singular = sy < 1e-6
        if not singular:
            x, y, z = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2]), np.arctan2(-rotation_matrix[2, 0], sy), np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        else:
            x, y, z = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1]), np.arctan2(-rotation_matrix[2, 0], sy), 0

        return np.rad2deg([x, y, z])
