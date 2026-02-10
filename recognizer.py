import cv2
import numpy as np
import threading
from sklearn.preprocessing import normalize
from logger_config import recognition_logger as logger

class FaceRecognizer:
    """Shared ArcFace inference engine."""
    def __init__(self, config):
        self.config = config
        self.lock = threading.Lock()
        try:
            self.model = cv2.dnn.readNetFromONNX(config['recognizer_model_path'])
            # Set to CPU optimization
            self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            logger.info("Global ArcFace Model loaded.")
        except Exception as e:
            logger.error(f"Failed to load ArcFace: {e}")
            raise

        self.reference_landmarks = np.array([
            [38.2946, 51.6963], [73.5318, 51.5014],
            [56.0252, 71.7366], [41.5493, 92.3655],
            [70.7299, 92.2041]
        ], dtype=np.float32)

    def _align_face(self, frame, landmarks):
        transform_matrix, _ = cv2.estimateAffinePartial2D(landmarks, self.reference_landmarks)
        if transform_matrix is None: return None
        return cv2.warpAffine(frame, transform_matrix, (112, 112), borderValue=0.0)

    def process_face(self, frame, face_data):
        """Align and get embedding. Thread-safe."""
        landmarks = face_data[4:14].reshape(5, 2).astype(np.float32)
        aligned_face = self._align_face(frame, landmarks)
        if aligned_face is None: return None
        
        blob = cv2.dnn.blobFromImage(aligned_face, 1.0/127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=True)
        
        with self.lock:
            self.model.setInput(blob)
            embedding = self.model.forward()
        
        return normalize(embedding).flatten()