

import cv2
import yaml
import numpy as np
from loguru import logger
from collections import deque
from sklearn.preprocessing import normalize

from detector import FaceDetector
from recognizer import FaceRecognizer
from tracker import CentroidTracker
from utils import draw_bounding_box
from quality_checker import QualityChecker

def main():
    """
    Main function for the professional, real-time face recognition application.
    Implements:
    1. Advanced Quality Filtering (pose, brightness, blur).
    2. Temporal Smoothing (multi-frame embedding averaging).
    3. L2 Normalization for all embeddings.
    4. Adaptive Similarity Thresholds based on face clarity.
    """
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    logger.add("logs/app.log", rotation="10 MB")
    logger.info("Application starting with professional-grade features.")

    detector = FaceDetector(config)
    recognizer = FaceRecognizer(config)
    tracker = CentroidTracker(max_disappeared=config['max_disappeared_frames'])
    quality_checker = QualityChecker(config)

    cap = cv2.VideoCapture(config['camera_url'])
    if not cap.isOpened():
        logger.error(f"Error: Could not open camera stream at {config['camera_url']}")
        return
    retry_attempts = 5
    for attempt in range(retry_attempts):
        if cap.isOpened():
            break
        logger.warning(f"Retrying to open camera stream... Attempt {attempt + 1}/{retry_attempts}")
        cap.open(config['camera_url'])

    tracked_faces = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.warning("End of stream or cannot read frame."); break

        faces = detector.detect(frame)
        rects = [face[0:4].astype(int) for face in faces]
        tracked_objects = tracker.update(rects)

        current_frame_faces = {}
        for face_data in faces:
            box = face_data[0:4].astype(int)
            cX = int(box[0] + box[2] / 2.0)
            cY = int(box[1] + box[3] / 2.0)
            min_dist, matched_id = float('inf'), None
            for (object_id, centroid) in tracked_objects.items():
                dist = np.linalg.norm(np.array(centroid) - np.array((cX, cY)))
                if dist < min_dist:
                    min_dist, matched_id = dist, object_id
            if matched_id is not None:
                current_frame_faces[matched_id] = face_data

        for object_id, centroid in tracked_objects.items():
            if object_id not in tracked_faces:
                tracked_faces[object_id] = {
                    "info": None, "similarity": 0.0, "box": None,
                    "embedding_history": deque(maxlen=config['temporal_smoothing_window']),
                    "status": "Tracking"
                }

            if object_id in current_frame_faces:
                face_data = current_frame_faces[object_id]
                tracked_faces[object_id]["box"] = face_data[0:4].astype(int)

                # --- 1. LIVE QUALITY CHECK ---
                # --- FIX #1: Correctly unpack the 'metrics' dictionary ---
                is_good, reasons, metrics = quality_checker.check_quality(frame, face_data)
                
                if not is_good:
                    tracked_faces[object_id]["status"] = f"Poor Quality: {reasons[0]}"
                    continue
                
                embedding = recognizer.process_face(frame, face_data)
                tracked_faces[object_id]["embedding_history"].append(embedding.flatten())
                
                history = tracked_faces[object_id]["embedding_history"]
                if len(history) < config['min_embeddings_for_recognition']:
                    tracked_faces[object_id]["status"] = f"Gathering frames ({len(history)}/{config['min_embeddings_for_recognition']})"
                    continue

                avg_embedding = normalize(np.mean(history, axis=0).reshape(1, -1))

                # --- FIX #2: Access 'clarity' from the now-existing 'metrics' dictionary ---
                clarity = metrics['clarity']

                # --- 4. ADAPTIVE SIMILARITY THRESHOLD ---
                base_threshold = config['recognition_threshold']
                current_threshold = base_threshold
                if clarity < config['clarity_threshold_for_adaptation']:
                    current_threshold += config['adaptive_threshold_increase']
                
                info, similarity = recognizer.recognize(avg_embedding)
                
                if info and similarity > current_threshold:
                    tracked_faces[object_id].update({
                        "info": info,
                        "similarity": similarity,
                        "status": f"Recognized (Thresh: {current_threshold:.2f})"
                    })
                else:
                    tracked_faces[object_id].update({
                        "info": None,
                        "similarity": 0.0,
                        "status": "Unknown"
                    })

        active_ids = list(tracked_objects.keys())
        tracked_faces = {id: data for id, data in tracked_faces.items() if id in active_ids}

        for object_id, data in tracked_faces.items():
            if data["box"] is not None:
                is_recognized = data["info"] is not None
                draw_bounding_box(frame, data["box"], is_recognized, data["info"], data["similarity"], config)
                # Optional: Display status text for debugging
                # (x, y, w, h) = data["box"]
                # cv2.putText(frame, data["status"], (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        cv2.imshow("Professional Face Recognition System", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    logger.info("Application shut down.")

if __name__ == "__main__":
    main()

