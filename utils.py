# utils.py

import cv2
import numpy as np
from PIL import Image, ExifTags
from loguru import logger

def load_and_correct_orientation(image_path):
    """
    Loads an image, corrects its orientation based on EXIF data,
    and returns it as an OpenCV-compatible BGR NumPy array.
    """
    try:
        image = Image.open(image_path)

        # Check for EXIF data and orientation tag
        if hasattr(image, '_getexif'):
            exif = image._getexif()
            if exif is not None:
                # Find the orientation tag
                for tag, value in exif.items():
                    tag_name = ExifTags.TAGS.get(tag, tag)
                    if tag_name == 'Orientation':
                        # Apply the corresponding rotation
                        if value == 3:
                            image = image.rotate(180, expand=True)
                            logger.info(f"Corrected orientation for {image_path}: Rotated 180°")
                        elif value == 6:
                            image = image.rotate(270, expand=True)
                            logger.info(f"Corrected orientation for {image_path}: Rotated 270°")
                        elif value == 8:
                            image = image.rotate(90, expand=True)
                            logger.info(f"Corrected orientation for {image_path}: Rotated 90°")
                        break
    except Exception as e:
        logger.error(f"Error processing image orientation for {image_path}: {e}")
        # Fallback to cv2.imread if Pillow fails
        return cv2.imread(image_path)

    # Convert from Pillow's RGB to OpenCV's BGR format
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def draw_bounding_box(frame, face_data, is_recognized, info=None, similarity=None, config=None):
    """Draws bounding boxes and text on the frame."""
    box = face_data[0:4].astype(int)
    (startX, startY, w, h) = box
    endX, endY = startX + w, startY + h
    
    color = tuple(config['green_color']) if is_recognized else tuple(config['red_color'])
    
    cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
    
    if is_recognized and info:
        text = f"ID: {info['employee_id']} ({info['name']})"
        similarity_text = f"Conf: {similarity:.2f}"
        
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.putText(frame, text, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, tuple(config['text_color']), 2, cv2.LINE_AA)
        cv2.putText(frame, similarity_text, (startX, y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, tuple(config['text_color']), 2, cv2.LINE_AA)
    else:
        text = "Unknown"
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.putText(frame, text, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)