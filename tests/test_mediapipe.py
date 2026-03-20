
import cv2
import mediapipe as mp
import time
import os
import numpy as np

def test_init():
    try:
        OBJECT_DETECTOR_MODEL = "models/efficientdet_lite0.tflite"
        GESTURE_RECOGNIZER_MODEL = "models/gesture_recognizer.task"

        BaseOptions = mp.tasks.BaseOptions
        ObjectDetector = mp.tasks.vision.ObjectDetector
        ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions
        GestureRecognizer = mp.tasks.vision.GestureRecognizer
        GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        print("Testing ObjectDetector init...")
        options_objects = ObjectDetectorOptions(
            base_options=BaseOptions(model_asset_path=OBJECT_DETECTOR_MODEL),
            running_mode=VisionRunningMode.VIDEO,
            score_threshold=0.5
        )
        detector = ObjectDetector.create_from_options(options_objects)
        print("ObjectDetector init OK")

        print("Testing GestureRecognizer init...")
        options_gestures = GestureRecognizerOptions(
            base_options=BaseOptions(model_asset_path=GESTURE_RECOGNIZER_MODEL),
            running_mode=VisionRunningMode.VIDEO,
        )
        recognizer = GestureRecognizer.create_from_options(options_gestures)
        print("GestureRecognizer init OK")

        print("Testing inference on dummy frame...")
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=dummy_frame)
        timestamp_ms = int(time.time() * 1000)

        detector.detect_for_video(mp_image, timestamp_ms)
        recognizer.recognize_for_video(mp_image, timestamp_ms)
        print("Inference OK")

        detector.close()
        recognizer.close()
        print("All tests passed!")

    except Exception as e:
        print(f"FAILED with error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_init()
