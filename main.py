import time

import cv2
import mediapipe as mp
import numpy as np

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


BaseOptions = python.BaseOptions
FaceLandmarker = vision.FaceLandmarker
FaceLandmarkerOptions = vision.FaceLandmarkerOptions
RunningMode = vision.RunningMode


options = FaceLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="face_landmarker_v2_with_blendshapes.task"
    ),
    running_mode=RunningMode.VIDEO,
    num_faces=1,
)

detector = FaceLandmarker.create_from_options(options)


# MediaPipe face mesh iris points
LEFT_IRIS = [
    474,
    475,
    476,
    477,
]

RIGHT_IRIS = [
    469,
    470,
    471,
    472,
]


def draw_landmarks(frame, landmarks):
    h, w = frame.shape[:2]

    for i, lm in enumerate(landmarks):
        x = int(lm.x * w)
        y = int(lm.y * h)

        # normal face landmarks
        cv2.circle(
            frame,
            (x, y),
            1,
            (0, 255, 0),
            -1,
        )

    # iris points
    for idx in LEFT_IRIS + RIGHT_IRIS:
        lm = landmarks[idx]

        x = int(lm.x * w)
        y = int(lm.y * h)

        cv2.circle(
            frame,
            (x, y),
            4,
            (0, 0, 255),
            -1,
        )


def iris_center(landmarks, indexes):
    points = []

    for i in indexes:
        points.append(
            [
                landmarks[i].x,
                landmarks[i].y,
            ]
        )

    points = np.array(points)

    return points.mean(axis=0)


cap = cv2.VideoCapture(0)

start = time.time()


while True:
    ret, frame = cap.read()

    if not ret:
        break

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB,
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb,
    )

    timestamp = int((time.time() - start) * 1000)

    result = detector.detect_for_video(
        mp_image,
        timestamp,
    )

    if result.face_landmarks:
        landmarks = result.face_landmarks[0]

        draw_landmarks(
            frame,
            landmarks,
        )

        left = iris_center(
            landmarks,
            LEFT_IRIS,
        )

        right = iris_center(
            landmarks,
            RIGHT_IRIS,
        )

        # Display normalized iris position
        cv2.putText(
            frame,
            f"Left iris: {left[0]:.3f}, {left[1]:.3f}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            frame,
            f"Right iris: {right[0]:.3f}, {right[1]:.3f}",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

    cv2.imshow(
        "MediaPipe Iris Tracking",
        frame,
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
