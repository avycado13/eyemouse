import statistics
import time
from collections import deque
import cv2
import mediapipe as mp
import numpy as np

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pyautogui

BaseOptions = python.BaseOptions
FaceLandmarker = vision.FaceLandmarker
FaceLandmarkerOptions = vision.FaceLandmarkerOptions
RunningMode = vision.RunningMode

WINDOW = 8
position_history = deque(maxlen=WINDOW)

SCREEN_W, SCREEN_H = pyautogui.size()

CALIBRATION_POINTS = [
    (0.1, 0.1),
    (0.5, 0.1),
    (0.9, 0.1),
    (0.1, 0.5),
    (0.5, 0.5),
    (0.9, 0.5),
    (0.1, 0.9),
    (0.5, 0.9),
    (0.9, 0.9),
]

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


def get_gaze(frame, timestamp):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect_for_video(mp_image, timestamp)

    if not result.face_landmarks:
        return None

    landmarks = result.face_landmarks[0]
    left = iris_center(landmarks, LEFT_IRIS)
    right = iris_center(landmarks, RIGHT_IRIS)

    avg_x = statistics.mean([left[0], right[0]])
    avg_y = statistics.mean([left[1], right[1]])

    return avg_x, avg_y, landmarks


def calibrate(cap, start):
    cv2.namedWindow("Calibration", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(
        "Calibration", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
    )

    gaze_samples = []
    screen_samples = []

    for fx, fy in CALIBRATION_POINTS:
        target_x = int(fx * SCREEN_W)
        target_y = int(fy * SCREEN_H)

        point_gaze = []
        while len(point_gaze) < 30:
            ret, frame = cap.read()
            if not ret:
                continue

            canvas = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)
            cv2.circle(canvas, (target_x, target_y), 20, (0, 0, 255), -1)
            cv2.putText(
                canvas,
                "Look at the dot",
                (SCREEN_W // 2 - 150, SCREEN_H - 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )
            cv2.imshow("Calibration", canvas)

            timestamp = int((time.time() - start) * 1000)
            gaze = get_gaze(frame, timestamp)
            if gaze is not None:
                point_gaze.append(gaze[:2])

            if cv2.waitKey(1) & 0xFF == ord("q"):
                cv2.destroyWindow("Calibration")
                return None

        gaze_samples.append(np.mean(point_gaze, axis=0))
        screen_samples.append((target_x, target_y))

    cv2.destroyWindow("Calibration")

    gaze_samples = np.array(gaze_samples)
    screen_samples = np.array(screen_samples)

    design = np.column_stack(
        [gaze_samples[:, 0], gaze_samples[:, 1], np.ones(len(gaze_samples))]
    )

    coeffs_x, _, _, _ = np.linalg.lstsq(design, screen_samples[:, 0], rcond=None)
    coeffs_y, _, _, _ = np.linalg.lstsq(design, screen_samples[:, 1], rcond=None)

    return coeffs_x, coeffs_y


def gaze_to_screen(coeffs_x, coeffs_y, gaze_x, gaze_y):
    screen_x = coeffs_x[0] * gaze_x + coeffs_x[1] * gaze_y + coeffs_x[2]
    screen_y = coeffs_y[0] * gaze_x + coeffs_y[1] * gaze_y + coeffs_y[2]

    margin = 1
    screen_x = min(max(screen_x, margin), SCREEN_W - 1 - margin)
    screen_y = min(max(screen_y, margin), SCREEN_H - 1 - margin)

    return screen_x, screen_y


cap = cv2.VideoCapture(0)

start = time.time()

calibration = calibrate(cap, start)
if calibration is None:
    cap.release()
    cv2.destroyAllWindows()
    raise SystemExit("Calibration cancelled")

coeffs_x, coeffs_y = calibration

while True:
    ret, frame = cap.read()

    if not ret:
        break

    timestamp = int((time.time() - start) * 1000)

    gaze = get_gaze(frame, timestamp)
    if gaze is not None:
        avg_x, avg_y, landmarks = gaze

        draw_landmarks(
            frame,
            landmarks,
        )

        position_history.append((avg_x, avg_y))

        smoothed_x = statistics.mean(p[0] for p in position_history)
        smoothed_y = statistics.mean(p[1] for p in position_history)

        screen_x, screen_y = gaze_to_screen(
            coeffs_x, coeffs_y, smoothed_x, smoothed_y
        )

        try:
            pyautogui.moveTo(screen_x, screen_y)
        except pyautogui.FailSafeException:
            print("fail-safe triggered, skipping move")
        print(f"mouse pos: {pyautogui.position()}")

    cv2.imshow("frame", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
