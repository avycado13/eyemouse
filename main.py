import cv2

# Capture video from webcam
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")


while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for x, y, w, h in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y : y + h, x : x + w]
        roi_color = frame[y : y + h, x : x + w]

        # Detect eyes within the face region
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for ex, ey, ew, eh in eyes:
            # cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
            eye_gray = roi_gray[ey : ey + eh, ex : ex + ew]
            eye_color = roi_color[ey : ey + eh, ex : ex + ew]

            blurred = cv2.GaussianBlur(eye_gray, (5, 5), 0)

            thresh = cv2.adaptiveThreshold(
                blurred,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                11,
                2,
            )
            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

            if contours:
                (cx, cy), radius = cv2.minEnclosingCircle(contours[0])
                cv2.circle(eye_color, (int(cx), int(cy)), int(radius), (255, 0, 0), 2)
                cv2.line(frame, (int(cx), int(cy)), (int(cx) + 50, int(cy)), (0, 255, 0), 2)
                cv2.line(frame, (int(cx), int(cy)), (int(cx), int(cy) + 50), (0, 255, 0), 2)
    cv2.imshow("Eye Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
