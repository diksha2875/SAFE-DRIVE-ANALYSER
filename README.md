# SAFE-DRIVE-ANALYSER
import numpy as np
import cv2
from imutils import face_utils
import dlib
from scipy.spatial import distance

cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

def distance_evaluation(point_a, point_b):
    euclidean_dist = np.linalg.norm(point_a - point_b)
    return euclidean_dist

def blinked(a, b, c, d, e, f):
    # Longitudinal distance evaluation
    d_l = distance_evaluation(d, b) + distance_evaluation(o, e)
    
    # Lateral distance evaluation
    d_a = distance_evaluation(a, f)

    # Ratio Longitudinal / (Lateral)
    ratio = d_l / d_a

    # Checking if it is blinked
    if ratio > 0.25:
        return 2
    elif 0.21 < ratio <= 0.25:
        return 1
    else:
        return 0

while True:
    _, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
        face_frame = frame[y1:y2, x1:x2]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        landmarks = predictor(gray, face)
        landmarks = face_utils.shape_to_np(landmarks)

        left_blink = blinked(landmarks[36], landmarks[37], landmarks[38], landmarks[41], landmarks[40], landmarks[39])
        right_blink = blinked(landmarks[42], landmarks[43], landmarks[44], landmarks[47], landmarks[46], landmarks[45])

        print("Left Eye Blink: ", left_blink)
        print("Right Eye Blink: ", right_blink)

cv2.destroyAllWindows()
cap.release()
