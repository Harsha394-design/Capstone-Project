import cv2
import mediapipe as mp
import numpy as np
import config

X_AXIS_CHEAT = []
Y_AXIS_CHEAT = []
PERSON_COUNT = 0

mp_face_mesh = mp.solutions.face_mesh
# The fix: Setting max_num_faces to a higher value than the default of 1
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

def process_frame(image):
    global X_AXIS_CHEAT, Y_AXIS_CHEAT, PERSON_COUNT
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = face_mesh.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    img_h, img_w, _ = image.shape
    face_ids = [33, 263, 1, 61, 291, 199]

    X_AXIS_CHEAT = []
    Y_AXIS_CHEAT = []
    PERSON_COUNT = 0

    if results.multi_face_landmarks:
        PERSON_COUNT = len(results.multi_face_landmarks)
        
        for i, face_landmarks in enumerate(results.multi_face_landmarks):
            face_2d = []
            face_3d = []

            for idx, lm in enumerate(face_landmarks.landmark):
                if idx in face_ids:
                    x, y = int(lm.x * img_w), int(lm.y * img_h)
                    face_2d.append([x, y])
                    face_3d.append([x, y, lm.z])

            face_2d = np.array(face_2d, dtype=np.float64)
            face_3d = np.array(face_3d, dtype=np.float64)

            focal_length = 1 * img_w
            cam_matrix = np.array([[focal_length, 0, img_w / 2],
                                   [0, focal_length, img_h / 2],
                                   [0, 0, 1]])
            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
            rmat, jac = cv2.Rodrigues(rot_vec)
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
            x_angle = angles[0] * 360
            y_angle = angles[1] * 360

            if y_angle < -10 or y_angle > 10:
                X_AXIS_CHEAT.append(1)
            else:
                X_AXIS_CHEAT.append(0)

            if x_angle < -5:
                Y_AXIS_CHEAT.append(1)
            else:
                Y_AXIS_CHEAT.append(0)

            x_pos = int(face_landmarks.landmark[0].x * img_w)
            y_pos = int(face_landmarks.landmark[0].y * img_h)

            text = f"Person {i+1}: "
            if y_angle < -10: text += "Looking Left"
            elif y_angle > 10: text += "Looking Right"
            elif x_angle < -10: text += "Looking Down"
            else: text += "Forward"

            cv2.putText(image, text, (x_pos, y_pos - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    cv2.putText(image, f"Faces detected: {PERSON_COUNT}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    return image

if __name__ == "__main__":
    import cv2
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = process_frame(frame)
        cv2.imshow("Head Pose Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
