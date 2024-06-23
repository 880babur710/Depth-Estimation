import cv2 as cv
import mediapipe as mp
import numpy as np


mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
# drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

LEFT_EYE_LANDMARKS = [33, 246, 7, 161, 163, 160, 144, 145, 159, 158, 157, 153,
                      154, 155, 173, 133]
RIGHT_EYE_LANDMARKS = [362, 398, 382, 384, 381, 385, 380, 386, 374, 387, 373,
                       388, 390, 466, 249]
LEFT_EYE_PUPIL = 473
RIGHT_EYE_PUPIL = 468

AVERAGE_PUPILLARY_DISTANCE = 6.3


def main():
    webcam = cv.VideoCapture(0)
    if not webcam.isOpened():
        print("Error: Could not open webcam.")
        return

    try:
        success, frame = webcam.read()
        frame_height, frame_width, _ = frame.shape
        focal_length = 1 * frame_width
        while success:
            # Flip the frame horizontally for a mirror effect
            frame = cv.flip(frame, 1)
            # Convert the BGR image to RGB
            rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            # Process the frame with MediaPipe FaceMesh
            results = face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Get the eye landmarks
                    left_eye = face_landmarks.landmark[LEFT_EYE_PUPIL]
                    right_eye = face_landmarks.landmark[RIGHT_EYE_PUPIL]

                    # Calculate the distance between the eyes in pixels
                    left_eye_x = left_eye.x * frame_width
                    left_eye_y = left_eye.y * frame_height
                    right_eye_x = right_eye.x * frame_width
                    right_eye_y = right_eye.y * frame_height
                    image_eye_distance = ((right_eye_x - left_eye_x) ** 2 + (right_eye_y - left_eye_y) ** 2) ** 0.5

                    head_webcam_distance = (focal_length / image_eye_distance) * AVERAGE_PUPILLARY_DISTANCE
                    text = f"Distance: {np.round(head_webcam_distance, 2)}"
                    cv.putText(frame, text, (20, 50), cv.FONT_HERSHEY_SIMPLEX,
                               2, (255, 191, 0), 2)

            cv.imshow('Webcam', frame)
            key = cv.waitKey(5) & 0xFF
            if key == ord('q') or key == ord('Q'):
                break

            success, frame = webcam.read()
    finally:
        webcam.release()
        cv.destroyAllWindows()


if __name__ == "__main__":
    main()
