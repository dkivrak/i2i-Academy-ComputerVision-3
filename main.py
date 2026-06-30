import cv2
import mediapipe as mp


class FingerCounter:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

        # MediaPipe hand landmark indexes
        self.finger_tips = {
            "thumb": 4,
            "index": 8,
            "middle": 12,
            "ring": 16,
            "pinky": 20
        }

        self.finger_pips = {
            "index": 6,
            "middle": 10,
            "ring": 14,
            "pinky": 18
        }

    def count_open_fingers(self, landmarks, handedness_label):
        """
        Counts how many fingers are open using hand landmark positions.

        For index/middle/ring/pinky:
        - Finger is open if fingertip is above the PIP joint.
        - In image coordinates, smaller y means higher position.

        For thumb:
        - Logic depends on whether the detected hand is Left or Right.
        - This works better because thumb opens sideways.
        """

        open_fingers = 0

        # Thumb
        thumb_tip = landmarks[self.finger_tips["thumb"]]
        thumb_ip = landmarks[3]

        if handedness_label == "Right":
            if thumb_tip.x < thumb_ip.x:
                open_fingers += 1
        else:
            if thumb_tip.x > thumb_ip.x:
                open_fingers += 1

        # Other four fingers
        for finger_name, tip_index in self.finger_tips.items():
            if finger_name == "thumb":
                continue

            pip_index = self.finger_pips[finger_name]

            fingertip = landmarks[tip_index]
            pip_joint = landmarks[pip_index]

            if fingertip.y < pip_joint.y:
                open_fingers += 1

        return open_fingers

    def process_frame(self, frame):
        # Flip frame horizontally for mirror-like webcam behavior
        frame = cv2.flip(frame, 1)

        # Convert BGR to RGB because MediaPipe expects RGB images
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.hands.process(rgb_frame)

        finger_count = 0
        status_text = "No hand detected"

        if results.multi_hand_landmarks and results.multi_handedness:
            hand_landmarks = results.multi_hand_landmarks[0]
            handedness = results.multi_handedness[0].classification[0].label

            finger_count = self.count_open_fingers(
                hand_landmarks.landmark,
                handedness
            )

            status_text = f"Fingers: {finger_count}"

            # Draw hand landmarks on the frame
            self.mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_styles.get_default_hand_landmarks_style(),
                self.mp_styles.get_default_hand_connections_style()
            )

        # Display result text
        cv2.rectangle(frame, (20, 20), (300, 90), (0, 0, 0), -1)
        cv2.putText(
            frame,
            status_text,
            (35, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (255, 255, 255),
            3
        )

        return frame

    def close(self):
        self.hands.close()


def main():
    camera = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

    if not camera.isOpened():
        print("Error: Could not open webcam.")
        return

    counter = FingerCounter()

    print("Webcam started. Press 'q' to quit.")

    while True:
        success, frame = camera.read()

        if not success:
            print("Error: Could not read frame from webcam.")
            break

        processed_frame = counter.process_frame(frame)

        cv2.imshow("Computer Vision - Finger Counter", processed_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    counter.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()