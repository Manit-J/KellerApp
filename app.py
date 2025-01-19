import cv2
import mediapipe as mp
from collections import deque

# Initialize Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Initialize a deque to store wrist positions
wrist_positions = deque(maxlen=20)  # Track the last 20 wrist positions
sequence_in_progress = False 


# Updated recognize_gesture function
def recognize_gesture(hand_landmarks, handedness):
    """
    Recognizes hand gestures including "Hello!" and "How are you?".
    Avoids detecting gestures on random movements.
    """
    thumb_tip = hand_landmarks.landmark[4]
    index_tip = hand_landmarks.landmark[8]
    middle_tip = hand_landmarks.landmark[12]
    ring_tip = hand_landmarks.landmark[16]
    pinky_tip = hand_landmarks.landmark[20]
    wrist = hand_landmarks.landmark[0]

    # Track wrist movement for "Hello!" gesture (wave)
    wrist_x = wrist.x
    wrist_positions.append(wrist_x)

    if len(wrist_positions) >= 5:  # Ensure we have enough positions to analyze
        direction_changes = 0
        total_movement = 0

        for i in range(1, len(wrist_positions)):
            movement = abs(wrist_positions[i] - wrist_positions[i - 1])
            total_movement += movement
            if (wrist_positions[i] - wrist_positions[i - 1]) * (
                wrist_positions[i - 1] - wrist_positions[i - 2]
            ) < 0:
                direction_changes += 1

        # "Hello!" gesture: sufficient movement and direction changes
        if direction_changes >= 10 and total_movement >= 0.2:
            wrist_positions.clear()
            return "Hello!"

    # "How" gesture detection (both hands curved)
    thumb_curved = thumb_tip.y > wrist.y and thumb_tip.x > index_tip.x
    fingers_curved = all(
        tip.y > wrist.y for tip in [index_tip, middle_tip, ring_tip, pinky_tip]
    )
    if thumb_curved and fingers_curved:
        return "How"

    # "You" gesture detection (index finger pointing outward)
    is_pointing = (
        index_tip.y < middle_tip.y
        and index_tip.y < ring_tip.y
        and index_tip.y < pinky_tip.y
        and abs(index_tip.x - thumb_tip.x) > 0.1
    )
    if is_pointing:
        return "You"

    # Default return if no specific gesture is detected
    return "Unknown"




# Initialize video capture
cap = cv2.VideoCapture(0)

gesture_sequence = deque(maxlen=5)  # Track recent gestures
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            handedness = results.multi_handedness[idx].classification[0].label
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            gesture = recognize_gesture(hand_landmarks, handedness)
            if gesture != "Unknown":
                gesture_sequence.append(gesture)
                print(gesture)

            # Detect "How are you?" sequence
            if list(gesture_sequence)[-2:] == ["How", "You"]:
                print("Detected: 'How are you?'")
                gesture_sequence.clear()

    cv2.imshow('Hand Gesture Interpreter', frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()
