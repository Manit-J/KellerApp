import cv2
import mediapipe as mp
from collections import deque

# Initialize Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Gesture recognition function
def recognize_gesture(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[4]
    index_tip = hand_landmarks.landmark[8]
    middle_tip = hand_landmarks.landmark[12]
    
    # Example gestures
    if thumb_tip.y < index_tip.y and thumb_tip.y < middle_tip.y:
        return "Thumbs Up"
    elif all(landmark.y > hand_landmarks.landmark[0].y for landmark in hand_landmarks.landmark[1:]):
        return "Fist"
    elif index_tip.y < thumb_tip.y and index_tip.y < middle_tip.y:
        return "Pointing"
    else:
        return "Unknown"

# Detect gesture sequence
def check_gesture_sequence(gesture_sequence, target_sequence):
    """Returns True if the target sequence appears at the end of gesture_sequence."""
    if len(gesture_sequence) >= len(target_sequence):
        return list(gesture_sequence)[-len(target_sequence):] == target_sequence
    return False

# Initialize video capture
cap = cv2.VideoCapture(0)

# Keep track of recent gestures
gesture_sequence = deque(maxlen=10)  # Stores the last 10 gestures
target_sequence = ["Thumbs Up", "Fist", "Pointing"]  # Target gesture sequence

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to RGB for Mediapipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # Draw hand landmarks and recognize gestures
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Recognize gesture
            gesture = recognize_gesture(hand_landmarks)
            if gesture != "Unknown":  # Avoid adding "Unknown" gestures to the sequence
                gesture_sequence.append(gesture)

            # Check if the target sequence appears
            if check_gesture_sequence(gesture_sequence, target_sequence):
                print("Sequence Detected! - Thumbs Up -> Fist -> Pointing")
                gesture_sequence.clear()  # Clear sequence to avoid multiple detections

    # Display the frame
    cv2.imshow('Hand Gesture Interpreter', frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()
