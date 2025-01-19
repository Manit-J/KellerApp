import cv2
import mediapipe as mp
from collections import deque

# Initialize Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Initialize a deque to store wrist positions
wrist_positions = deque(maxlen=20)  # Track the last 20 wrist positions

# Gesture recognition function
def recognize_gesture(hand_landmarks, handedness, sequence_state):
    thumb_tip = hand_landmarks.landmark[4]
    index_tip = hand_landmarks.landmark[8]
    middle_tip = hand_landmarks.landmark[12]
    ring_tip = hand_landmarks.landmark[16]
    pinky_tip = hand_landmarks.landmark[20]
    wrist = hand_landmarks.landmark[0]

    # Track wrist positions for wave gesture
    wrist_x = wrist.x
    wrist_positions.append(wrist_x)

    # Check for "How" gesture (both hands curved)
    thumb_curved = thumb_tip.y > wrist.y and thumb_tip.x > index_tip.x
    fingers_curved = all(
        tip.y > wrist.y for tip in [index_tip, middle_tip, ring_tip, pinky_tip]
    )

    # Check for "You" gesture (index finger pointing)
    is_pointing = (
        index_tip.y < middle_tip.y
        and index_tip.y < ring_tip.y
        and index_tip.y < pinky_tip.y
        and abs(index_tip.x - thumb_tip.x) > 0.1
    )
    
    # Check for hand wave gesture
    if len(wrist_positions) >= 5:
        direction_changes = 0
        total_movement = 0

        for i in range(1, len(wrist_positions)):
            movement = abs(wrist_positions[i] - wrist_positions[i - 1])
            total_movement += movement
            if i > 1 and (wrist_positions[i] - wrist_positions[i - 1]) * (
                wrist_positions[i - 1] - wrist_positions[i - 2]
            ) < 0:
                direction_changes += 1

        if direction_changes >= 4 and total_movement >= 0.2:
            wrist_positions.clear()
            if sequence_state is None:  # Only trigger "Hello" if not in a sequence
                return "Hello!"

    # Prioritize detecting "How" and "You" if sequence is active
    if thumb_curved and fingers_curved:
        return "How"
    elif is_pointing:
        return "You"

    return "Unknown"


# Initialize video capture
cap = cv2.VideoCapture(0)

gesture_sequence = deque(maxlen=5)  # Track recent gestures
sequence_state = None  # Track if a sequence is being detected (e.g., "How" -> "You")
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

            # Pass sequence_state to prevent overlaps
            gesture = recognize_gesture(hand_landmarks, handedness, sequence_state)
            if gesture != "Unknown":
                if len(gesture_sequence) == 0 or gesture_sequence[-1] != gesture:
                    gesture_sequence.append(gesture)
                    print(gesture)

                # Handle sequence detection
                if gesture == "How":
                    sequence_state = "How"  # Set state to wait for "You"
                elif sequence_state == "How" and gesture == "You":
                    print("Detected: 'How are you?'")
                    sequence_state = None  # Reset state after full sequence
                    gesture_sequence.clear()
                elif gesture == "Hello!":
                    sequence_state = None  # Ensure "Hello" resets sequence

    cv2.imshow('Hand Gesture Interpreter', frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()
