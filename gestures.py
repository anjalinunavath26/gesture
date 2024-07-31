import cv2
import mediapipe as mp

# Initialize mediapipe hands and drawing utilities
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# Start video capture
cap = cv2.VideoCapture(0)

while cap.isOpened(): 
    ret, frame = cap.read()
    if not ret: 
        break
    
    # Flip the frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)
    
    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame to detect hand landmarks
    results = hands.process(frame_rgb)
    
    # List to store landmark positions
    landmark_list = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks on the frame
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            for id, lm in enumerate(hand_landmarks.landmark):
                # Get the dimensions of the frame
                h, w, c = frame.shape
                # Convert normalized coordinates to pixel values
                cx, cy = int(lm.x * w), int(lm.y * h)
                landmark_list.append([cx, cy])

    # Detect gesture based on landmark positions
    if len(landmark_list) != 0:
        gesture = None

        # Thumb up gesture
        if landmark_list[4][1] < landmark_list[3][1] and landmark_list[8][1] > landmark_list[6][1]:
            gesture = "Thumb Up Detected"

        # Index finger pointing up
        elif landmark_list[8][1] < landmark_list[6][1] and landmark_list[12][1] > landmark_list[10][1]:
            gesture = "Index Finger Up Detected"

        # Peace sign
        elif landmark_list[8][1] < landmark_list[6][1] and landmark_list[12][1] < landmark_list[10][1] and landmark_list[4][1] > landmark_list[3][1]:
            gesture = "Peace Sign Detected"

        # Fist (all fingers down)
        elif all(landmark_list[finger][1] > landmark_list[finger - 2][1] for finger in [8, 12, 16, 20]):
            gesture = "Fist Detected"

        # Palm open (all fingers up)
        elif all(landmark_list[finger][1] < landmark_list[finger - 2][1] for finger in [8, 12, 16, 20]):
            gesture = "Palm Open Detected"

        # Display gesture on the frame
        if gesture:
            cv2.putText(frame, gesture, (landmark_list[0][0] - 50, landmark_list[0][1] - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3, cv2.LINE_AA)

    # Show the frame
    cv2.imshow('Hand Gesture Recognition', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and destroy all windows
cap.release()
cv2.destroyAllWindows()
