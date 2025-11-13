#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simple Finger Counting Demo
A standalone application that focuses purely on finger counting.
"""

import cv2 as cv
import mediapipe as mp
from utils.finger_counter import FingerCounter


def main():
    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )
    
    # Initialize finger counter
    finger_counter = FingerCounter()
    
    # Initialize webcam
    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("Finger Counting Demo")
    print("Press 'q' to quit")
    print("Show numbers 0-5 with your fingers!")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Flip frame horizontally for mirror effect
        frame = cv.flip(frame, 1)
        
        # Convert BGR to RGB
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        
        # Process the frame
        results = hands.process(rgb_frame)
        
        # Draw finger count
        finger_count = 0
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Get landmark points
                landmark_list = []
                h, w = frame.shape[:2]
                
                for landmark in hand_landmarks.landmark:
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    landmark_list.append([x, y])
                
                # Count fingers
                finger_count = finger_counter.count_fingers(landmark_list)
                
                # Draw hand landmarks
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        # Create finger count display
        h, w = frame.shape[:2]
        
        # Draw background rectangle for finger count
        cv.rectangle(frame, (10, 10), (150, 120), (0, 0, 0), -1)
        cv.rectangle(frame, (10, 10), (150, 120), (0, 255, 0), 2)
        
        # Draw finger count number
        cv.putText(frame, str(finger_count), (40, 90), 
                   cv.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 5)
        
        # Draw label
        cv.putText(frame, "Fingers:", (20, 35), 
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw instructions
        cv.putText(frame, "Show numbers 0-5!", (w - 250, h - 20), 
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Show the frame
        cv.imshow('Finger Counting Demo', frame)
        
        # Check for quit
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv.destroyAllWindows()
    hands.close()


if __name__ == '__main__':
    main()