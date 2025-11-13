#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script for finger counting functionality
"""

import cv2 as cv
import mediapipe as mp
from utils.finger_counter import FingerCounter
import numpy as np


def test_finger_counter():
    """Test the finger counter with synthetic data"""
    print("Testing Finger Counter...")
    
    finger_counter = FingerCounter()
    
    # Create synthetic hand landmark data for testing
    # This represents a hand with all fingers up (number 5)
    def create_test_landmarks(fingers_up):
        """Create synthetic landmark data for testing"""
        landmarks = []
        
        # Wrist (base point)
        wrist = [100, 200]
        landmarks.append(wrist)
        
        # Palm landmarks (simplified)
        for i in range(1, 5):
            landmarks.append([wrist[0] + i * 20, wrist[1] + i * 10])
        
        # Thumb landmarks
        thumb_base = [wrist[0] - 30, wrist[1] - 20]
        thumb_middle = [thumb_base[0] - 20, thumb_base[1] - 10]
        thumb_tip = [thumb_middle[0] - 25, thumb_middle[1] - 15]
        
        landmarks.extend([thumb_base, thumb_middle, thumb_tip])
        
        # Finger landmarks (4 fingers, each with 4 points)
        finger_base_positions = [
            [wrist[0] + 20, wrist[1] - 30],   # Index
            [wrist[0] + 40, wrist[1] - 35],   # Middle  
            [wrist[0] + 60, wrist[1] - 30],   # Ring
            [wrist[0] + 80, wrist[1] - 25]    # Pinky
        ]
        
        for i, (finger_base_x, finger_base_y) in enumerate(finger_base_positions):
            # MCP joint
            landmarks.append([finger_base_x, finger_base_y])
            
            # PIP joint
            pip_y = finger_base_y - 30
            landmarks.append([finger_base_x, pip_y])
            
            # DIP joint
            dip_y = pip_y - 25
            landmarks.append([finger_base_x, dip_y])
            
            # Tip
            if fingers_up[i]:  # Finger is up
                tip_y = dip_y - 35
            else:  # Finger is down
                tip_y = finger_base_y + 20
            landmarks.append([finger_base_x, tip_y])
        
        return landmarks
    
    # Test different finger configurations
    test_cases = [
        (0, [False, False, False, False]),  # Zero fingers
        (1, [True, False, False, False]),   # One finger (thumb)
        (2, [True, True, False, False]),    # Two fingers
        (3, [True, True, True, False]),     # Three fingers
        (4, [True, True, True, True]),      # Four fingers
        (5, [True, True, True, True]),      # Five fingers (all up)
    ]
    
    print("\nSynthetic Test Results:")
    print("-" * 40)
    
    for expected_count, finger_states in test_cases:
        # Create landmarks with thumb up and specified finger states
        landmarks = create_test_landmarks(finger_states)
        
        # Test finger counting
        count = finger_counter.count_fingers(landmarks)
        states = finger_counter.get_finger_states(landmarks)
        
        print(f"Expected: {expected_count}, Got: {count}")
        print(f"Finger states: {states}")
        print("-" * 40)
    
    print("\nTesting with real webcam...")
    print("Show numbers 0-5 with your fingers!")
    print("Press 'q' to quit")
    
    # Test with real webcam
    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
    
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv.flip(frame, 1)
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        
        results = hands.process(rgb_frame)
        
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
                
                finger_count = finger_counter.count_fingers(landmark_list)
                
                # Draw landmarks
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        # Draw finger count
        cv.rectangle(frame, (10, 10), (150, 120), (0, 0, 0), -1)
        cv.rectangle(frame, (10, 10), (150, 120), (0, 255, 0), 2)
        cv.putText(frame, str(finger_count), (40, 90), 
                   cv.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 5)
        cv.putText(frame, "Fingers:", (20, 35), 
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv.imshow('Finger Counter Test', frame)
        
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv.destroyAllWindows()
    hands.close()
    
    print("\nTest completed!")


if __name__ == '__main__':
    test_finger_counter()