#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Finger Counting Training Data Collection
Collect training data for finger counting gestures (0-5 fingers).
"""

import csv
import cv2 as cv
import mediapipe as mp
from utils.finger_counter import FingerCounter
import os


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
    
    # Create data directory if it doesn't exist
    data_dir = 'model/finger_count_classifier'
    os.makedirs(data_dir, exist_ok=True)
    
    # CSV file for storing training data
    csv_path = os.path.join(data_dir, 'finger_count_training_data.csv')
    
    # Check if file exists, if not create with header
    if not os.path.exists(csv_path):
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            header = ['finger_count'] + [f'landmark_{i}_{coord}' for i in range(21) for coord in ['x', 'y']]
            writer.writerow(header)
    
    # Labels for finger counts
    finger_count_labels = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five']
    
    # Current collection settings
    current_finger_count = 0
    collection_mode = False
    samples_collected = 0
    
    print("Finger Counting Training Data Collection")
    print("Instructions:")
    print("- Press number keys 0-5 to select finger count to collect")
    print("- Press SPACE to start/stop collection mode")
    print("- Press 'q' to quit")
    print("- Press 's' to save current sample manually")
    print("\nCurrent status will be shown on screen")
    
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
        
        # Initialize variables
        finger_count = 0
        landmark_list = []
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Get landmark points
                h, w = frame.shape[:2]
                landmark_points = []
                
                for landmark in hand_landmarks.landmark:
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    landmark_points.append([x, y])
                
                landmark_list = landmark_points
                finger_count = finger_counter.count_fingers(landmark_points)
                
                # Draw hand landmarks
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        # Create status display
        h, w = frame.shape[:2]
        
        # Draw background rectangles
        cv.rectangle(frame, (10, 10), (400, 150), (0, 0, 0), -1)
        cv.rectangle(frame, (10, 10), (400, 150), (0, 255, 0) if collection_mode else (0, 0, 255), 2)
        
        # Status text
        status_color = (0, 255, 0) if collection_mode else (0, 165, 255)
        status_text = "COLLECTING" if collection_mode else "IDLE"
        
        cv.putText(frame, f"Status: {status_text}", (20, 35), 
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        cv.putText(frame, f"Target: {current_finger_count} fingers ({finger_count_labels[current_finger_count]})", 
                   (20, 60), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv.putText(frame, f"Detected: {finger_count} fingers", 
                   (20, 85), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv.putText(frame, f"Samples collected: {samples_collected}", 
                   (20, 110), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv.putText(frame, "Press SPACE: Start/Stop | 0-5: Select count | q: Quit | s: Save", 
                   (20, 135), cv.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Auto-save if in collection mode and finger count matches target
        if collection_mode and finger_count == current_finger_count and len(landmark_list) == 21:
            # Flatten landmark data
            landmark_data = []
            for point in landmark_list:
                landmark_data.extend(point)
            
            # Save to CSV
            with open(csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([current_finger_count] + landmark_data)
            
            samples_collected += 1
            print(f"Sample saved: {current_finger_count} fingers (Total: {samples_collected})")
        
        # Show the frame
        cv.imshow('Finger Counting Data Collection', frame)
        
        # Handle keyboard input
        key = cv.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord(' '):  # Spacebar
            collection_mode = not collection_mode
            print(f"Collection mode: {'ON' if collection_mode else 'OFF'}")
        elif key >= ord('0') and key <= ord('5'):
            current_finger_count = key - ord('0')
            print(f"Selected finger count: {current_finger_count}")
        elif key == ord('s') and len(landmark_list) == 21:  # Manual save
            # Flatten landmark data
            landmark_data = []
            for point in landmark_list:
                landmark_data.extend(point)
            
            # Save to CSV
            with open(csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([current_finger_count] + landmark_data)
            
            samples_collected += 1
            print(f"Manual sample saved: {current_finger_count} fingers (Total: {samples_collected})")
    
    # Cleanup
    cap.release()
    cv.destroyAllWindows()
    hands.close()
    
    print(f"\nData collection complete!")
    print(f"Total samples collected: {samples_collected}")
    print(f"Data saved to: {csv_path}")


if __name__ == '__main__':
    main()