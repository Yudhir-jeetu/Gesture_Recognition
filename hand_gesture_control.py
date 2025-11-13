#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Hand Gesture Control System
Controls computer functions using hand gestures with pyautogui integration.
Right Hand: Brightness, Screenshot, Window Management
Left Hand: Volume, Media Controls, Skip/Play/Pause
"""

import cv2 as cv
import mediapipe as mp
import pyautogui
import time
from utils.finger_counter import FingerCounter
import numpy as np


class GestureController:
    def __init__(self):
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        )
        
        # Initialize finger counter
        self.finger_counter = FingerCounter()
        
        # Gesture tracking
        self.right_hand_state = {'fingers': 0, 'gesture': None, 'last_gesture': None}
        self.left_hand_state = {'fingers': 0, 'gesture': None, 'last_gesture': None}
        
        # Control states - simplified for hotkeys
        
        # Timing control
        self.last_action_time = {'right': 0, 'left': 0}
        self.action_cooldown = 1.0  # 1 second between actions
        
        # Screenshot counter
        self.screenshot_count = 0
        
        # Screen dimensions
        self.screen_width, self.screen_height = pyautogui.size()
        
        print("Hand Gesture Hotkey Control System Initialized")
        print("=" * 50)
        print("Right Hand Hotkeys:")
        print("  1 finger → Brightness Up (Fn+F3)")
        print("  2 fingers → Brightness Down (Fn+F2)")
        print("  3 fingers → Screenshot (Win+Shift+S)")
        print("  4 fingers → Move window right (Win+Right)")
        print("  5 fingers → Move window left (Win+Left)")
        print()
        print("Left Hand Hotkeys:")
        print("  1 finger → Volume Up (Volume Up key)")
        print("  2 fingers → Volume Down (Volume Down key)")
        print("  3 fingers → Mute/Unmute (Volume Mute key)")
        print("  4 fingers → Skip/Next (Next Track key)")
        print("  Open hand (5 fingers) → Play/Pause (Play/Pause key)")
        print("=" * 50)
        print("Press 'q' to quit, 'h' for help")
    
    def initialize_audio(self):
        """Initialize audio - not needed since we use hotkeys"""
        pass
    
    def volume_up(self):
        """Increase volume using media key"""
        try:
            pyautogui.press('volumeup')
            print("Volume increased")
        except Exception as e:
            print(f"Volume up failed: {e}")
    
    def volume_down(self):
        """Decrease volume using media key"""
        try:
            pyautogui.press('volumedown')
            print("Volume decreased")
        except Exception as e:
            print(f"Volume down failed: {e}")
    
    def toggle_mute(self):
        """Toggle mute using media key"""
        try:
            pyautogui.press('volumemute')
            print("Mute toggled")
        except Exception as e:
            print(f"Mute toggle failed: {e}")
    
    def brightness_up(self):
        """Increase screen brightness using hotkeys"""
        try:
            # Try common brightness up hotkeys
            pyautogui.hotkey('fn', 'f3')  # Common on laptops
            time.sleep(0.1)
            pyautogui.press('brightnessup')  # Media key
            print("Brightness increased")
        except Exception as e:
            print(f"Brightness up failed: {e}")
    
    def brightness_down(self):
        """Decrease screen brightness using hotkeys"""
        try:
            # Try common brightness down hotkeys
            pyautogui.hotkey('fn', 'f2')  # Common on laptops
            time.sleep(0.1)
            pyautogui.press('brightnessdown')  # Media key
            print("Brightness decreased")
        except Exception as e:
            print(f"Brightness down failed: {e}")
    
    def take_screenshot(self):
        """Take screenshot using Windows hotkey"""
        try:
            # Use Windows + Shift + S for screenshot tool
            pyautogui.hotkey('win', 'shift', 's')
            print("Screenshot tool activated (Win+Shift+S)")
        except Exception as e:
            print(f"Screenshot failed: {e}")
    
    def move_window_right(self):
        """Move window right using Win+Right hotkey"""
        try:
            pyautogui.hotkey('win', 'right')
            print("Window moved right (Win+Right)")
        except Exception as e:
            print(f"Window move right failed: {e}")
    
    def move_window_left(self):
        """Move window left using Win+Left hotkey"""
        try:
            pyautogui.hotkey('win', 'left')
            print("Window moved left (Win+Left)")
        except Exception as e:
            print(f"Window move left failed: {e}")
    
    def media_play_pause(self):
        """Play/Pause media using playpause media key"""
        try:
            pyautogui.press('playpause')
            print("Play/Pause")
        except Exception as e:
            print(f"Play/Pause failed: {e}")
    
    def media_next(self):
        """Next track/skip using nexttrack media key"""
        try:
            pyautogui.press('nexttrack')
            print("Next/Skip")
        except Exception as e:
            print(f"Next/Skip failed: {e}")
    
    def detect_rotation(self, point_history):
        """Rotation detection not needed - using finger counts instead"""
        return None
    
    def can_perform_action(self, hand):
        """Check if enough time has passed since last action"""
        current_time = time.time()
        if current_time - self.last_action_time[hand] >= self.action_cooldown:
            self.last_action_time[hand] = current_time
            return True
        return False
    
    def start_countdown(self, target_action, hand):
        """Countdown not needed for hotkey version"""
        pass
    
    def countdown_worker(self):
        """Countdown not needed for hotkey version"""
        pass
    
    def process_frame(self, frame):
        """Process a single frame and detect gestures"""
        # Convert BGR to RGB
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.hands.process(rgb_frame)
        
        # Initialize variables
        right_hand_detected = False
        left_hand_detected = False
        
        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # Get hand side (Left/Right)
                hand_side = handedness.classification[0].label
                
                # Get landmark points
                h, w = frame.shape[:2]
                landmark_list = []
                for landmark in hand_landmarks.landmark:
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    landmark_list.append([x, y])
                
                # Count fingers
                finger_count = self.finger_counter.count_fingers(landmark_list)
                
                # Process based on hand side
                if hand_side == "Right":
                    right_hand_detected = True
                    self.process_right_hand(finger_count, landmark_list)
                    
                    # Draw right hand landmarks in blue
                    self.mp_hands.solutions.drawing_utils.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                        landmark_drawing_spec=self.mp_hands.solutions.drawing_styles
                        .get_default_hand_landmarks_style())
                    
                elif hand_side == "Left":
                    left_hand_detected = True
                    self.process_left_hand(finger_count)
                    
                    # Draw left hand landmarks in green
                    self.mp_hands.solutions.drawing_utils.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                        landmark_drawing_spec=self.mp_hands.solutions.drawing_styles
                        .get_default_hand_landmarks_style())
        
        # Reset states if hands not detected
        if not right_hand_detected:
            self.right_hand_state = {'fingers': 0, 'gesture': None, 'last_gesture': None}
        if not left_hand_detected:
            self.left_hand_state = {'fingers': 0, 'gesture': None, 'last_gesture': None}
        
        return frame
    
    def process_right_hand(self, finger_count, landmark_list):
        """Process right hand gestures with immediate hotkey actions"""
        self.right_hand_state['fingers'] = finger_count
        
        # Immediate hotkey actions for right hand
        if self.right_hand_state['last_gesture'] != finger_count and self.can_perform_action("right"):
            if finger_count == 1:
                self.brightness_up()
            elif finger_count == 2:
                self.brightness_down()
            elif finger_count == 3:
                self.take_screenshot()
            elif finger_count == 4:
                self.move_window_right()
            elif finger_count == 5:
                self.move_window_left()
        
        self.right_hand_state['last_gesture'] = finger_count
    
    def process_left_hand(self, finger_count):
        """Process left hand gestures"""
        self.left_hand_state['fingers'] = finger_count
        
        # Immediate actions (no countdown needed)
        if self.left_hand_state['last_gesture'] != finger_count and self.can_perform_action("left"):
            if finger_count == 1:
                self.volume_up()
            elif finger_count == 2:
                self.volume_down()
            elif finger_count == 3:
                self.toggle_mute()
            elif finger_count == 4:
                self.media_next()
            elif finger_count == 5:
                self.media_play_pause()
        
        self.left_hand_state['last_gesture'] = finger_count
    
    def draw_ui(self, frame):
        """Draw simplified UI overlay on frame"""
        h, w = frame.shape[:2]
        
        # Background for status display
        cv.rectangle(frame, (10, 10), (300, 120), (0, 0, 0), -1)
        cv.rectangle(frame, (10, 10), (300, 120), (255, 255, 255), 2)
        
        # Title
        cv.putText(frame, "Gesture Hotkey Control", (20, 35), 
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Right hand status
        cv.putText(frame, f"Right: {self.right_hand_state['fingers']} fingers", 
                   (20, 60), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Left hand status
        cv.putText(frame, f"Left: {self.left_hand_state['fingers']} fingers", 
                   (20, 85), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Instructions
        cv.putText(frame, "Press 'q' to quit", (20, 110), 
                   cv.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        return frame
    
    def run(self):
        """Main run loop"""
        cap = cv.VideoCapture(0)
        cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
        
        print("Starting gesture control...")
        print("Make sure to keep your hands in view of the camera")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv.flip(frame, 1)
            
            # Process frame for gesture detection
            frame = self.process_frame(frame)
            
            # Draw UI
            frame = self.draw_ui(frame)
            
            # Show frame
            cv.imshow('Hand Gesture Control System', frame)
            
            # Handle keyboard input
            key = cv.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('h'):
                print("\nHand Gesture Hotkey Controls:")
                print("Right Hand:")
                print("  1 finger → Brightness Up (Fn+F3)")
                print("  2 fingers → Brightness Down (Fn+F2)")
                print("  3 fingers → Screenshot (Win+Shift+S)")
                print("  4 fingers → Move window right (Win+Right)")
                print("  5 fingers → Move window left (Win+Left)")
                print("\nLeft Hand:")
                print("  1 finger → Volume Up")
                print("  2 fingers → Volume Down")
                print("  3 fingers → Mute/Unmute")
                print("  4 fingers → Next/Skip")
                print("  5 fingers → Play/Pause")
        
        # Cleanup
        cap.release()
        cv.destroyAllWindows()
        self.hands.close()
        print("Gesture control stopped.")


if __name__ == '__main__':
    controller = GestureController()
    controller.run()