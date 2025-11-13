import numpy as np
import math

class FingerCounter:
    """
    Finger counting utility that determines how many fingers are raised
    based on hand landmarks from MediaPipe.
    """
    
    def __init__(self):
        # MediaPipe hand landmark indices
        self.THUMB_TIP = 4
        self.THUMB_IP = 3
        self.INDEX_FINGER_TIP = 8
        self.INDEX_FINGER_PIP = 6
        self.MIDDLE_FINGER_TIP = 12
        self.MIDDLE_FINGER_PIP = 10
        self.RING_FINGER_TIP = 16
        self.RING_FINGER_PIP = 14
        self.PINKY_TIP = 20
        self.PINKY_PIP = 18
        self.WRIST = 0
        self.MIDDLE_FINGER_MCP = 9
        
    def count_fingers(self, landmark_list):
        """
        Count how many fingers are raised based on hand landmarks.
        
        Args:
            landmark_list: List of [x, y] coordinates for each hand landmark
            
        Returns:
            int: Number of fingers raised (0-5)
        """
        if len(landmark_list) != 21:  # MediaPipe provides 21 landmarks
            return 0
            
        fingers_up = []
        
        # Check thumb (special case - horizontal movement)
        thumb_up = self._is_thumb_up(landmark_list)
        fingers_up.append(thumb_up)
        
        # Check other fingers (vertical movement)
        fingers_up.append(self._is_finger_up(landmark_list, self.INDEX_FINGER_TIP, self.INDEX_FINGER_PIP))
        fingers_up.append(self._is_finger_up(landmark_list, self.MIDDLE_FINGER_TIP, self.MIDDLE_FINGER_PIP))
        fingers_up.append(self._is_finger_up(landmark_list, self.RING_FINGER_TIP, self.RING_FINGER_PIP))
        fingers_up.append(self._is_finger_up(landmark_list, self.PINKY_TIP, self.PINKY_PIP))
        
        return sum(fingers_up)
    
    def _is_thumb_up(self, landmark_list):
        """Check if thumb is up (extended)"""
        thumb_tip = landmark_list[self.THUMB_TIP]
        thumb_ip = landmark_list[self.THUMB_IP]
        middle_finger_mcp = landmark_list[self.MIDDLE_FINGER_MCP]
        
        # Calculate horizontal distance from thumb tip to middle finger MCP
        horizontal_distance = abs(thumb_tip[0] - middle_finger_mcp[0])
        
        # Calculate vertical distance from thumb tip to thumb IP joint
        vertical_distance = abs(thumb_tip[1] - thumb_ip[1])
        
        # Thumb is considered up if it's extended horizontally beyond a threshold
        # and the tip is above or at the same level as the IP joint
        return horizontal_distance > 30 and thumb_tip[1] <= thumb_ip[1] + 10
    
    def _is_finger_up(self, landmark_list, tip_index, pip_index):
        """Check if a finger is up (extended)"""
        tip = landmark_list[tip_index]
        pip = landmark_list[pip_index]
        wrist = landmark_list[self.WRIST]
        
        # Finger is up if the tip is above the PIP joint
        # and significantly above the wrist level
        return tip[1] < pip[1] and tip[1] < wrist[1] - 20
    
    def get_finger_states(self, landmark_list):
        """
        Get individual finger states for more detailed analysis.
        
        Returns:
            dict: Individual finger states (True = up, False = down)
        """
        if len(landmark_list) != 21:
            return {'thumb': False, 'index': False, 'middle': False, 'ring': False, 'pinky': False}
            
        return {
            'thumb': self._is_thumb_up(landmark_list),
            'index': self._is_finger_up(landmark_list, self.INDEX_FINGER_TIP, self.INDEX_FINGER_PIP),
            'middle': self._is_finger_up(landmark_list, self.MIDDLE_FINGER_TIP, self.MIDDLE_FINGER_PIP),
            'ring': self._is_finger_up(landmark_list, self.RING_FINGER_TIP, self.RING_FINGER_PIP),
            'pinky': self._is_finger_up(landmark_list, self.PINKY_TIP, self.PINKY_PIP)
        }
    
    def get_number_gesture(self, landmark_list):
        """
        Get the number being shown (0-5) based on finger count.
        
        Returns:
            int: Number of fingers raised (0-5)
        """
        return self.count_fingers(landmark_list)