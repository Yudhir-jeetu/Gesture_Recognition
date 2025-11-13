#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Enhanced Hand Gesture Recognition with Finger Counting + Actions
Added gesture-to-action mapping:
- Right hand: 1/2/3 fingers -> 3s countdown then action (brightness high/low/screenshot)
  clockwise/counterclockwise -> snap window right/left
- Left hand: immediate actions (volume up/down/mute/next/play-pause)
Uses pyautogui for hotkeys; tries screen_brightness_control for brightness.
"""

import csv
import copy
import argparse
import itertools
from collections import Counter, deque
import time
import os
from datetime import datetime

import cv2 as cv
import numpy as np
import mediapipe as mp

# new imports
import pyautogui
try:
    import screen_brightness_control as sbc
except Exception:
    sbc = None  # fallback if not installed

from utils import CvFpsCalc
from utils.finger_counter import FingerCounter
from model import KeyPointClassifier
from model import PointHistoryClassifier

# create screenshot folder
os.makedirs('screenshots', exist_ok=True)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)

    parser.add_argument('--show_finger_count', action='store_true', default=True,
                        help='Display finger count on screen')
    parser.add_argument('--show_finger_states', action='store_true', default=False,
                        help='Display individual finger states')

    args = parser.parse_args()

    return args


# ------------------- Action helpers -------------------

def set_brightness_high():
    """Set brightness high (try screen_brightness_control, fallback to repeated key presses)."""
    try:
        if sbc:
            sbc.set_brightness(100)
            return True
    except Exception:
        pass
    # Fallback: attempt to press brightness key (may not work on all systems)
    try:
        for _ in range(10):
            pyautogui.press('brightnessup')
        return True
    except Exception:
        return False


def set_brightness_low():
    """Set brightness low (try screen_brightness_control, fallback to repeated key presses)."""
    try:
        if sbc:
            sbc.set_brightness(20)
            return True
    except Exception:
        pass
    try:
        for _ in range(10):
            pyautogui.press('brightnessdown')
        return True
    except Exception:
        return False


def take_screenshot():
    """Take screenshot and save to screenshots/ with timestamp."""
    t = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = os.path.join('screenshots', f'screenshot_{t}.png')
    try:
        img = pyautogui.screenshot()
        img.save(path)
        return path
    except Exception:
        return None


def snap_window_right():
    """Snap current active window to right (Windows: Win+Right)."""
    try:
        pyautogui.hotkey('win', 'ctrl', 'right')
        return True
    except Exception:
        return False


def snap_window_left():
    """Snap current active window to left (Windows: Win+Left)."""
    try:
        pyautogui.hotkey('win', 'ctrl', 'left')
        return True
    except Exception:
        return False


def volume_up():
    try:
        pyautogui.press('volumeup')
        return True
    except Exception:
        return False


def volume_down():
    try:
        pyautogui.press('volumedown')
        return True
    except Exception:
        return False


def volume_mute_toggle():
    try:
        pyautogui.press('volumemute')
        return True
    except Exception:
        return False


def next_track():
    try:
        pyautogui.press('nexttrack')
        return True
    except Exception:
        # fallback: try media next (some systems)
        try:
            pyautogui.hotkey('ctrl', 'right')
            return True
        except Exception:
            return False


def play_pause():
    try:
        pyautogui.press('playpause')
        return True
    except Exception:
        # fallback: send space (works in many media apps)
        try:
            pyautogui.press('space')
            return True
        except Exception:
            return False


# ------------------- Drawing helpers -------------------

def draw_countdown_overlay(image, seconds_left):
    h, w = image.shape[:2]
    overlay = image.copy()
    box_w, box_h = 200, 120
    x1, y1 = (w // 2 - box_w // 2, h // 2 - box_h // 2)
    x2, y2 = (x1 + box_w, y1 + box_h)
    cv.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv.addWeighted(overlay, 0.6, image, 0.4, 0, image)
    txt = f'Activating in {int(seconds_left)}'
    cv.putText(image, txt, (x1 + 20, y1 + 70),
               cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv.LINE_AA)
    return image


def draw_action_status(image, text):
    cv.putText(image, text, (10, 140),
               cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv.LINE_AA)
    return image


# ------------------- Main -------------------

def main():
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    use_brect = True

    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=1,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    keypoint_classifier = KeyPointClassifier()
    point_history_classifier = PointHistoryClassifier()
    finger_counter = FingerCounter()

    # Read labels
    with open('model/keypoint_classifier/keypoint_classifier_label.csv',
              encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [row[0] for row in keypoint_classifier_labels]
    with open('model/point_history_classifier/point_history_classifier_label.csv',
              encoding='utf-8-sig') as f:
        point_history_classifier_labels = csv.reader(f)
        point_history_classifier_labels = [row[0] for row in point_history_classifier_labels]

    cvFpsCalc = CvFpsCalc(buffer_len=10)

    history_length = 16
    point_history = deque(maxlen=history_length)
    finger_gesture_history = deque(maxlen=history_length)
    finger_count_history = deque(maxlen=5)

    # Right-hand countdown state
    right_count_target = None
    right_count_start = None
    right_count_duration = 3.0  # seconds to hold before action
    last_action_text = ""  # to display last action performed

    mode = 0

    while True:
        fps = cvFpsCalc.get()

        key = cv.waitKey(10)
        if key == 27:  # ESC
            break
        number, mode = select_mode(key, mode)

        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)
        debug_image = copy.deepcopy(image)

        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        finger_count = 0
        finger_states = None
        gesture_text = ""

        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                pre_processed_landmark_list = pre_process_landmark(landmark_list)
                pre_processed_point_history_list = pre_process_point_history(debug_image, point_history)
                logging_csv(number, mode, pre_processed_landmark_list, pre_processed_point_history_list)

                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                if hand_sign_id == 2:  # Point gesture
                    point_history.append(landmark_list[8])
                else:
                    point_history.append([0, 0])

                finger_gesture_id = 0
                point_history_len = len(pre_processed_point_history_list)
                if point_history_len == (history_length * 2):
                    finger_gesture_id = point_history_classifier(pre_processed_point_history_list)

                finger_gesture_history.append(finger_gesture_id)
                most_common_fg_id = Counter(finger_gesture_history).most_common()
                if most_common_fg_id:
                    fg_idx = most_common_fg_id[0][0]
                    gesture_text = point_history_classifier_labels[fg_idx] if fg_idx < len(point_history_classifier_labels) else ""

                finger_count = finger_counter.count_fingers(landmark_list)
                finger_count_history.append(finger_count)
                if len(finger_count_history) > 0:
                    finger_count = max(set(finger_count_history), key=finger_count_history.count)
                if args.show_finger_states:
                    finger_states = finger_counter.get_finger_states(landmark_list)

                # Determine handedness label (string like 'Left' or 'Right')
                hand_label = handedness.classification[0].label

                # Drawing
                debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                debug_image = draw_landmarks(debug_image, landmark_list)
                debug_image = draw_info_text(
                    debug_image,
                    brect,
                    handedness,
                    keypoint_classifier_labels[hand_sign_id],
                    point_history_classifier_labels[most_common_fg_id[0][0]] if most_common_fg_id else "",
                )
                if args.show_finger_count:
                    debug_image = draw_finger_count_info(debug_image, finger_count, finger_states, args.show_finger_states)

                # --- ACTION LOGIC FOR RIGHT HAND (with 3s countdown for 1/2/3) ---
                if hand_label.lower().startswith('right'):
                    # If finger_count is 1/2/3, start/continue countdown
                    if finger_count in (1, 2, 3):
                        if right_count_target != finger_count:
                            # new target, reset timer
                            right_count_target = finger_count
                            right_count_start = time.time()
                        else:
                            # same target, check elapsed
                            elapsed = time.time() - (right_count_start or time.time())
                            seconds_left = max(0, right_count_duration - elapsed)
                            debug_image = draw_countdown_overlay(debug_image, seconds_left)
                            # if held long enough -> trigger action and reset
                            if elapsed >= right_count_duration:
                                if right_count_target == 1:
                                    ok = set_brightness_high()
                                    last_action_text = "Right hand: brightness -> HIGH" if ok else "Brightness HIGH failed"
                                elif right_count_target == 2:
                                    ok = set_brightness_low()
                                    last_action_text = "Right hand: brightness -> LOW" if ok else "Brightness LOW failed"
                                elif right_count_target == 3:
                                    p = take_screenshot()
                                    last_action_text = f"Right hand: screenshot saved: {p}" if p else "Screenshot failed"
                                # reset countdown after action
                                right_count_target = None
                                right_count_start = None
                    else:
                        # no trigger fingers: check gesture for clockwise/ccw
                        right_count_target = None
                        right_count_start = None
                        # gesture_text may contain words 'clockwise' or 'counter' -> snap window
                        if gesture_text:
                            gt = gesture_text.lower()
                            if 'clockwise' in gt:
                                snap_window_right()
                                last_action_text = "Right gesture: snap -> RIGHT"
                            elif 'counter' in gt or 'anticlockwise' in gt or 'anti' in gt:
                                snap_window_left()
                                last_action_text = "Right gesture: snap -> LEFT"

                # --- ACTION LOGIC FOR LEFT HAND (immediate) ---
                elif hand_label.lower().startswith('left'):
                    # immediate actions (no countdown)
                    if finger_count == 1:
                        ok = volume_up()
                        last_action_text = "Left hand: volume up" if ok else "Volume up failed"
                    elif finger_count == 2:
                        ok = volume_down()
                        last_action_text = "Left hand: volume down" if ok else "Volume down failed"
                    elif finger_count == 3:
                        ok = volume_mute_toggle()
                        last_action_text = "Left hand: mute toggle" if ok else "Mute toggle failed"
                    elif finger_count == 4:
                        ok = next_track()
                        last_action_text = "Left hand: next track" if ok else "Next track failed"
                    else:
                        # if hand open (all fingers up) -> play/pause
                        # We'll check finger_states if available or use finger_count == 5
                        if (args.show_finger_states and finger_states is not None):
                            # determine open by checking all fingers UP (True)
                            all_up = all(finger_states.get(f, False) for f in ['thumb', 'index', 'middle', 'ring', 'little'])
                            if all_up:
                                ok = play_pause()
                                last_action_text = "Left hand: play/pause" if ok else "Play/pause failed"
                        else:
                            if finger_count == 5:
                                ok = play_pause()
                                last_action_text = "Left hand: play/pause" if ok else "Play/pause failed"

        else:
            point_history.append([0, 0])
            finger_count_history.append(0)
            # reset countdown if no hand
            right_count_target = None
            right_count_start = None

        debug_image = draw_point_history(debug_image, point_history)
        debug_image = draw_info(debug_image, fps, mode, number)
        if args.show_finger_count:
            debug_image = draw_finger_count_overlay(debug_image, finger_count)

        # show last action text
        if last_action_text:
            debug_image = draw_action_status(debug_image, last_action_text)

        cv.imshow('Hand Gesture Recognition with Finger Counting', debug_image)

    cap.release()
    cv.destroyAllWindows()


# ---------- existing utility functions remain unchanged ----------

def select_mode(key, mode):
    number = -1
    if 48 <= key <= 57:  # 0 ~ 9
        number = key - 48
    if key == 110:  # n
        mode = 0
    if key == 107:  # k
        mode = 1
    if key == 104:  # h
        mode = 2
    return number, mode


def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list))) if len(temp_landmark_list) > 0 else 1

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


def pre_process_point_history(image, point_history):
    image_width, image_height = image.shape[1], image.shape[0]

    temp_point_history = copy.deepcopy(point_history)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, point in enumerate(temp_point_history):
        if index == 0:
            base_x, base_y = point[0], point[1]

        temp_point_history[index][0] = (temp_point_history[index][0] - base_x) / image_width if image_width else 0
        temp_point_history[index][1] = (temp_point_history[index][1] - base_y) / image_height if image_height else 0

    # Convert to a one-dimensional list
    temp_point_history = list(itertools.chain.from_iterable(temp_point_history))

    return temp_point_history


def logging_csv(number, mode, landmark_list, point_history_list):
    if mode == 0:
        pass
    if mode == 1 and (0 <= number <= 9):
        csv_path = 'model/keypoint_classifier/keypoint.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])
    if mode == 2 and (0 <= number <= 9):
        csv_path = 'model/point_history_classifier/point_history.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *point_history_list])
    return


def draw_landmarks(image, landmark_point):
    if len(landmark_point) > 0:
        # Thumb
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (255, 255, 255), 2)

        # Index finger
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (255, 255, 255), 2)

        # Middle finger
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (255, 255, 255), 2)

        # Ring finger
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (255, 255, 255), 2)

        # Little finger
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (255, 255, 255), 2)

        # Palm
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (255, 255, 255), 2)

    # Key Points
    for index, landmark in enumerate(landmark_point):
        color_fill = (255, 255, 255)
        color_edge = (0, 0, 0)
        radius = 5 if index % 4 != 0 else 8
        cv.circle(image, (landmark[0], landmark[1]), radius, color_fill, -1)
        cv.circle(image, (landmark[0], landmark[1]), radius, color_edge, 1)

    return image


def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 0, 0), 1)
    return image


def draw_info_text(image, brect, handedness, hand_sign_text,
                   finger_gesture_text):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                 (0, 0, 0), -1)

    info_text = handedness.classification[0].label[0:]
    if hand_sign_text != "":
        info_text = info_text + ':' + hand_sign_text
    cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

    if finger_gesture_text != "":
        cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
        cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
                   cv.LINE_AA)

    return image


def draw_finger_count_info(image, finger_count, finger_states, show_states):
    if finger_states and show_states:
        y_offset = 120
        for finger, state in finger_states.items():
            color = (0, 255, 0) if state else (0, 0, 255)
            text = f"{finger.capitalize()}: {'UP' if state else 'DOWN'}"
            cv.putText(image, text, (10, y_offset),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv.LINE_AA)
            y_offset += 20
    return image


def draw_finger_count_overlay(image, finger_count):
    h, w = image.shape[:2]
    overlay = image.copy()
    cv.rectangle(overlay, (w - 150, 10), (w - 10, 110), (0, 0, 0), -1)
    cv.addWeighted(overlay, 0.7, image, 0.3, 0, image)
    cv.putText(image, str(finger_count), (w - 120, 85),
               cv.FONT_HERSHEY_SIMPLEX, 3.0, (0, 255, 0), 5, cv.LINE_AA)
    cv.putText(image, "Fingers", (w - 140, 30),
               cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv.LINE_AA)
    return image


def draw_point_history(image, point_history):
    for index, point in enumerate(point_history):
        if point[0] != 0 and point[1] != 0:
            cv.circle(image, (point[0], point[1]), 1 + int(index / 2),
                      (152, 251, 152), 2)
    return image


def draw_info(image, fps, mode, number):
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (255, 255, 255), 2, cv.LINE_AA)

    mode_string = ['Logging Key Point', 'Logging Point History']
    if 1 <= mode <= 2:
        cv.putText(image, "MODE:" + mode_string[mode - 1], (10, 90),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                   cv.LINE_AA)
        if 0 <= number <= 9:
            cv.putText(image, "NUM:" + str(number), (10, 110),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                       cv.LINE_AA)
    return image


if __name__ == '__main__':
    main()
