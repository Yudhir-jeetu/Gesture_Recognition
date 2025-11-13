#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
app_with_actions_v2.py
Variant of the action-enabled script with these changes:
- 1 finger = pointer / clockwise / counterclockwise only (no actions)
- All actionable gestures start at 2 fingers
  - Right hand: 2 -> brightness HIGH, 3 -> brightness LOW, 4 -> screenshot (3s hold)
  - Left hand: 2 -> volume UP, 3 -> volume DOWN, 4 -> mute (immediate)
  - Open left hand (all fingers) -> play/pause
- Removed left-hand "next track"
Keep this file separate from your original script.
"""

import csv
import copy
import argparse
import itertools
import os
import time
from collections import Counter, deque
from datetime import datetime

import cv2 as cv
import numpy as np
import mediapipe as mp

import pyautogui
try:
    import screen_brightness_control as sbc
except Exception:
    sbc = None

from utils import CvFpsCalc
from utils.finger_counter import FingerCounter
from model import KeyPointClassifier
from model import PointHistoryClassifier

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
                        type=float,
                        default=0.5)
    parser.add_argument('--show_finger_count', action='store_true', default=True,
                        help='Display finger count on screen')
    parser.add_argument('--show_finger_states', action='store_true', default=False,
                        help='Display individual finger states')
    return parser.parse_args()


# --------- Action helpers ---------

def set_brightness_high():
    try:
        if sbc:
            sbc.set_brightness(100)
            return True
    except Exception:
        pass
    try:
        for _ in range(8):
            pyautogui.press('brightnessup')
        return True
    except Exception:
        return False


def set_brightness_low():
    try:
        if sbc:
            sbc.set_brightness(20)
            return True
    except Exception:
        pass
    try:
        for _ in range(8):
            pyautogui.press('brightnessdown')
        return True
    except Exception:
        return False


def take_screenshot():
    t = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = os.path.join('screenshots', f'screenshot_{t}.png')
    try:
        img = pyautogui.screenshot()
        img.save(path)
        return path
    except Exception:
        return None


def snap_window_right():
    try:
        pyautogui.hotkey('win','ctrl', 'right')
        return True
    except Exception:
        return False


def snap_window_left():
    try:
        pyautogui.hotkey('win','ctrl', 'left')
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


def play_pause():
    try:
        pyautogui.press('playpause')
        return True
    except Exception:
        try:
            pyautogui.press('space')
            return True
        except Exception:
            return False


# --------- Drawing helpers ---------

def draw_countdown_overlay(image, seconds_left):
    h, w = image.shape[:2]
    overlay = image.copy()
    box_w, box_h = 240, 140
    x1, y1 = (w // 2 - box_w // 2, h // 2 - box_h // 2)
    x2, y2 = (x1 + box_w, y1 + box_h)
    cv.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv.addWeighted(overlay, 0.6, image, 0.4, 0, image)
    txt = f'Activating in {int(seconds_left)+1}' if seconds_left > 0 else 'Activating...'
    cv.putText(image, txt, (x1 + 20, y1 + 85),
               cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv.LINE_AA)
    return image


def draw_action_status(image, text):
    cv.putText(image, text, (10, 140),
               cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv.LINE_AA)
    return image


# --------- Core ---------

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
        max_num_hands=2,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    keypoint_classifier = KeyPointClassifier()
    point_history_classifier = PointHistoryClassifier()
    finger_counter = FingerCounter()

    with open('model/keypoint_classifier/keypoint_classifier_label.csv',
              encoding='utf-8-sig') as f:
        keypoint_classifier_labels = [row[0] for row in csv.reader(f)]
    with open('model/point_history_classifier/point_history_classifier_label.csv',
              encoding='utf-8-sig') as f:
        point_history_classifier_labels = [row[0] for row in csv.reader(f)]

    cvFpsCalc = CvFpsCalc(buffer_len=10)
    history_length = 16
    point_history = deque(maxlen=history_length)
    finger_gesture_history = deque(maxlen=history_length)
    per_hand_counts = {'Left': deque(maxlen=5), 'Right': deque(maxlen=5)}

    # Right-hand countdown state (for actions that require hold)
    right_count_target = None
    right_count_start = None
    right_count_duration = 3.0
    last_action_text = ""

    mode = 0

    while True:
        fps = cvFpsCalc.get()
        key = cv.waitKey(10)
        if key == 27:
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

        finger_count_display = 0
        finger_states_display = None
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
                if hand_sign_id == 2:
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
                    if 0 <= fg_idx < len(point_history_classifier_labels):
                        gesture_text = point_history_classifier_labels[fg_idx]
                    else:
                        gesture_text = ""

                # count fingers and smoothing
                fc = finger_counter.count_fingers(landmark_list)
                label = handedness.classification[0].label  # 'Left' or 'Right'
                if label not in per_hand_counts:
                    per_hand_counts[label] = deque(maxlen=5)
                per_hand_counts[label].append(fc)
                if len(per_hand_counts[label]) > 0:
                    smoothed = max(set(per_hand_counts[label]), key=per_hand_counts[label].count)
                else:
                    smoothed = fc

                if args.show_finger_states:
                    finger_states = finger_counter.get_finger_states(landmark_list)
                else:
                    finger_states = None

                # Draw visuals
                debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                debug_image = draw_landmarks(debug_image, landmark_list)
                info_fg_text = ""
                if most_common_fg_id:
                    try:
                        info_fg_text = point_history_classifier_labels[most_common_fg_id[0][0]]
                    except Exception:
                        info_fg_text = ""
                debug_image = draw_info_text(
                    debug_image,
                    brect,
                    handedness,
                    keypoint_classifier_labels[hand_sign_id] if 0 <= hand_sign_id < len(keypoint_classifier_labels) else "",
                    info_fg_text,
                )
                if args.show_finger_count:
                    debug_image = draw_finger_count_info(debug_image, smoothed, finger_states, args.show_finger_states)

                # ---------- ACTION RULES (UPDATED) ----------
                # RIGHT HAND:
                #  - 1 finger => pointer / clockwise / counterclockwise only (no actions)
                #  - 2 => brightness HIGH (3s hold)
                #  - 3 => brightness LOW (3s hold)
                #  - 4 => screenshot (3s hold)
                if label.lower().startswith('right'):
                    finger_count_display = smoothed
                    finger_states_display = finger_states

                    if smoothed == 1:
                        # pointer / check gesture_text for clockwise/counterclockwise snapping
                        if gesture_text:
                            gt = gesture_text.lower()
                            if 'clockwise' in gt or 'cw' in gt:
                                snap_window_right()
                                last_action_text = "Right gesture: snap -> RIGHT"
                            elif 'counter' in gt or 'anticlockwise' in gt or 'anti' in gt or 'ccw' in gt:
                                snap_window_left()
                                last_action_text = "Right gesture: snap -> LEFT"
                        # don't start countdown for single finger
                        right_count_target = None
                        right_count_start = None
                    elif smoothed in (2, 3, 4):
                        # start or continue countdown for actions
                        if right_count_target != smoothed:
                            right_count_target = smoothed
                            right_count_start = time.time()
                        else:
                            elapsed = time.time() - (right_count_start or time.time())
                            seconds_left = max(0, right_count_duration - elapsed)
                            debug_image = draw_countdown_overlay(debug_image, seconds_left)
                            if elapsed >= right_count_duration:
                                if right_count_target == 2:
                                    ok = set_brightness_high()
                                    last_action_text = "Right: Brightness -> HIGH" if ok else "Right: Brightness HIGH failed"
                                elif right_count_target == 3:
                                    ok = set_brightness_low()
                                    last_action_text = "Right: Brightness -> LOW" if ok else "Right: Brightness LOW failed"
                                elif right_count_target == 4:
                                    p = take_screenshot()
                                    last_action_text = f"Right: Screenshot saved: {p}" if p else "Right: Screenshot failed"
                                right_count_target = None
                                right_count_start = None
                    else:
                        right_count_target = None
                        right_count_start = None

                # LEFT HAND:
                #  - 1 finger => pointer / clockwise / counterclockwise only (no actions)
                #  - 2 => volume UP (immediate)
                #  - 3 => volume DOWN (immediate)
                #  - 4 => mute/unmute (immediate)
                #  - open hand (all fingers up) => play/pause
                elif label.lower().startswith('left'):
                    finger_count_display = smoothed
                    finger_states_display = finger_states

                    if smoothed == 1:
                        # pointer / rotation only; do not trigger actions
                        # (if your point_history classifier reports clockwise/counter, you can use it similarly)
                        pass
                    elif smoothed == 2:
                        ok = volume_up()
                        last_action_text = "Left: volume up" if ok else "Left: volume up failed"
                    elif smoothed == 3:
                        ok = volume_down()
                        last_action_text = "Left: volume down" if ok else "Left: volume down failed"
                    elif smoothed == 4:
                        ok = volume_mute_toggle()
                        last_action_text = "Left: mute toggle" if ok else "Left: mute toggle failed"
                    else:
                        # detect open hand (all fingers up) either via finger_states or smoothed==5
                        is_open = False
                        if finger_states is not None:
                            try:
                                is_open = all(finger_states.get(k, False) for k in ['thumb', 'index', 'middle', 'ring', 'little'])
                            except Exception:
                                is_open = False
                        else:
                            is_open = (smoothed == 5)
                        if is_open:
                            ok = play_pause()
                            last_action_text = "Left: play/pause" if ok else "Left: play/pause failed"

        else:
            point_history.append([0, 0])
            per_hand_counts['Left'].append(0)
            per_hand_counts['Right'].append(0)
            right_count_target = None
            right_count_start = None

        debug_image = draw_point_history(debug_image, point_history)
        debug_image = draw_info(debug_image, fps, mode, number)
        if args.show_finger_count:
            debug_image = draw_finger_count_overlay(debug_image, finger_count_display)

        if last_action_text:
            debug_image = draw_action_status(debug_image, last_action_text)

        cv.imshow('Hand Gesture Recognition with Actions (v2)', debug_image)

    cap.release()
    cv.destroyAllWindows()


# --------- Original helper functions (unchanged) ---------

def select_mode(key, mode):
    number = -1
    if 48 <= key <= 57:
        number = key - 48
    if key == 110:
        mode = 0
    if key == 107:
        mode = 1
    if key == 104:
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
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])
    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]
        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
    max_value = max(list(map(abs, temp_landmark_list))) if len(temp_landmark_list) > 0 else 1
    def normalize_(n):
        return n / max_value
    temp_landmark_list = list(map(normalize_, temp_landmark_list))
    return temp_landmark_list


def pre_process_point_history(image, point_history):
    image_width, image_height = image.shape[1], image.shape[0]
    temp_point_history = copy.deepcopy(point_history)
    base_x, base_y = 0, 0
    for index, point in enumerate(temp_point_history):
        if index == 0:
            base_x, base_y = point[0], point[1]
        temp_point_history[index][0] = (temp_point_history[index][0] - base_x) / image_width if image_width else 0
        temp_point_history[index][1] = (temp_point_history[index][1] - base_y) / image_height if image_height else 0
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
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]), (255, 255, 255), 2)
        # Index
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]), (255, 255, 255), 2)
        # Middle
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]), (255, 255, 255), 2)
        # Ring
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]), (255, 255, 255), 2)
        # Little
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]), (255, 255, 255), 2)
        # Palm lines
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]), (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]), (255, 255, 255), 2)

    for index, landmark in enumerate(landmark_point):
        if index == 0:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        else:
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255), -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)

    return image


def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]), (0, 0, 0), 1)
    return image


def draw_info_text(image, brect, handedness, hand_sign_text, finger_gesture_text):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22), (0, 0, 0), -1)
    info_text = handedness.classification[0].label[0:]
    if hand_sign_text != "":
        info_text = info_text + ':' + hand_sign_text
    cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)
    if finger_gesture_text != "":
        cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
        cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv.LINE_AA)
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
            cv.circle(image, (point[0], point[1]), 1 + int(index / 2), (152, 251, 152), 2)
    return image


def draw_info(image, fps, mode, number):
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (255, 255, 255), 2, cv.LINE_AA)
    mode_string = ['Logging Key Point', 'Logging Point History']
    if 1 <= mode <= 2:
        cv.putText(image, "MODE:" + mode_string[mode - 1], (10, 90),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)
        if 0 <= number <= 9:
            cv.putText(image, "NUM:" + str(number), (10, 110),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)
    return image


if __name__ == '__main__':
    main()
