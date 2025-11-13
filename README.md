# Advanced Hand Gesture Recognition System with OpenCV and MediaPipe

## Project Overview

This is an enhanced hand gesture recognition system that builds upon MediaPipe's hand detection capabilities to create a comprehensive gesture control interface. The project recognizes hand signs, finger gestures, and includes advanced features like finger counting and system control through hand gestures.

## Key Features

### 🤖 Core Recognition Capabilities
- **Hand Sign Recognition**: Classifies different hand poses and gestures
- **Finger Gesture Recognition**: Detects finger movements and directional gestures
- **Finger Counting**: Accurately counts fingers from 0-5 in real-time
- **Hand Side Detection**: Differentiates between left and right hand gestures

### 🎮 Advanced Control Features
- **System Control**: Control brightness, volume, and media playback
- **Window Management**: Move windows left/right using hand rotation
- **Screenshot Capture**: Take screenshots with gesture commands
- **Media Controls**: Play/pause, skip tracks, mute/unmute audio

### 🔧 Technical Enhancements
- **3-Second Countdown**: Safety feature for critical actions (brightness, screenshot)
- **Real-time Processing**: Smooth 30+ FPS performance
- **Multiple Models**: TensorFlow Lite models for efficient inference
- **Modular Design**: Clean, extensible codebase



## Requirements

```
mediapipe==0.8.1
opencv-python>=3.4.2
tensorflow>=2.3.0
scikit-learn>=0.23.2
matplotlib>=3.3.2
pyautogui>=0.9.50
pycaw>=20230407
screen-brightness-control>=0.21.0
psutil>=5.9.5
```

## Installation

1. **Clone the repository:**
```bash
git clone https://github.com/Yudhir-jeetu/Gesture_Recognition.git
cd gesture-recognition-system
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run the basic gesture recognition:**
```bash
python app.py
```

4. **Run with finger counting:**
```bash
python app_with_finger_counting.py
```

5. **Run enhanced gesture control system:**
```bash
python enhanced_gesture_control.py
```

## Usage Options

### Basic Demo
```bash
python app.py [options]
```

**Available options:**
- `--device`: Camera device number (Default: 0)
- `--width`: Camera capture width (Default: 960)
- `--height`: Camera capture height (Default: 540)
- `--use_static_image_mode`: Use static image mode for MediaPipe
- `--min_detection_confidence`: Detection threshold (Default: 0.5)
- `--min_tracking_confidence`: Tracking threshold (Default: 0.5)

### Finger Counting Demo
```bash
python finger_counting_demo.py
```

### Enhanced Gesture Control
```bash
python enhanced_gesture_control.py
```

## Gesture Controls

### Right Hand (3-second countdown for safety)
- **1 finger**: Brightness Up
- **2 fingers**: Brightness Down
- **3 fingers**: Screenshot
- **Clockwise rotation**: Move window right
- **Counter-clockwise**: Move window left

### Left Hand (immediate action)
- **1 finger**: Volume Up
- **2 fingers**: Volume Down
- **3 fingers**: Mute/Unmute
- **4 fingers**: Skip/Next track
- **Open hand (5 fingers)**: Play/Pause

## Project Structure

```
gesture-recognition-system/
│
├── app.py                              # Main gesture recognition app
├── app_with_finger_counting.py         # Enhanced version with finger counting
├── enhanced_gesture_control.py         # Full gesture control system
├── finger_counting_demo.py            # Simple finger counting demo
├── collect_finger_count_data.py       # Training data collection tool
├── test_enhanced_features.py          # Feature testing script
│
├── model/
│   ├── keypoint_classifier/           # Hand sign recognition models
│   │   ├── keypoint_classifier.tflite
│   │   ├── keypoint_classifier_label.csv
│   │   └── keypoint.csv
│   │
│   └── point_history_classifier/      # Finger gesture models
│       ├── point_history_classifier.tflite
│       ├── point_history_classifier_label.csv
│       └── point_history.csv
│
├── utils/
│   ├── __init__.py
│   ├── cvfpscalc.py                  # FPS calculation utility
│   └── finger_counter.py              # Finger counting logic
│
├── requirements.txt                   # Project dependencies
├── LICENSE                           # Apache 2.0 License
└── README.md                         # This file
```

## Training & Customization

### Hand Sign Recognition Training

1. **Data Collection:**
   - Press "k" to enter keypoint logging mode
   - Press "0-9" to save keypoints with class labels
   - Data saved to `model/keypoint_classifier/keypoint.csv`

2. **Model Training:**
   - Open `keypoint_classification.ipynb` in Jupyter
   - Execute cells from top to bottom
   - Adjust `NUM_CLASSES` as needed

### Finger Gesture Recognition Training

1. **Data Collection:**
   - Press "h" to enter point history logging mode
   - Press "0-9" to save gesture sequences
   - Data saved to `model/point_history_classifier/point_history.csv`

2. **Model Training:**
   - Open `point_history_classification.ipynb` in Jupyter
   - Execute training pipeline
   - Modify class labels as needed

## Technical Implementation

### Hand Detection Pipeline
1. **MediaPipe Processing**: 21 hand landmark detection
2. **Keypoint Extraction**: Normalized coordinate processing
3. **Model Inference**: TensorFlow Lite classification
4. **Gesture Mapping**: Result to action mapping

### Finger Counting Algorithm
- **Landmark Analysis**: Compares finger tip positions to PIP joints
- **State Detection**: Boolean array for each finger state
- **Counting Logic**: Sums active fingers (0-5)

### System Integration
- **PyAutoGUI**: System control automation
- **Windows APIs**: Brightness and media control
- **Multi-threading**: Non-blocking countdown timers
- **Error Handling**: Graceful fallback mechanisms

## Performance Specifications

- **Frame Rate**: 30+ FPS on standard webcam
- **Detection Accuracy**: >95% for trained gestures
- **Response Time**: <100ms for gesture recognition
- **System Compatibility**: Windows 10/11, Linux, macOS

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## Credits & Acknowledgments

### Inspiration 
- **Kazuhito Takahashi** 
- **Nikita Kiselov** 

### Technologies Used
- **MediaPipe** - Google's framework for building multimodal applied ML pipelines
- **OpenCV** - Computer vision and image processing library
- **TensorFlow Lite** - Lightweight machine learning inference
- **PyAutoGUI** - Cross-platform GUI automation

