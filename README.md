# Gesture Controller

Control your laptop with hand gestures using MediaPipe and PyAutoGUI.

## Gestures
- **Cursor**: Index finger up
- **Left Click**: Index tip + Thumb tip pinch
- **Right Click**: Middle tip + Thumb tip pinch

## Setup
1. Download `hand_landmarker.task` from [here](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task) and place it in the project folder
2. `pip install mediapipe opencv-python pyautogui`
3. `python hand_detector.py`

Press `Q` to quit.

## TODO
- [ ] Drag & Drop
- [ ] Volume Control
- [ ] Presentation Mode
