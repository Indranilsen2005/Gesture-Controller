import cv2
import mediapipe as mp
import pyautogui
import math

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.3,
    min_hand_presence_confidence=0.3
)

screen_w, screen_h = pyautogui.size()
pyautogui.FAILSAFE = False

MARGIN_X = 0.2
MARGIN_Y = 0.2
SMOOTHING = 0.7
CLICK_THRESHOLD = 0.05
FREEZE_THRESHOLD = 0.08

smooth_x, smooth_y = screen_w // 2, screen_h // 2
left_click_ready = True
right_click_ready = True

def get_distance(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def finger_is_up(landmarks, tip_idx, pip_idx):
    return landmarks[tip_idx].y < landmarks[pip_idx].y

cap = cv2.VideoCapture(0)

with HandLandmarker.create_from_options(options) as landmarker:
    prev_scroll_y = None  

    while True:
        success, frame = cap.read()
        h, w, _ = frame.shape

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        timestamp = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        result = landmarker.detect_for_video(mp_image, timestamp)

        if result.hand_landmarks:
            landmarks = result.hand_landmarks[0]

            index_tip = landmarks[8]
            middle_tip = landmarks[12]
            thumb_tip = landmarks[4]

            index_up  = finger_is_up(landmarks, 8, 6)
            middle_up = finger_is_up(landmarks, 12, 10)

            left_dist  = get_distance(thumb_tip, index_tip)
            right_dist = get_distance(thumb_tip, middle_tip)

            # Determine gesture
            if index_up and middle_up:
                gesture = 'scroll'
            elif left_dist < FREEZE_THRESHOLD:
                gesture = 'left_click'
            elif right_dist < FREEZE_THRESHOLD:
                gesture = 'right_click'
            else:
                gesture = 'cursor'

            # Execute gesture
            if gesture == 'scroll':
                left_click_ready = True
                right_click_ready = True
                prev_scroll_y = None

                index_y = (landmarks[8].y + landmarks[12].y) / 2  # 0 = top, 1 = bottom

                DEAD_TOP    = 0.45
                DEAD_BOTTOM = 0.50

                if index_y < DEAD_TOP:
                    speed = int((DEAD_TOP - index_y) * 1000)
                    pyautogui.scroll(speed)   # scroll up
                elif index_y > DEAD_BOTTOM:
                    speed = int((index_y - DEAD_BOTTOM) * 1000)
                    pyautogui.scroll(-speed)  # scroll down

            elif gesture == 'left_click':
                prev_scroll_y = None
                if left_dist < CLICK_THRESHOLD:
                    if left_click_ready:
                        pyautogui.click(button='left')
                        left_click_ready = False
                else:
                    left_click_ready = True

            elif gesture == 'right_click':
                prev_scroll_y = None
                if right_dist < CLICK_THRESHOLD:
                    if right_click_ready:
                        pyautogui.click(button='right')
                        right_click_ready = False
                else:
                    right_click_ready = True

            else:  # cursor
                prev_scroll_y = None
                left_click_ready = True
                right_click_ready = True

                x = (index_tip.x - MARGIN_X) / (1 - 2 * MARGIN_X)
                y = (index_tip.y - MARGIN_Y) / (1 - 2 * MARGIN_Y)
                x = max(0, min(1, x))
                y = max(0, min(1, y))

                target_x = int((1 - x) * screen_w)
                target_y = int(y * screen_h)

                smooth_x = smooth_x * SMOOTHING + target_x * (1 - SMOOTHING)
                smooth_y = smooth_y * SMOOTHING + target_y * (1 - SMOOTHING)
                pyautogui.moveTo(int(smooth_x), int(smooth_y))

            # Draw index tip
            px = int(index_tip.x * w)
            py = int(index_tip.y * h)
            cv2.circle(frame, (px, py), 8, (0, 255, 0), -1)

        cv2.imshow("Hand Detector", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()