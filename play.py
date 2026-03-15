import cv2
import time
import numpy as np 
import random
import sys
import os
import pygame

from compute_angle import compute_angles
from compare import find_closest_vector

from utils import (
    initialize_pose_model,
    process_frame,
    blink_screen,
    draw_bold_text,
    extract_landmarks,
    save_data,
    predict_pose_v2,
    display_gameover_message,
)

import mediapipe as mp

LABEL_MAP = {"Jorunor": 0, "Josuke": 1, "Dio": 2, "Jonathan": 3, "Popo": 4, "Jotaro": 5, "Killer Queen": 6, "Joseph": 7, "Jolin": 8, "Comic": 9}
POSES = [pose for pose in LABEL_MAP.keys()]
SAVE_DATA = False
SAVE_PATH = "play_data"
RECT_TOP_LEFT = (950, 20)
RECT_BOTTOM_RIGHT = (350, 700)
RECT_COLOR = (0, 255, 0)
RECT_THICKNESS = 4

# ✅ 預載入背景圖
BACKGROUND_IMAGES = {
    "Jorunor": cv2.imread("backgrounds/Jorunor.png"),
    "Josuke": cv2.imread("backgrounds/Josuke.png"),
    "Dio": cv2.imread("backgrounds/Dio.png"),
    "Jonathan": cv2.imread("backgrounds/Jonathan.png"),
    "Popo": cv2.imread("backgrounds/Popo.png"),
    "Jotaro": cv2.imread("backgrounds/Jotaro1.png"),
    "Killer Queen": cv2.imread("backgrounds/killer_queen.jpeg"),
    "Joseph": cv2.imread("backgrounds/Joseph.png"),
    "Jolin": cv2.imread("backgrounds/Jolin.png"),
    "Comic": cv2.imread("backgrounds/Comic.png"), 
    "Wrong": cv2.imread("backgrounds/Wrong.png"),
}

POSE_IMAGES = {
    "Jorunor": cv2.imread("image_resize/joruno_resized.jpg"),
    "Josuke": cv2.imread("image_resize/josuke_resized.jpg"),
    "Dio": cv2.imread("image_resize/dio_resized.jpg"),
    "Jonathan": cv2.imread("image_resize/jonathan_resized.jpg"),
    "Popo": cv2.imread("image_resize/popo_resized.jpg"),
    "Jotaro": cv2.imread("image_resize/jotaro1_resized.jpg"),
    "Killer Queen": cv2.imread("image_resize/killer_queen.jpg"),
    "Joseph": cv2.imread("image_resize/joseph_resized.jpg"),
    "Jolin": cv2.imread("image_resize/jolin_resized.jpg"),
    "Comic": cv2.imread("image_resize/comic_resized.jpg"),    
    "Wrong": cv2.imread("backgrounds/Wrong.png"),
    "JoJo": cv2.imread("image_resize/jojo.jpg")
}

def get_pose_hint_image(pose_name, frame_height):
    hint_img = POSE_IMAGES.get(pose_name)
    if hint_img is None:
        return np.zeros((frame_height, int(frame_height * 3 / 4), 3), dtype=np.uint8)  # 給個空白圖
    # Resize 保持比例，讓高度跟 frame 一樣
    scale = frame_height / hint_img.shape[0]
    new_width = int(hint_img.shape[1] * scale)
    hint_resized = cv2.resize(hint_img, (new_width, frame_height))
    return hint_resized


# ✅ 顯示姿勢對應背景圖（人物去背）
def show_pose_background(cap, pose_label, selfie_seg_model, background_dict, duration=2):
    bg_image = background_dict.get(pose_label)
    if bg_image is None:
        return

    start_time = time.time()
    while time.time() - start_time < duration:
        success, frame = cap.read()
        if not success:
            break
        frame = cv2.flip(frame, 1)
        results_seg = selfie_seg_model.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        mask = results_seg.segmentation_mask
        condition = np.stack((mask,) * 3, axis=-1) > 0.3
        bg_resized = cv2.resize(bg_image, (frame.shape[1], frame.shape[0]))
        output_frame = np.where(condition, frame, bg_resized)
        cv2.imshow("JOJO POSE!", output_frame)
        if cv2.waitKey(5) & 0xFF == ord("q"):
            break

# ✅ 處理命令列參數
if len(sys.argv) != 3:
    print("EXITING GAME")
    print("Please provide the number of rounds and the time between poses as integers.")
    print("Usage: python play.py <ROUNDS> <TIME>; e.g., python play.py 10 5")
    sys.exit(1)

try:
    ROUNDS = int(sys.argv[1])
    COUNTDOWN = int(sys.argv[2])
except ValueError:
    print("Invalid arguments. Please use integers.")
    sys.exit(1)

def main():
    next_pose = random.choice(POSES)
    current_pose = None
    countdown = False
    count = COUNTDOWN
    rounds_left = ROUNDS
    points = 0
    correct_pose = False
    incorrect_pose = False

    capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    pose = initialize_pose_model()
    selfie_segmentation_model = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)

    pygame.init()
    pygame.mixer.init()
    audio_played = False
    pose_sound_played = False
    game_over_sound_played = False

    record_time = time.time()
    last_time_check = time.time()

    while True:
        success, frame = capture.read()

        if success:
            cv2.rectangle(frame, RECT_TOP_LEFT, RECT_BOTTOM_RIGHT, RECT_COLOR, thickness=RECT_THICKNESS)
            results, frame = process_frame(frame, pose)
            frame = cv2.flip(frame, 1)

            if countdown:
                if rounds_left > 0:
                    if correct_pose:
                        frame, correct_pose = blink_screen(frame, 1, record_time, correct_pose)
                    elif incorrect_pose:
                        frame, incorrect_pose = blink_screen(frame, 2, record_time, incorrect_pose)

                    text_playing = [
                        {"text": f"Left: {rounds_left}", "position": (50, 100)},
                        {"text": f"{next_pose.upper()}: {count} s", "position": (430, 100)},
                    ]
                    for data in text_playing:
                        draw_bold_text(frame, data["text"], data["position"], color=(0, 0, 255))

                    if time.time() - last_time_check >= 1:
                        count -= 1
                        last_time_check = time.time()

                    if count == 0:
                        # image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        # results = pose.process(image_rgb)
                        cv2.imwrite("./captured_frame.jpg", frame)
                        # ========= input vector =========
                        out_angles = compute_angles(results)
                        
                        count = COUNTDOWN
                        record_time = time.time()
                        current_pose = next_pose
                        rounds_left -= 1
                        pose_sound_played = False

                        while next_pose == current_pose:
                            next_pose = random.choice(POSES)
                            
                        #模擬情境
                        n = 12
                        error_type='SE'
                        known_vectors = [  #多個動作的向量
                            [295, 242, 214, 336, 165, 263, 272, 3, 261, 256, 273, 293],
                            [67, 153, 148, 5, 195, 264, 276, 344, 284, 272, 263, 258],
                            [14, 120, 264, 336, 195, 274, 282, 16, 276, 254, 271, 254],
                            [294, 299, 280, 83, 189, 267, 274, 359, 254, 251, 278, 289],
                            [36, 177, 223, 217, 186, 258, 268, 349, 269, 289, 263, 288],
                            [301, 244, 250, 103, 180, 266, 273, 359, 258, 257, 270, 270],
                            [156, 80, 210, 272, 192, 265, 267, 4, 246, 283, 276, 306],
                            [82, 152, 99, 28, 180, 267, 273, 7, 271, 277, 282, 264],
                            [272, 229, 287, 65, 188, 270, 277, 15, 271, 275, 291, 255],
                            [82, 165, 263, 270, 190, 266, 274, 357, 274, 271, 264, 253]
                        ]

                        pred_idx = find_closest_vector(n, out_angles, known_vectors, error_type)
                        
                        print(f'pred_idx = {pred_idx}')
                        predicted_pose = ""
                        for k, v in LABEL_MAP.items():
                            if v == pred_idx:
                                predicted_pose = k
                                break
                                                 
                        print(f"\nRound {ROUNDS - rounds_left}")
                        print(f"Pose to pose: {current_pose}")
                        print(f"Recorded Pose: {predicted_pose}")

                        if predicted_pose == current_pose:
                            correct_pose = True
                            points += 1
                            print("***\nGood job!")
                            print(f"Points:{points}/{ROUNDS}")

                            # ✅ 顯示背景
                            show_pose_background(capture, current_pose, selfie_segmentation_model, BACKGROUND_IMAGES)

                        else:
                            incorrect_pose = True
                            print("***\nBetter luck next time!")
                            print(f"Points:{points}/{ROUNDS}")
                            show_pose_background(capture, "Wrong", selfie_segmentation_model, BACKGROUND_IMAGES)

                        if SAVE_DATA:
                            save_data(SAVE_PATH, current_pose, frame, results, predicted_pose)
                            print(f"Data for Round {ROUNDS - rounds_left} saved in '{SAVE_PATH}'.")

                if rounds_left == 0:
                    if not game_over_sound_played:
                        print("\nGame Over!")
                        print(f"Points:{points}/{ROUNDS}")
                        game_over_sound_played = True

                    white_rect = np.zeros_like(frame) + 255
                    frame = cv2.addWeighted(frame, 0.5, white_rect, 0.5, 0)
                    display_gameover_message(frame, points, ROUNDS)

                    key = cv2.waitKey(1)
                    if key == ord(" "):
                        while pygame.mixer.get_busy():
                            pass
                        countdown = True
                        rounds_left = ROUNDS
                        points = 0
                        last_time_check = time.time()
                        game_over_sound_played = False
                        next_pose = random.choice(POSES)
                        current_pose = None

            else:
                white_rect = np.zeros_like(frame) + 255
                frame = cv2.addWeighted(frame, 0.5, white_rect, 0.5, 0)
                start_screen_text = [
                    {"text": "Press 'SPACE' to start", "position": (270, 650), "scale": 2, "color": (255, 0, 0), "thickness": 2, "offset": 1},
                    {"text": "JOJO", "position": (470, 110), "scale": 3, "color": (0, 0, 255), "thickness": 4, "offset": 3},
                    {"text": "POSE!", "position": (440, 210), "scale": 3, "color": (0, 0, 255), "thickness": 4, "offset": 3},
                ]
                for element in start_screen_text:
                    if element["text"] == "Press 'SPACE' to start" and int(time.time() * 2) % 2 != 0:
                        continue
                    draw_bold_text(frame, element["text"], element["position"], font_scale=element["scale"], color=element["color"], thickness=element["thickness"], offset=element["offset"])

            if countdown and rounds_left != 0:
                pose_hint_name = next_pose
            elif rounds_left == ROUNDS:
                pose_hint_name = "JoJo"  # 開始前畫面
            elif rounds_left == 0:
                pose_hint_name = "JoJo"  # 結束後畫面
            else:
                pose_hint_name = "Wrong"  # 其他情況預設為 Comic

            hint_image = get_pose_hint_image(pose_hint_name, frame.shape[0])
            # hint_image = get_pose_hint_image(next_pose if countdown else "Comic", frame.shape[0])
            
            # --- 修改開始：固定總視窗寬度，動態調整主畫面寬度 ---
            TARGET_TOTAL_WIDTH = 1500  # 設定適合 1920x1080 螢幕的固定總寬度 (你也可以依喜好微調成 1440 或 1600)
            main_frame_width = TARGET_TOTAL_WIDTH - hint_image.shape[1]
            
            # 將主畫面縮放至剩下的寬度，高度則保持一致
            frame_resized = cv2.resize(frame, (main_frame_width, hint_image.shape[0]))
            # --- 修改結束 ---

            combined_display = np.hstack((frame_resized, hint_image))
            cv2.imshow("JOJO POSE!", combined_display)
            # cv2.imshow("STRIKE A POSE!", frame)

            key = cv2.waitKey(5)
            if key == ord("q") or key == ord("Q"):
                pygame.mixer.stop()
                break
            if key == ord(" "):
                countdown = True
                last_time_check = time.time()
            if key == ord("r"):
                capture.release()
                cv2.destroyAllWindows()
                main()

        else:
            print("Closing camera.")
            break

    capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
