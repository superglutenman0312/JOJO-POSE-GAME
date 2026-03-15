import numpy as np

def compute_angles(results):
    # 設定要計算哪些向量的角度
    vector_list = [(11, 13), (13, 15), (12, 14), (14, 16),(11, 12),(11, 23),(12, 24),(24, 23),(24, 26),(26, 28),(23, 25),(25, 27)]  
    angles = []
    ground_vec = np.array([1.0, 0.0])  # 地面水平向右

    if not results.pose_landmarks:
        return angles

    landmarks = results.pose_landmarks.landmark
    # for idx, value in enumerate(landmarks):
    #     print(idx, value)
    

    for joint_a, joint_b in vector_list:
        ax, ay = landmarks[joint_a].x, landmarks[joint_a].y
        bx, by = landmarks[joint_b].x, landmarks[joint_b].y

        joint_vec = np.array([bx - ax, by - ay])

        if np.linalg.norm(joint_vec) == 0:
	    #angle = 0
            angle = None
        else:
            # 計算逆時針角度
            angle_rad = np.arctan2(joint_vec[1], joint_vec[0]) - np.arctan2(ground_vec[1], ground_vec[0])
            angle_deg_ccw = np.degrees(angle_rad) % 360
            # 轉換為順時針角度
            angle_deg_cw = (360 - angle_deg_ccw) % 360
            angle = int(angle_deg_cw)

        angles.append(angle)

    return angles
