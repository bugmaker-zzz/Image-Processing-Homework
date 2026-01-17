import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import RunningMode
import time
import argparse
import os
import urllib.request
import math
from collections import deque, Counter


class GestureBuffer:
    def __init__(self, buffer_len=5):
        # 队列长度为5，需要连续一致或由于概率高才能切换，极大降低误触
        self.buffer = deque(maxlen=buffer_len)
    
    def add_gesture(self, gesture):
        self.buffer.append(gesture)
    
    def get_stable_gesture(self):
        """返回缓冲区中出现次数最多的手势"""
        if len(self.buffer) < 1:
            return "UNKNOWN"
        count = Counter(self.buffer)
        most_common, num = count.most_common(1)[0]
        return most_common

class HandDetector:
    def __init__(self):
        """初始化手部检测器"""
        model_path = self._ensure_model_file()
        
        # 初始化防抖器
        self.gesture_buffer = GestureBuffer(buffer_len=5)
        
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=RunningMode.IMAGE,
            num_hands=2,  # 建议先设为1，确保控制逻辑稳定
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.5
        )
        
        self.hand_landmarker = vision.HandLandmarker.create_from_options(options)
        self.tip_ids = [4, 8, 12, 16, 20]
        self.results = None
    
    def _ensure_model_file(self):
        """确保模型文件存在"""
        if not os.path.exists('./model'):
            os.makedirs('./model')
        
        model_path = './model/hand_landmarker.task'
        
        if not os.path.exists(model_path):
            print("正在下载手部检测模型...")
            model_url = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task'
            try:
                urllib.request.urlretrieve(model_url, model_path)
                print(f"模型已下载到: {model_path}")
            except Exception as e:
                print(f"下载模型失败: {e}")
                raise
        return model_path

    def find_hands(self, img, draw=True):
        """检测手部并绘制骨架"""
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, c = img.shape
        
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        self.results = self.hand_landmarker.detect(mp_image)
        
        if draw and self.results.hand_landmarks:
            for hand_landmarks in self.results.hand_landmarks:
                self._draw_hand_landmarks(img, hand_landmarks, h, w)
        return img
    
    def _draw_hand_landmarks(self, img, hand_landmarks, h, w):
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (0, 9), (9, 10), (10, 11), (11, 12),
            (0, 13), (13, 14), (14, 15), (15, 16),
            (0, 17), (17, 18), (18, 19), (19, 20),
            (5, 9), (9, 13), (13, 17)
        ]
        points = []
        for landmark in hand_landmarks:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            points.append((x, y))
            cv2.circle(img, (x, y), 5, (0, 255, 0), -1) # 关键点绿色
        
        for start, end in connections:
            if start < len(points) and end < len(points):
                cv2.line(img, points[start], points[end], (255, 0, 0), 2) # 连线蓝色

    def get_gesture(self, img):
        """核心逻辑：宽容判定的手势识别"""
        if not self.results or not self.results.hand_landmarks:
            return "NO HAND", "None"
        
        if len(self.results.hand_landmarks) >= 2:
            return "TWO_HANDS", "None"

        hand_landmarks = self.results.hand_landmarks[0]
        
        # 1. 左右手标签
        mp_label = self.results.handedness[0][0].category_name
        hand_label = "Right" if mp_label == "Left" else "Left"
        
        lm_list = []
        h, w, c = img.shape
        for id, landmark in enumerate(hand_landmarks):
            cx, cy = int(landmark.x * w), int(landmark.y * h)
            lm_list.append({'id': id, 'x': cx, 'y': cy})

        # --- 手指状态列表 ---
        # fingers[0] = 拇指, fingers[1] = 食指, ... fingers[4] = 小指
        fingers = []

        # 2. 拇指判定 (欧氏距离法，保留备用，但降低 FIST 判定的权重)
        thumb_tip_x, thumb_tip_y = lm_list[4]['x'], lm_list[4]['y']
        pinky_mcp_x, pinky_mcp_y = lm_list[17]['x'], lm_list[17]['y']
        index_mcp_x, index_mcp_y = lm_list[5]['x'], lm_list[5]['y']
        
        thumb_dist = math.hypot(thumb_tip_x - pinky_mcp_x, thumb_tip_y - pinky_mcp_y)
        palm_width = math.hypot(index_mcp_x - pinky_mcp_x, index_mcp_y - pinky_mcp_y)

        # 稍微放宽拇指张开的阈值
        if thumb_dist > palm_width * 0.85:
            fingers.append(1) # 拇指张开
        else:
            fingers.append(0) # 拇指弯曲

        # 3. 四指判定 (食指到小指)
        for id in range(1, 5):
            tip_y = lm_list[self.tip_ids[id]]['y']
            pip_y = lm_list[self.tip_ids[id] - 2]['y'] 
            mcp_y = lm_list[self.tip_ids[id] - 3]['y']
            
            # 必须严格高于指根才算伸出
            if tip_y < pip_y and tip_y < mcp_y:
                fingers.append(1)
            else:
                fingers.append(0)

        # --- 核心修改：基于组合逻辑的宽容判定 ---
        
        # 统计除拇指外的 4 根手指伸出的数量
        # fingers[1:] 就是 [食指, 中指, 无名指, 小指]
        four_fingers_count = fingers[1:].count(1)
        
        gesture = "UNKNOWN"
        
        # 判定优先级逻辑：
        
        # FIST (拳头) 优化：
        # 只要 4 根主手指都缩回去 (four_fingers_count == 0)，就算拳头
        if four_fingers_count == 0:
            gesture = "FIST"
            
        # PALM (手掌)：5 指全开
        elif four_fingers_count == 4 and fingers[0] == 1:
            gesture = "PALM"
            
        # FOUR (四指)：大拇指弯曲，其他4指伸出
        elif four_fingers_count == 4 and fingers[0] == 0:
            gesture = "FOUR"
            
        # TWO (剪刀手)：食指+中指伸出，无名指+小指弯曲
        elif fingers[1]==1 and fingers[2]==1 and fingers[3]==0 and fingers[4]==0:
            gesture = "TWO"
            
        # ONE (单指)：食指伸出，其他弯曲
        elif fingers[1]==1 and fingers[2]==0 and fingers[3]==0 and fingers[4]==0:
            gesture = "ONE"
            
        # THREE (三指/OK)：食指+中指+无名指
        elif fingers[1]==1 and fingers[2]==1 and fingers[3]==1 and fingers[4]==0:
            gesture = "THREE"
        
        # --- 放入防抖缓冲 ---
        self.gesture_buffer.add_gesture(gesture)
        stable_gesture = self.gesture_buffer.get_stable_gesture()
            
        return stable_gesture, hand_label

def run_video_mode():
    """视频流测试模式"""
    cap = cv2.VideoCapture(0)
    detector = HandDetector()
    p_time = 0

    print(">>> 视频模式已启动。请对着摄像头做手势...")
    print(">>> 按 'q' 键退出")

    while True:
        success, img = cap.read()
        if not success:
            print("无法读取摄像头")
            break

        # 1. 镜像翻转
        img = cv2.flip(img, 1)

        # 2. 检测手部
        img = detector.find_hands(img, draw=True)

        # 3. 识别手势
        gesture, hand_label = detector.get_gesture(img)

        # 4. 显示 FPS 和 手势结果
        c_time = time.time()
        fps = 1 / (c_time - p_time) if (c_time - p_time) > 0 else 0
        p_time = c_time
        
        # 绘制背景板让文字更清晰
        cv2.rectangle(img, (0,0), (250, 120), (0,0,0), cv2.FILLED)
        
        cv2.putText(img, f'FPS: {int(fps)}', (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        color = (0, 255, 0) if gesture != "NO HAND" and gesture != "UNKNOWN" else (0, 0, 255)
        
        cv2.putText(img, f'Gesture: {gesture}', (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        if hand_label != "None":
            cv2.putText(img, f'Hand: {hand_label}', (10, 110), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        cv2.imshow("Hand Gesture Test", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def run_image_mode(image_path):
    """单张图片测试模式"""
    if not os.path.exists(image_path):
        print(f"错误：找不到文件 {image_path}")
        return

    print(f"正在分析图片: {image_path} ...")
    img = cv2.imread(image_path)
    
    detector = HandDetector()
    
    img = detector.find_hands(img, draw=True)
    gesture, hand_label = detector.get_gesture(img)
    
    print(f"-----------------------------")
    print(f"检测结果: {gesture}")
    print(f"手部类型: {hand_label}")
    print(f"-----------------------------")

    cv2.putText(img, f'Gesture: {gesture}', (10, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(img, f'Hand: {hand_label}', (10, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    
    if not os.path.exists("./data/output"):
        os.makedirs("./data/output")
        
    img_name = os.path.basename(image_path).split('.')[0]
    output_path = "./data/output/" + f"gesture_result_{img_name}.jpg"
    cv2.imwrite(output_path, img)
    print(f"结果已保存至: {output_path}")

    cv2.imshow("Image Test Result", img)
    print("按任意键关闭图片窗口...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="video", choices=["video", "image"], 
                        help="运行模式: video 或 image")
    parser.add_argument("--path", type=str, help="图片路径 (仅 image 模式需要)")
    
    args = parser.parse_args()

    if args.mode == "video":
        run_video_mode()
    elif args.mode == "image":
        if args.path:
            run_image_mode(args.path)
        else:
            print("请提供图片路径。示例：python test.py --mode image --path my_hand.jpg")