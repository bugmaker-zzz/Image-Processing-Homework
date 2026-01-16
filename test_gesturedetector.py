import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import RunningMode
import time
import argparse
import os
import urllib.request

class HandDetector:
    def __init__(self):
        """
        初始化手部检测器
        """       
        # 确保模型文件存在
        model_path = self._ensure_model_file()
        
        # 使用 IMAGE 模式进行逐帧检测
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.5
        )
        
        self.hand_landmarker = vision.HandLandmarker.create_from_options(options)
        self.tip_ids = [4, 8, 12, 16, 20]   # 指尖的索引
        self.results = None
    
    def _ensure_model_file(self):
        """确保模型文件存在"""
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
        """绘制手部关键点和连接线"""
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
            cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
        
        for start, end in connections:
            if start < len(points) and end < len(points):
                cv2.line(img, points[start], points[end], (255, 0, 0), 2)

    def get_gesture(self, img):
        """核心逻辑：判断手指状态并返回手势名称"""
        if not self.results or not self.results.hand_landmarks:
            return "NO HAND"

        hand_landmarks = self.results.hand_landmarks[0]
        lm_list = []
        
        h, w, c = img.shape
        for id, landmark in enumerate(hand_landmarks):
            cx, cy = int(landmark.x * w), int(landmark.y * h)
            lm_list.append({'id': id, 'x': cx, 'y': cy})

        # --- 手指计数逻辑 ---
        fingers = []

        # 1. 拇指判断
        if lm_list[self.tip_ids[0]]['x'] < lm_list[self.tip_ids[0] - 1]['x']:
            fingers.append(1)
        else:
            fingers.append(0)

        # 2. 其他四指判断 (食指到小指)
        for id in range(1, 5):
            if lm_list[self.tip_ids[id]]['y'] < lm_list[self.tip_ids[id] - 2]['y']:
                fingers.append(1)
            else:
                fingers.append(0)

        total_fingers = fingers.count(1)

        # --- 手势定义 ---
        gesture = "UNKNOWN"
        
        if total_fingers == 0:
            gesture = "FIST"
        elif total_fingers == 5:
            gesture = "PALM"
        elif total_fingers == 2 and fingers[1]==1 and fingers[2]==1:
            gesture = "TWO"
        elif total_fingers == 1 and fingers[1]==1:
            gesture = "ONE"
        elif total_fingers == 3 and fingers[1]==1 and fingers[2]==1 and fingers[3]==1:
            gesture = "THREE"
        elif total_fingers == 4 and fingers[1]==1 and fingers[2]==1 and fingers[3]==1 and fingers[4]==1:
            gesture = "FOUR"
            
        return gesture

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

        # 1. 镜像翻转 (让画面像镜子一样，符合直觉)
        img = cv2.flip(img, 1)

        # 2. 检测手部
        img = detector.find_hands(img, draw=True)

        # 3. 识别手势
        gesture = detector.get_gesture(img)

        # 4. 显示 FPS 和 手势结果
        c_time = time.time()
        fps = 1 / (c_time - p_time) if (c_time - p_time) > 0 else 0
        p_time = c_time
        
        cv2.putText(img, f'FPS: {int(fps)}', (10, 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # 根据是否有手显示不同颜色
        color = (0, 255, 0) if gesture != "NO HAND" else (0, 0, 255)
        cv2.putText(img, f'Result: {gesture}', (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

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
    img = cv2.resize(img, (300, 400))
    
    detector = HandDetector()
    
    # 检测并画图
    img = detector.find_hands(img, draw=True)
    gesture = detector.get_gesture(img)
    
    print(f"-----------------------------")
    print(f"检测结果: {gesture}")
    print(f"-----------------------------")

    # 在图上写字
    cv2.putText(img, f'Gesture: {gesture}', (10, 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    img_name = os.path.basename(image_path).split('.')[0]
    output_path = "./data/output/" + f"gesture_result_{img_name}.jpg"
    cv2.imwrite(output_path, img)

    cv2.imshow("Image Test Result", img)
    print("按任意键关闭图片窗口...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # 命令行参数解析
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