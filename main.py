"""
基于手势识别的智能图像滤镜切换系统
主入口程序

功能：
- 实时视频流中识别手势，根据手势切换图像滤镜，并支持拍照功能。
- 支持演示模式，展示所有滤镜效果。
- 支持图片处理模式，对指定图片应用滤镜。
手势映射关系:
- ONE (1) -> 直方图均衡化
- TWO (2) -> 流年特效
- THREE (3) -> 灰度特效
- FOUR (4) -> 怀旧特效
"""

import cv2
import sys
import os
import time
import argparse

# # 添加 utils 路径
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))

from utils.gesturedetector import HandDetector
from utils.imagefilter import ImageFilter, FilterController, add_filter_info_overlay


class GestureFilterSystem:
    """基于手势识别的图像滤镜系统"""
    
    def __init__(self):
        """初始化系统"""
        self.hand_detector = HandDetector()
        self.image_filter = ImageFilter()
        self.filter_controller = FilterController()
        self.current_gesture = "NO HAND"
        self.current_filter = "original"
        self.last_filter_gesture = None  # 上一次改变滤镜的手势
        self.p_time = 0
        self.frame_count = 0
        
        # 拍照功能相关
        self.gesture_sequence = []  # 记录手势序列
        self.sequence_timeout = 0.5  # 手势序列超时时间（秒）
        self.last_gesture_time = 0
        
        # 延迟拍照相关
        self.photo_triggered = False  # 是否触发了拍照
        self.photo_trigger_time = 0  # 触发拍照的时间
        self.photo_delay = 2.0  # 拍照延迟时间（秒）
        
        # 保存无UI的干净版本
        self.clean_frame = None  # 保存最后一帧的无UI版本
        
        print("="*60)
        print("基于手势识别的智能图像滤镜切换系统")
        print("="*60)
        print("\n手势映射关系:")
        print("  ONE   (1)  → 直方图均衡化")
        print("  TWO   (2)  → 流年特效")
        print("  THREE (3)  → 灰度特效")
        print("  FOUR  (4)  → 怀旧特效")
        print("\n其他手势:")
        print("  FIST  → 不改变滤镜")
        print("  PALM  → 不改变滤镜")
        print("  两只手同时出现 → 取消滤镜（变为原图）")
        print("\n特殊功能:")
        print("  PALM → FIST → 延迟2秒后拍照保存")
        print("\n控制按键:")
        print("  Q/ESC → 退出")
        print("="*60 + "\n")
    
    def process_frame(self, frame):
        """处理单帧图像"""
        # 1. 手势检测
        frame = self.hand_detector.find_hands(frame, draw=True)
        self.current_gesture = self.hand_detector.get_gesture(frame)
        
        # 2. 如果检测到两只手，取消滤镜，使用原图
        if self.current_gesture == "TWO_HANDS":
            self.current_filter = "original"
            self.last_filter_gesture = None
        # 3. 只有在做出数字手势（ONE/TWO/THREE/FOUR）时才改变滤镜
        elif self.current_gesture in ["ONE", "TWO", "THREE", "FOUR"]:
            # 检测到新的数字手势（与上一次改变的手势不同）
            if self.last_filter_gesture != self.current_gesture:
                self.last_filter_gesture = self.current_gesture
                self.current_filter = self.filter_controller.update_by_gesture(self.current_gesture)
        
        # 4. 检测拍照手势序列 (PALM -> FIST)
        should_trigger_photo = self._check_photo_gesture_sequence()
        if should_trigger_photo:
            self.photo_triggered = True
            self.photo_trigger_time = time.time()
        
        # 5. 检查延迟拍照是否到达
        should_take_photo = False
        if self.photo_triggered:
            elapsed = time.time() - self.photo_trigger_time
            if elapsed >= self.photo_delay:
                should_take_photo = True
                self.photo_triggered = False
        
        # 6. 应用滤镜
        processed_frame = self.image_filter.apply_filter(frame.copy(), self.current_filter)
        
        # 7. 保存无UI版本用于拍照
        self.clean_frame = processed_frame.copy()
        
        # 8. 添加信息叠加层（仅用于显示）
        display_frame = self._add_ui_overlay(processed_frame.copy())
        
        # 9. 如果拍照时间到达，则拍照（使用无UI版本）
        if should_take_photo:
            self.save_frame(self.clean_frame, is_photo=True)
        
        return display_frame
    
    def _check_photo_gesture_sequence(self):
        """检测拍照手势序列 (PALM -> FIST)"""
        current_time = time.time()
        
        # 清理过期的手势记录
        if self.gesture_sequence and (current_time - self.last_gesture_time) > self.sequence_timeout:
            self.gesture_sequence = []
        
        # 如果手势发生变化，记录新手势
        if self.current_gesture != "NO HAND":
            if not self.gesture_sequence or self.gesture_sequence[-1] != self.current_gesture:
                self.gesture_sequence.append(self.current_gesture)
                self.last_gesture_time = current_time
        
        # 检测 PALM -> FIST 序列
        if len(self.gesture_sequence) >= 2:
            if self.gesture_sequence[-2] == "PALM" and self.gesture_sequence[-1] == "FIST":
                self.gesture_sequence = []  # 清空序列
                return True
        
        return False
    
    def _add_ui_overlay(self, frame):
        """添加UI信息叠加层"""
        h, w = frame.shape[:2]
        
        # 上方信息框 - 手势和滤镜名称
        overlay = frame.copy()
        overlay_height = 90 if self.photo_triggered else 70
        cv2.rectangle(overlay, (0, 0), (w, overlay_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        # 手势信息
        gesture_color = (0, 255, 0) if self.current_gesture != "NO HAND" else (0, 0, 255)
        
        # 区分不同手势类型的显示
        if self.current_gesture == "TWO_HANDS":
            gesture_text = f'Gesture: {self.current_gesture} (Filter Cancelled)'
        elif self.current_gesture in ["ONE", "TWO", "THREE", "FOUR"]:
            gesture_text = f'Gesture: {self.current_gesture} (Filter Changed)'
        else:
            gesture_text = f'Gesture: {self.current_gesture}'
        
        cv2.putText(frame, gesture_text, (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, gesture_color, 2)
        
        # 滤镜名称 - 显示当前持久化的滤镜
        cv2.putText(frame, f'Current Filter: {self.current_filter}', (10, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # 拍照倒计时
        if self.photo_triggered:
            elapsed = time.time() - self.photo_trigger_time
            countdown = max(0, self.photo_delay - elapsed)
            countdown_text = f'Photo in {countdown:.1f}s'
            cv2.putText(frame, countdown_text, (10, 75),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        
        # FPS 显示
        c_time = time.time()
        fps = 1 / (c_time - self.p_time) if (c_time - self.p_time) > 0 else 0
        self.p_time = c_time
        cv2.putText(frame, f'FPS: {int(fps)}', (w - 150, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # 帧数显示
        self.frame_count += 1
        cv2.putText(frame, f'Frame: {self.frame_count}', (w - 150, 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # 下方帮助信息
        help_text = "Q:Quit | Space:Save | PALM+FIST:Photo(2s delay) | 2Hands:Cancel Filter"
        cv2.putText(frame, help_text, (15, h - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return frame
    
    def save_frame(self, frame, is_photo=False):
        """保存当前帧到文件"""
        output_dir = "./data/output"
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        prefix = "photo" if is_photo else "screenshot"
        filename = f"{output_dir}/{prefix}_{timestamp}_{self.current_filter}.jpg"
        
        cv2.imwrite(filename, frame)
        
        if is_photo:
            print(f"📷 拍照已保存: {filename}")
        else:
            print(f"✓ 已保存: {filename}")
    
    def run(self):
        """运行主程序"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("✗ 错误：无法打开摄像头")
            return
        
        print("✓ 摄像头已打开，系统开始运行...\n")
        
        window_name = "Gesture-based Image Filter System"
        
        try:
            while True:
                success, frame = cap.read()
                
                if not success:
                    print("✗ 无法读取摄像头")
                    break
                
                # 镜像翻转
                frame = cv2.flip(frame, 1)
                
                # 处理帧
                display_frame = self.process_frame(frame)
                
                # 显示
                cv2.imshow(window_name, display_frame)
                
                # 处理按键
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == 27:  # Q 或 ESC
                    print("\n✓ 用户退出程序")
                    break
                elif key == ord(' '):  # 空格键保存当前帧
                    self.save_frame(self.clean_frame, is_photo=False)
        
        except KeyboardInterrupt:
            print("\n✓ 已被中断")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("✓ 系统关闭")


def run_demo_mode():
    """演示模式：使用测试图片"""
    print("\n进入演示模式...")
    test_image_path = "./data/input/apple.png"
    
    if not os.path.exists(test_image_path):
        print(f"✗ 测试图片不存在: {test_image_path}")
        return
    
    frame = cv2.imread(test_image_path)
    
    # 创建系统
    system = GestureFilterSystem()
    
    # 显示每个滤镜效果
    filter_names = system.image_filter.get_filter_names()
    
    print(f"\n展示所有 {len(filter_names)} 种滤镜效果:\n")
    
    for filter_name in filter_names:
        # 应用滤镜
        result = system.image_filter.apply_filter(frame.copy(), filter_name)
        
        # 添加文字
        h, w = result.shape[:2]
        overlay = result.copy()
        cv2.rectangle(overlay, (0, 0), (w, 60), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, result, 0.7, 0, result)
        
        cv2.putText(result, f'Filter: {filter_name}', (15, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 显示
        cv2.imshow("Gesture-based Image Filter System - Demo Mode", result)
        
        print(f"  {filter_name:20} (按任意键查看下一个滤镜...)")
        cv2.waitKey(0)
    
    cv2.destroyAllWindows()
    print("\n✓ 演示完成")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="基于手势识别的智能图像滤镜切换系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument("--demo", action="store_true", 
                       help="运行演示模式 (显示所有滤镜效果)")
    parser.add_argument("--mode", default="realtime", choices=["realtime", "image"],
                       help="运行模式 (realtime: 实时视频, image: 图片处理)")
    parser.add_argument("--img_path", default="./data/input/apple.png", type=str,
                       help="图片路径(仅image模式需要)")
    
    args = parser.parse_args()
    
    if args.demo:
        run_demo_mode()
    else:
        if args.mode == "realtime":
            system = GestureFilterSystem()
            system.run()
        elif args.mode == "image":
            if not os.path.exists(args.img_path):
                print(f"✗ 图片不存在: {args.img_path}")
                return
            frame = cv2.imread(args.img_path)
            system = GestureFilterSystem()
            # original:原图；histogram_equalization:直方图均衡化；flowing_years:流年特效；grayscale:灰度特效；sepia:怀旧特效
            processed_frame = system.image_filter.apply_filter(frame.copy(), "grayscale")
            # 保存结果
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"./data/output/photo_{timestamp}_grayscale.jpg"
            cv2.imwrite(filename, processed_frame)
            cv2.imshow("Gesture-based Image Filter System - Image Mode", processed_frame)
            print("按任意键关闭图片窗口...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
