"""
图像滤镜测试接口
支持实时查看所有滤镜效果、交互式测试和批量导出
"""

import cv2
import numpy as np
import os
import sys
from pathlib import Path

# 添加 utils 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))

from imagefilter import ImageFilter, FilterController, add_filter_info_overlay


class FilterTester:
    """图像滤镜测试工具"""
    
    def __init__(self, image_path: str = None):
        """
        初始化滤镜测试工具
        :param image_path: 图片路径（可选）
        """
        self.filter = ImageFilter()
        self.controller = FilterController()
        self.filters = self.filter.get_filter_names()
        self.current_filter_idx = 0
        self.image_path = image_path
        self.original_img = None
        self.current_img = None
        
        if image_path and os.path.exists(image_path):
            self.load_image(image_path)
        else:
            print("未指定图片路径，将创建演示图像")
            self.create_demo_image()
    
    def load_image(self, image_path: str):
        """加载图片"""
        img = cv2.imread(image_path)
        if img is None:
            print(f"无法读取图片: {image_path}")
            self.create_demo_image()
        else:
            # 调整图片大小以适应屏幕
            h, w = img.shape[:2]
            if w > 800 or h > 600:
                scale = min(800/w, 600/h)
                new_w = int(w * scale)
                new_h = int(h * scale)
                img = cv2.resize(img, (new_w, new_h))
            
            self.original_img = img
            self.image_path = image_path
            print(f"✓ 图片已加载: {image_path}")
            print(f"  分辨率: {img.shape[1]}x{img.shape[0]}")
    
    def create_demo_image(self):
        """创建演示图像"""
        img = np.zeros((400, 600, 3), dtype=np.uint8)
        
        # 绘制渐变背景
        for i in range(400):
            img[i, :] = [int(255 * i / 400), int(128 * i / 400), int(64 * i / 400)]
        
        # 绘制图形
        cv2.circle(img, (300, 200), 80, (0, 255, 255), -1)
        cv2.rectangle(img, (50, 100), (200, 300), (255, 0, 0), 3)
        cv2.putText(img, 'Filter Demo', (180, 380),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        self.original_img = img
        self.image_path = None
        print("✓ 演示图像已创建")
    
    def apply_current_filter(self):
        """应用当前选中的滤镜"""
        filter_name = self.filters[self.current_filter_idx]
        self.current_img = self.filter.apply_filter(self.original_img.copy(), filter_name)
        return self.current_img
    
    def get_filter_info_text(self) -> str:
        """获取滤镜信息文本"""
        filter_name = self.filters[self.current_filter_idx]
        total = len(self.filters)
        return f"[{self.current_filter_idx + 1}/{total}] {filter_name}"
    
    def show_interactive(self):
        """交互式查看滤镜"""
        print("\n" + "="*50)
        print("图像滤镜测试 - 交互模式")
        print("="*50)
        print(f"\n可用滤镜 ({len(self.filters)} 个):")
        for i, f in enumerate(self.filters, 1):
            print(f"  {i}. {f}")
        
        print("\n控制按键:")
        print("  A / D  : 上/下一个滤镜")
        print("  Space : 保存当前效果")
        print("  R     : 重置为原图")
        print("  Q/ESC : 退出\n")
        
        window_name = "image Filter Test"
        
        while True:
            self.apply_current_filter()
            
            # 添加信息叠加层
            display_img = self.current_img.copy()
            filter_name = self.filters[self.current_filter_idx]
            
            # 绘制标题
            h, w = display_img.shape[:2]
            overlay = display_img.copy()
            cv2.rectangle(overlay, (0, 0), (w, 60), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.3, display_img, 0.7, 0, display_img)
            
            info_text = self.get_filter_info_text()
            cv2.putText(display_img, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_img, "Arrow keys: switch | Space: save | R: reset | Q: quit",
                       (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            cv2.imshow(window_name, display_img)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:  # Q 或 ESC
                break
            elif key == ord('a') or key == ord('d'):  # A 向左, D 向右
                if key == ord('a'):
                    self.current_filter_idx = (self.current_filter_idx - 1) % len(self.filters)
                    print(f"← 上一个滤镜: {self.filters[self.current_filter_idx]}")
                else:
                    self.current_filter_idx = (self.current_filter_idx + 1) % len(self.filters)
                    print(f"→ 下一个滤镜: {self.filters[self.current_filter_idx]}")
            elif key == ord(' '):
                self.save_current()
            elif key == ord('r') or key == ord('R'):
                self.current_filter_idx = 0
                print("✓ 已重置为原图")
        
        cv2.destroyAllWindows()
    
    def show_all_filters_grid(self):
        """网格显示所有滤镜效果"""
        print("\n" + "="*50)
        print("图像滤镜测试 - 网格对比模式")
        print("="*50)
        
        # 计算网格布局
        n_filters = len(self.filters)
        cols = 3
        rows = (n_filters + cols - 1) // cols
        
        # 获取原始图片尺寸
        h, w = self.original_img.shape[:2]
        thumb_w = min(300, w)
        thumb_h = min(300, h)
        
        # 创建大画布
        canvas_w = thumb_w * cols + (cols - 1) * 10
        canvas_h = (thumb_h + 50) * rows + (rows - 1) * 10
        canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255
        
        print(f"\n生成网格图像... ({rows}x{cols})")
        
        for idx, filter_name in enumerate(self.filters):
            row = idx // cols
            col = idx % cols
            
            # 应用滤镜
            filtered = self.filter.apply_filter(self.original_img.copy(), filter_name)
            
            # 调整大小
            if len(filtered.shape) == 2:  # 灰度图
                filtered = cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR)
            filtered = cv2.resize(filtered, (thumb_w, thumb_h))
            
            # 添加滤镜名称
            overlay = filtered.copy()
            cv2.rectangle(overlay, (0, thumb_h - 35), (thumb_w, thumb_h), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.4, filtered, 0.6, 0, filtered)
            cv2.putText(filtered, filter_name, (5, thumb_h - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # 放置到画布
            y = row * (thumb_h + 50)
            x = col * (thumb_w + 10)
            canvas[y:y+thumb_h, x:x+thumb_w] = filtered
            
            print(f"  ✓ {filter_name}")
        
        cv2.imshow("所有滤镜效果 - 网格对比", canvas)
        print("\n按任意键关闭窗口...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # 保存网格
        output_path = "./data/output/filters_grid.jpg"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, canvas)
        print(f"✓ 网格已保存: {output_path}")
    
    def export_all_filters(self):
        """导出所有滤镜效果"""
        output_dir = "./data/output"
        os.makedirs(output_dir, exist_ok=True)
        
        if self.image_path:
            base_name = Path(self.image_path).stem
        else:
            base_name = "demo"
        
        print(f"\n" + "="*50)
        print("导出所有滤镜效果")
        print("="*50)
        print(f"\n正在处理 {len(self.filters)} 个滤镜...\n")
        
        for idx, filter_name in enumerate(self.filters, 1):
            filtered = self.filter.apply_filter(self.original_img.copy(), filter_name)
            
            # 转换为BGR（如果是灰度图）
            if len(filtered.shape) == 2:
                filtered = cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR)
            
            # 保存
            output_path = os.path.join(output_dir, f"{base_name}_{filter_name}.jpg")
            cv2.imwrite(output_path, filtered)
            print(f"  [{idx}/{len(self.filters)}] ✓ {filter_name:12} -> {Path(output_path).name}")
        
        print(f"\n✓ 所有滤镜已导出到 {output_dir}/")
    
    def save_current(self):
        """保存当前滤镜效果"""
        output_dir = "./data/output"
        os.makedirs(output_dir, exist_ok=True)
        
        if self.image_path:
            base_name = Path(self.image_path).stem
        else:
            base_name = "demo"
        
        filter_name = self.filters[self.current_filter_idx]
        output_path = os.path.join(output_dir, f"{base_name}_{filter_name}.jpg")
        
        img_to_save = self.current_img
        if len(img_to_save.shape) == 2:
            img_to_save = cv2.cvtColor(img_to_save, cv2.COLOR_GRAY2BGR)
        
        cv2.imwrite(output_path, img_to_save)
        print(f"✓ 已保存: {output_path}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="图像滤镜测试工具")
    parser.add_argument("--image", "-i", type=str, help="输入图片路径")
    parser.add_argument("--mode", "-m", type=str, default="interactive",
                       choices=["interactive", "grid", "export"],
                       help="测试模式: interactive(交互), grid(网格), export(导出)")
    
    args = parser.parse_args()
    
    print("\n" + "="*50)
    print("图像滤镜测试工具")
    print("="*50)
    
    tester = FilterTester(args.image)
    
    if args.mode == "interactive":
        tester.show_interactive()
    elif args.mode == "grid":
        tester.show_all_filters_grid()
    elif args.mode == "export":
        tester.export_all_filters()

if __name__ == "__main__":
    main()
