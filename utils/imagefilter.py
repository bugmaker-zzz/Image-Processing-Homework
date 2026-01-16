import cv2
import numpy as np
from typing import Dict

class ImageFilter:
    """图像滤镜系统 - 4种滤镜效果"""
    
    def __init__(self):
        """初始化滤镜系统"""
        self.current_filter = "original"
        self.filters = {
            "original": self.filter_original,
            "histogram_equalization": self.filter_histogram_equalization,  # 手势 ONE (1)
            "flowing_years": self.filter_flowing_years,                   # 手势 TWO (2)
            "grayscale": self.filter_grayscale,                    # 手势 THREE (3)
            "sepia": self.filter_sepia,                           # 手势 FOUR (4)
        }
    
    def apply_filter(self, img: np.ndarray, filter_name: str) -> np.ndarray:
        """
        应用指定的滤镜
        :param img: 输入图像 (OpenCV BGR格式)
        :param filter_name: 滤镜名称
        :return: 处理后的图像
        """
        if filter_name in self.filters:
            self.current_filter = filter_name
            return self.filters[filter_name](img)
        else:
            return img
    
    def get_filter_names(self) -> list:
        """获取所有可用的滤镜名称"""
        return list(self.filters.keys())
    
    def get_current_filter(self) -> str:
        """获取当前激活的滤镜名称"""
        return self.current_filter
    
    # ======================== 滤镜实现 ========================
    
    def filter_original(self, img: np.ndarray) -> np.ndarray:
        """ 原图 """
        return img.copy()
    
    def filter_histogram_equalization(self, img: np.ndarray) -> np.ndarray:
        """ 直方图均衡化特效 (手势ONE) """
        rows, cols = img.shape[:2]
        dst = np.zeros((rows, cols, 3), dtype="uint8")
        (b, g, r) = cv2.split(img)

        #彩色图像均衡化
        bH = cv2.equalizeHist(b)
        gH = cv2.equalizeHist(g)
        rH = cv2.equalizeHist(r)
        dst = cv2.merge((bH, gH, rH))

        return dst
    
    def filter_flowing_years(self, img: np.ndarray) -> np.ndarray:
        """ 流年特效 (手势TWO) """
        # 分离通道
        b, g, r = cv2.split(img)
        
        # 对B通道进行平方根处理并乘以12
        b_float = b.astype(np.float32)
        b_processed = np.sqrt(b_float) * 12
        b_processed = np.clip(b_processed, 0, 255).astype(np.uint8)
        
        # 合并通道
        dst = cv2.merge((b_processed, g, r))
        return dst

    def filter_grayscale(self, img: np.ndarray) -> np.ndarray:
        """ 灰度特效 (手势THREE) """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    def filter_sepia(self, img: np.ndarray) -> np.ndarray:
        """ 怀旧特效 (手势FOUR) """
        # 转换为float32用于计算
        img_float = img.astype(np.float32)
        
        # 棕褐色变换矩阵 (BGR 格式)
        kernel = np.array([[0.272, 0.534, 0.131],
                          [0.349, 0.686, 0.168],
                          [0.393, 0.769, 0.189]], dtype=np.float32)
        
        # reshape 图像为 (高*宽, 3) 的矩阵
        h, w = img.shape[:2]
        img_reshaped = img_float.reshape(-1, 3)
        
        # 应用变换矩阵
        result = np.dot(img_reshaped, kernel.T)
        
        # reshape 回原来的形状
        result = result.reshape(h, w, 3)
        
        # 限制范围到 0-255
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        return result
    
    # ======================== 手势-滤镜映射 ========================
    
    @staticmethod
    def get_gesture_filter_map() -> Dict[str, str]:
        """
        获取手势到滤镜的映射关系
        :return: 映射字典
        """
        return {
            "original": "original",           # 默认/无滤镜
            "ONE": "histogram_equalization",   # 单指 -> 直方图均衡化
            "TWO": "flowing_years",          # 两指 -> 流年特效
            "THREE": "grayscale",        # 三指 -> 灰度特效
            "FOUR": "sepia",         # 四指 -> 怀旧特效
        }


class FilterController:
    """滤镜控制器 - 管理滤镜的应用和切换"""
    
    def __init__(self):
        """初始化滤镜控制器"""
        self.filter = ImageFilter()
        self.gesture_filter_map = ImageFilter.get_gesture_filter_map()
        self.last_gesture = "NO HAND"
    
    def update_by_gesture(self, gesture: str) -> str:
        """
        根据手势更新滤镜
        :param gesture: 识别的手势
        :return: 当前应用的滤镜名称
        """
        if gesture in self.gesture_filter_map:
            filter_name = self.gesture_filter_map[gesture]
            self.last_gesture = gesture
            return filter_name
        return self.filter.get_current_filter()
    
    def apply_gesture_filter(self, img: np.ndarray, gesture: str) -> np.ndarray:
        """
        根据手势应用对应的滤镜
        :param img: 输入图像
        :param gesture: 识别的手势
        :return: 处理后的图像
        """
        filter_name = self.update_by_gesture(gesture)
        return self.filter.apply_filter(img, filter_name)
    
    def get_info_text(self) -> tuple:
        """
        获取显示信息
        :return: (当前滤镜, 当前手势)
        """
        return self.filter.get_current_filter(), self.last_gesture


def add_filter_info_overlay(img: np.ndarray, filter_name: str, gesture: str,
                            position: str = "top") -> np.ndarray:
    """
    在图像上添加滤镜信息叠加层
    :param img: 输入图像
    :param filter_name: 滤镜名称
    :param gesture: 手势名称
    :param position: 文字位置 ("top" 或 "bottom")
    :return: 添加叠加层的图像
    """
    h, w = img.shape[:2]
    
    # 确定文本位置
    if position == "top":
        y_offset = 30
    else:
        y_offset = h - 30
    
    # 绘制半透明背景框
    overlay = img.copy()
    cv2.rectangle(overlay, (5, y_offset - 25), (350, y_offset + 10),
                  (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)
    
    # 绘制文本
    cv2.putText(img, f'Filter: {filter_name}', (15, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(img, f'Gesture: {gesture}', (15, y_offset + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    return img
