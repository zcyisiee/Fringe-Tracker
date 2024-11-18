import cv2
import numpy as np
from scipy.signal import savgol_filter, find_peaks
from collections import deque
import sys
from tqdm import tqdm
import os

class RobustFringeTracker:
    def __init__(self):
        # 基本参数
        self.points = []
        self.window_length = 11
        self.polyorder = 3
        self.wavelength = None
        self.total_fringes = 0
        self.is_tracking = False
        
        # 视频和显示相关
        self.cap = None
        self.first_frame = None
        self.display_image = None
        
        # 峰值追踪参数
        self.tracked_peaks = None  # 追踪的峰值组
        self.center_peak_idx = None  # 中心峰在峰值组中的索引
        self.valid_region = None  # 有效追踪区域
        self.tracked_point = None  # 显示用的追踪点
        
        # 稳定性参数
        self.min_peaks = 3  # 最小峰值数量
        self.max_peak_shift = 10  # 最大允许峰值偏移
        self.position_history = deque(maxlen=5)
        self.peak_heights = None  # 记录峰值高度特征
        self.peak_distances = None  # 记录峰值间距特征

        # 匹配失败阈值上限
        self.consecutive_failures = 0
        self.max_failures = 30  # 最大允许连续失配次数

        self.video_writer = None

        self.input_video_path = None
    
    def reset_tracking_state(self, profile_smooth, profile_peaks, x, y):
        """重置追踪状态"""
        print("Resetting tracking state due to consecutive failures")
        # 保留重要状态
        saved_points = self.points
        saved_fringes = self.total_fringes
        saved_wavelength = self.wavelength
        
        # 重置追踪相关状态
        self.tracked_peaks = None
        self.center_peak_idx = None
        self.valid_region = None
        self.tracked_point = None
        self.peak_heights = None
        self.peak_distances = None
        self.position_history.clear()
        self.is_tracking = False
        self.consecutive_failures = 0
        
        # 恢复保存的状态
        self.points = saved_points
        self.total_fringes = saved_fringes
        self.wavelength = saved_wavelength
        
        # 尝试重新初始化追踪
        if self.initialize_tracking(profile_smooth, profile_peaks, x, y):
            print("Successfully reinitialized tracking")
            return True
        else:
            print("Failed to reinitialize tracking")
            return False

    def load_video(self, video_path):
        """加载视频"""
        self.input_video_path = video_path  # 保存输入视频路径
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError("Cannot load video")
            
        # 读取第一帧
        ret, self.first_frame = self.cap.read()
        if not ret:
            raise ValueError("Cannot read first frame")
            
        # 预处理第一帧
        self.first_frame = self.preprocess_frame(self.first_frame)
    
    def mouse_callback(self, event, x, y, flags, param):
        """处理鼠标事件并提供实时视觉反馈"""
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.points) < 2:
                self.points.append((x, y))
                
        # 每次创建新的显示图像
        self.display_image = self.first_frame.copy()
        if len(self.display_image.shape) == 2:
            self.display_image = cv2.cvtColor(self.display_image, cv2.COLOR_GRAY2BGR)
            
        # 显示点击的点
        for point in self.points:
            cv2.circle(self.display_image, point, 3, (0, 0, 255), -1)
        
        # 如果有第一个点，显示当前线段
        if len(self.points) == 1:
            cv2.line(self.display_image, self.points[0], (x, y), (0, 255, 0), 2)
        elif len(self.points) == 2:
            cv2.line(self.display_image, self.points[0], self.points[1], (0, 255, 0), 2)
        
        # 更新显示
        cv2.imshow('Fringe Tracking', self.display_image)


    def preprocess_frame(self, frame):
        """预处理图像帧"""
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.equalizeHist(frame)
        
        # 轻微的高斯滤波去噪
        frame = cv2.GaussianBlur(frame, (3, 3), 0)
        return frame

    def get_line_profile(self, image):
        """获取基准线上的强度分布"""
        if len(self.points) != 2:
            return None, None, None
            
        x1, y1 = self.points[0]
        x2, y2 = self.points[1]
        length = int(np.sqrt((x2-x1)**2 + (y2-y1)**2))
        x = np.linspace(x1, x2, length).astype(np.int32)
        y = np.linspace(y1, y2, length).astype(np.int32)
        
        # 获取强度分布
        profile = image[y, x]
        
        # 两次滤波：轻度滤波用于峰值检测，强滤波用于精确定位
        profile_smooth = savgol_filter(profile, window_length=7, polyorder=3)  # 轻度滤波
        profile_peaks = savgol_filter(profile, window_length=self.window_length, polyorder=self.polyorder)  # 强滤波
        
        return profile_smooth, profile_peaks, x, y
    
    def find_peaks_with_features(self, profile):
        """查找峰值并提取特征"""
        if profile is None:
            return None, None, None
            
        # 使用更宽松的参数查找峰值
        peaks, properties = find_peaks(profile, 
                                    distance=10,  # 降低最小距离要求
                                    prominence=5,  # 降低突出度要求
                                    height=np.mean(profile) * 0.5)  # 添加高度阈值
        
        if len(peaks) >= self.min_peaks:
            # 提取峰值特征
            heights = profile[peaks]
            distances = np.diff(peaks)
            return peaks, heights, distances
        return None, None, None
    
    def initialize_tracking(self, profile_smooth, profile_peaks, x, y):
        """初始化追踪"""
        peaks, heights, distances = self.find_peaks_with_features(profile_smooth)
        if peaks is None:
            return False
            
        # 估计波长
        self.wavelength = np.mean(distances) if distances is not None else None
        
        # 选择中间的3个峰值作为追踪组
        mid_idx = len(peaks) // 2
        if mid_idx >= 1 and mid_idx < len(peaks) - 1:
            self.tracked_peaks = peaks[mid_idx-1:mid_idx+2]
            self.center_peak_idx = 1  # 中间峰的索引
            self.peak_heights = heights[mid_idx-1:mid_idx+2]
            self.peak_distances = distances[mid_idx-1:mid_idx+1] if len(distances) >= 2 else None
            
            # 设置有效追踪区域
            region_width = int(self.wavelength * 1) if self.wavelength else 50
            center = self.tracked_peaks[self.center_peak_idx]
            self.valid_region = (max(0, center - region_width), 
                               min(len(profile_smooth), center + region_width))
            
            # 初始化追踪点
            self.tracked_point = (x[self.tracked_peaks[self.center_peak_idx]], 
                                y[self.tracked_peaks[self.center_peak_idx]])
            self.position_history.append(self.tracked_point)
            self.is_tracking = True
            return True
            
        return False
    def get_output_video_name(self, input_video_path):
        """根据输入视频路径生成输出视频路径"""
        # 分离文件名和扩展名
        basename = os.path.basename(input_video_path)
        filename, _ = os.path.splitext(basename)  # 忽略原始扩展名
        # 生成新文件名，强制使用.mp4扩展名
        output_filename = f"{filename}_技术版.mp4"
        # 返回与输入视频相同目录下的新文件路径
        return os.path.join(os.path.dirname(input_video_path), output_filename)

    
    def process_and_save_video(self, output_path):
        """处理视频并直接写入输出文件,避免存储所有帧"""
        # 获取视频参数
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 使用H.264编码器
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            print("Failed to create output video file. Trying alternative codec...")
            # 如果H.264编码器不可用，尝试使用平台默认编码器
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            if not out.isOpened():
                raise ValueError("Failed to create output video file")
        
        try:
            # 使用tqdm创建进度条
            with tqdm(total=total_frames, desc="Processing video", 
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:
                
                while True:
                    ret, frame = self.cap.read()
                    if not ret:
                        break
                    
                    # 处理帧
                    processed_frame = self.preprocess_frame(frame)
                    profile_smooth, profile_peaks, x, y = self.get_line_profile(processed_frame)
                    
                    if profile_smooth is not None:
                        self.update_tracking(profile_smooth, profile_peaks, x, y)
                    
                    # 绘制追踪信息
                    frame_with_overlay = self.draw_overlay(frame)
                    
                    # 写入处理后的帧
                    out.write(frame_with_overlay)
                    
                    # 更新进度条
                    pbar.update(1)
                    
                    # 主动释放不需要的内存
                    del processed_frame
                    del profile_smooth
                    del profile_peaks
                    
        except Exception as e:
            print(f"\nError during processing: {str(e)}")
            raise
        finally:
            # 确保资源被释放
            out.release()
            
        print(f"\nOutput video saved as: {output_path}")
        
    def validate_peaks(self, peaks, heights, distances):
        """验证峰值的有效性"""
        if peaks is None or len(peaks) < self.min_peaks:
            return False
            
        if self.peak_heights is not None:
            # Find corresponding peak heights by matching indices
            peak_indices = np.searchsorted(peaks, self.tracked_peaks)
            if np.any(peak_indices >= len(heights)):  # Check bounds
                return False
            selected_heights = heights[peak_indices]
            # Now compare heights of corresponding peaks
            height_diff = np.abs(selected_heights - self.peak_heights)
            if np.max(height_diff) > np.mean(self.peak_heights) * 0.3:  # 允许30%的变化
                return False
        
        # 检查峰值间距的一致性
        if self.peak_distances is not None and distances is not None:
            peak_indices = np.searchsorted(peaks[:-1], self.tracked_peaks[:-1])
            if np.any(peak_indices >= len(distances)):  # Check bounds
                return False
            selected_distances = distances[peak_indices]
            dist_diff = np.abs(selected_distances - self.peak_distances)
            if np.max(dist_diff) > np.mean(self.peak_distances) * 0.3:  # 允许30%的变化
                return False
        
        return True
    
    def find_matching_peaks(self, peaks, profile_smooth, profile_peaks, x, y):
        """查找匹配的峰值组"""
        if len(peaks) < self.min_peaks:
            return None
            
        best_match = None
        min_diff = float('inf')
        
        # 降低最大允许移动距离为半个波长，使切换更灵敏
        max_allowed_shift = self.wavelength * 0.5 if self.wavelength is not None else float('inf')
        current_center = self.tracked_peaks[self.center_peak_idx]
        
        # 首先寻找距离当前中心峰最近的峰
        closest_peak_idx = None
        min_distance = float('inf')
        
        for i, peak in enumerate(peaks):
            distance = abs(peak - current_center)
            if distance < min_distance:
                min_distance = distance
                closest_peak_idx = i
        
        # 如果找到的最近峰距离在允许范围内，构建峰值组
        if min_distance <= max_allowed_shift:
            if closest_peak_idx is not None and closest_peak_idx > 0 and closest_peak_idx < len(peaks) - 1:
                peak_group = peaks[closest_peak_idx-1:closest_peak_idx+2]
                if self.valid_region[0] <= peak_group[1] <= self.valid_region[1]:
                    return peak_group
        
        # 在允许范围内查找所有可能的峰值组
        valid_matches = []
        for i in range(len(peaks) - 2):
            peak_group = peaks[i:i+3]
            center_peak = peak_group[1]
            if (self.valid_region[0] <= center_peak <= self.valid_region[1]):
                # 计算与当前追踪组的差异
                position_diff = abs(center_peak - current_center)
                if position_diff <= max_allowed_shift:
                    valid_matches.append((position_diff, peak_group))
        
        # 如果有有效匹配，选择差异最小的
        if valid_matches:
            valid_matches.sort(key=lambda x: x[0])
            return valid_matches[0][1]
        
        # 如果没有在阈值范围内的匹配，寻找最近的可用峰值组
        for i in range(len(peaks) - 2):
            peak_group = peaks[i:i+3]
            position_diff = np.sum(np.abs(peak_group - self.tracked_peaks))
            
            center_peak = peak_group[1]
            if (self.valid_region[0] <= center_peak <= self.valid_region[1] and 
                position_diff < min_diff):
                min_diff = position_diff
                best_match = peak_group
                
        return best_match
    
    def update_tracking(self, profile_smooth, profile_peaks, x, y):
        """更新追踪"""
        if profile_smooth is None:  # 处理黑屏情况
            print("Warning: Invalid profile, maintaining position")
            self.consecutive_failures += 1
            return
            
        peaks, heights, distances = self.find_peaks_with_features(profile_smooth)
        if peaks is None:
            self.consecutive_failures += 1
            return
            
        # 验证峰值有效性
        if not self.validate_peaks(peaks, heights, distances):
            self.consecutive_failures += 1
            
            # 检查是否超过最大失配次数
            if self.consecutive_failures >= self.max_failures:
                self.reset_tracking_state(profile_smooth, profile_peaks, x, y)
            return
            
        # 查找最佳匹配的峰值组
        new_peaks = self.find_matching_peaks(peaks, profile_smooth, profile_peaks, x, y)
        
        # 如果没有找到有效的峰值组，增加失配计数
        if new_peaks is None:
            print("Warning: No valid peaks found in current frame, maintaining position")
            self.consecutive_failures += 1
            
            # 检查是否超过最大失配次数
            if self.consecutive_failures >= self.max_failures:
                self.reset_tracking_state(profile_smooth, profile_peaks, x, y)
            return
        
        # 找到有效匹配，重置失配计数
        self.consecutive_failures = 0
        
        # 计算移动距离
        old_center = self.tracked_peaks[self.center_peak_idx]
        new_center = new_peaks[self.center_peak_idx]
        movement = new_center - old_center
        
        # 更新条纹计数 - 使用更敏感的阈值
        if self.wavelength is not None:
            if abs(movement) > self.wavelength * 0.5:  # 降低移动距离阈值
                if movement > 0:
                    self.total_fringes += 1
                else:
                    self.total_fringes -= 1
        
        # 更新追踪信息
        self.tracked_peaks = new_peaks
        new_point = (x[new_center], y[new_center])
        
        # 平滑更新显示位置
        if len(self.position_history) >= self.position_history.maxlen:
            x_avg = int(np.mean([p[0] for p in self.position_history]))
            y_avg = int(np.mean([p[1] for p in self.position_history]))
            self.tracked_point = (x_avg, y_avg)
        else:
            self.tracked_point = new_point
            
        self.position_history.append(new_point)
        
        # 更新特征
        self.peak_heights = heights[np.searchsorted(peaks, new_peaks)]
        if len(distances) >= 2:
            self.peak_distances = distances[np.searchsorted(peaks[:-1], new_peaks[:-1])]
    
    def run(self):
        """运行程序"""
        try:
            if not hasattr(self, 'cap'):
                return
                    
            cv2.namedWindow('Fringe Tracking')
            cv2.setMouseCallback('Fringe Tracking', self.mouse_callback)
            
            # 等待用户选择基准线
            print("Click two points to select the reference line")
            first_frame = self.preprocess_frame(self.first_frame)
            self.display_image = cv2.cvtColor(first_frame, cv2.COLOR_GRAY2BGR)
            
            while len(self.points) < 2:
                cv2.imshow('Fringe Tracking', self.display_image)
                if cv2.waitKey(1) == ord('q'):
                    return
            
            # 关闭选择窗口
            cv2.destroyAllWindows()
            cv2.waitKey(1)  # 确保窗口正确关闭
            
            # 初始化追踪
            profile_smooth, profile_peaks, x, y = self.get_line_profile(first_frame)
            if not self.initialize_tracking(profile_smooth, profile_peaks, x, y):
                print("Failed to initialize tracking")
                return
                
            # 处理并保存视频
            output_path = self.get_output_video_name(self.input_video_path)  # 使用保存的路径
            self.process_and_save_video(output_path)
            
            # 输出最终条纹计数
            print(f"\nFinal fringe count: {self.total_fringes}")
            
        except Exception as e:
            print(f"\nError occurred: {str(e)}")
            raise
        finally:
            # 确保资源被释放
            if hasattr(self, 'cap'):
                self.cap.release()
            
            # 确保所有窗口都被关闭
            for _ in range(4):
                cv2.destroyAllWindows()
                cv2.waitKey(1)

    def draw_overlay(self, frame):
        """在视频帧上叠加信息显示，增强视觉效果"""
        height, width = frame.shape[:2]
        
        # 计算字体大小（基于帧宽度）
        font_scale = width / 1000.0  # 基准：1000像素宽时字体大小为1
        line_thickness = max(1, int(width/500))  # 线条粗细
        
        # 首先绘制参考线
        if len(self.points) == 2:
            # 确保完整绘制参考线
            cv2.line(frame, self.points[0], self.points[1], 
                    (0, 255, 0), line_thickness)
            
        # 绘制跟踪点
        if self.tracked_point is not None:
            # 计算跟踪点标记的大小
            circle_radius = max(3, int(width/200))
            
            # 绘制跟踪点及其外圈
            cv2.circle(frame, self.tracked_point, circle_radius, (0, 0, 255), -1)
            cv2.circle(frame, self.tracked_point, circle_radius + 3, 
                    (255, 0, 0), line_thickness)
        
        # 绘制条纹计数文本
        text = f"Fringes: {self.total_fringes}"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]
        
        # 计算字体大小 - 增大字体
        font_scale = width / 600.0  # 调整基准值使字体更大
        
        # 计算文本位置 - 左上角
        margin = int(width * 0.03)  # 边距为宽度的3%
        text_x = margin
        text_y = int(height * 0.15)  # 距顶部15%位置，给更大空间
        
        # 计算文本背景框的尺寸和位置
        padding = int(width * 0.02)  # 内边距
        bg_x1 = text_x - padding
        bg_y1 = text_y - text_size[1] - padding
        bg_x2 = text_x + text_size[0] + padding
        bg_y2 = text_y + padding
        
        # 绘制半透明背景
        overlay = frame.copy()
        cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), 
                    (0, 0, 0), -1)
        
        # 添加圆角效果
        corner_radius = int(padding * 0.8)
        cv2.circle(overlay, (bg_x1 + corner_radius, bg_y1 + corner_radius), 
                corner_radius, (0, 0, 0), -1)
        cv2.circle(overlay, (bg_x2 - corner_radius, bg_y1 + corner_radius), 
                corner_radius, (0, 0, 0), -1)
        cv2.circle(overlay, (bg_x1 + corner_radius, bg_y2 - corner_radius), 
                corner_radius, (0, 0, 0), -1)
        cv2.circle(overlay, (bg_x2 - corner_radius, bg_y2 - corner_radius), 
                corner_radius, (0, 0, 0), -1)
        
        # 应用透明度
        alpha = 0.7
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # 绘制文本边框
        border_color = (0, 200, 0)  # 深绿色边框
        cv2.putText(frame, text, (text_x, text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, 
                    (0, 0, 0), line_thickness + 2)  # 外边框
        
        # 绘制文本
        cv2.putText(frame, text, (text_x, text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, 
                    (0, 255, 0), line_thickness)  # 内部文字
                    
        return frame
    # 使用示例
if __name__ == "__main__":
    tracker = RobustFringeTracker()
    tracker.load_video("示例视频.mov")  # 替换为你的视频文件路径
    tracker.run()