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
        self.tracked_peaks = None
        self.center_peak_idx = None
        self.valid_region = None
        self.tracked_point = None
        
        # 稳定性参数
        self.min_peaks = 3
        self.max_peak_shift = 10
        self.position_history = deque(maxlen=5)
        self.peak_heights = None
        self.peak_distances = None

        self.consecutive_failures = 0
        self.max_failures = 30

        self.input_video_path = None
    
    def reset_tracking_state(self, profile_smooth, profile_peaks, x, y):
        """重置追踪状态"""
        saved_points = self.points
        saved_fringes = self.total_fringes
        saved_wavelength = self.wavelength
        
        self.tracked_peaks = None
        self.center_peak_idx = None
        self.valid_region = None
        self.tracked_point = None
        self.peak_heights = None
        self.peak_distances = None
        self.position_history.clear()
        self.is_tracking = False
        self.consecutive_failures = 0
        
        self.points = saved_points
        self.total_fringes = saved_fringes
        self.wavelength = saved_wavelength
        
        return self.initialize_tracking(profile_smooth, profile_peaks, x, y)

    def load_video(self, video_path):
        """加载视频"""
        self.input_video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError("Cannot load video")
            
        ret, self.first_frame = self.cap.read()
        if not ret:
            raise ValueError("Cannot read first frame")
            
        self.first_frame = self.preprocess_frame(self.first_frame)
    
    def mouse_callback(self, event, x, y, flags, param):
        """处理鼠标事件并提供实时视觉反馈"""
        if event == cv2.EVENT_LBUTTONDOWN and len(self.points) < 2:
            self.points.append((x, y))
                
        self.display_image = self.first_frame.copy()
        if len(self.display_image.shape) == 2:
            self.display_image = cv2.cvtColor(self.display_image, cv2.COLOR_GRAY2BGR)
            
        for point in self.points:
            cv2.circle(self.display_image, point, 3, (0, 0, 255), -1)
        
        if len(self.points) == 1:
            cv2.line(self.display_image, self.points[0], (x, y), (0, 255, 0), 2)
        elif len(self.points) == 2:
            cv2.line(self.display_image, self.points[0], self.points[1], (0, 255, 0), 2)
        
        cv2.imshow('Fringe Tracking', self.display_image)

    def preprocess_frame(self, frame):
        """预处理图像帧"""
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.equalizeHist(frame)
        frame = cv2.GaussianBlur(frame, (3, 3), 0)
        return frame
    
    def get_output_video_name(self, input_video_path):
        """根据输入视频路径生成输出视频路径"""
        basename = os.path.basename(input_video_path)
        filename, _ = os.path.splitext(basename)
        output_filename = f"{filename}_技术版.mp4"
        return os.path.join(os.path.dirname(input_video_path), output_filename)
    
    def process_and_save_video(self, output_path):
        """处理视频并直接写入输出文件"""
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            if not out.isOpened():
                raise ValueError("Failed to create output video file")
        
        try:
            with tqdm(total=total_frames, desc="Processing video", 
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:
                
                while True:
                    ret, frame = self.cap.read()
                    if not ret:
                        break
                    
                    processed_frame = self.preprocess_frame(frame)
                    profile_smooth, profile_peaks, x, y = self.get_line_profile(processed_frame)
                    
                    if profile_smooth is not None:
                        self.update_tracking(profile_smooth, profile_peaks, x, y)
                    
                    frame_with_overlay = self.draw_overlay(frame)
                    out.write(frame_with_overlay)
                    pbar.update(1)
                    
        finally:
            out.release()
            
        print(f"\nOutput video saved as: {output_path}")
        
    def get_line_profile(self, image):
        """获取基准线上的强度分布"""
        if len(self.points) != 2:
            return None, None, None, None
            
        x1, y1 = self.points[0]
        x2, y2 = self.points[1]
        length = int(np.sqrt((x2-x1)**2 + (y2-y1)**2))
        x = np.linspace(x1, x2, length).astype(np.int32)
        y = np.linspace(y1, y2, length).astype(np.int32)
        
        profile = image[y, x]
        profile_smooth = savgol_filter(profile, window_length=7, polyorder=3)
        profile_peaks = savgol_filter(profile, window_length=self.window_length, polyorder=self.polyorder)
        
        return profile_smooth, profile_peaks, x, y
    
    def find_peaks_with_features(self, profile):
        """查找峰值并提取特征"""
        if profile is None:
            return None, None, None
            
        peaks, properties = find_peaks(profile, 
                                    distance=10,
                                    prominence=5,
                                    height=np.mean(profile) * 0.5)
        
        if len(peaks) >= self.min_peaks:
            heights = profile[peaks]
            distances = np.diff(peaks)
            return peaks, heights, distances
        return None, None, None
    
    def initialize_tracking(self, profile_smooth, profile_peaks, x, y):
        """初始化追踪"""
        peaks, heights, distances = self.find_peaks_with_features(profile_smooth)
        if peaks is None:
            return False
            
        self.wavelength = np.mean(distances) if distances is not None else None
        
        mid_idx = len(peaks) // 2
        if mid_idx >= 1 and mid_idx < len(peaks) - 1:
            self.tracked_peaks = peaks[mid_idx-1:mid_idx+2]
            self.center_peak_idx = 1
            self.peak_heights = heights[mid_idx-1:mid_idx+2]
            self.peak_distances = distances[mid_idx-1:mid_idx+1] if len(distances) >= 2 else None
            
            region_width = int(self.wavelength * 1) if self.wavelength else 50
            center = self.tracked_peaks[self.center_peak_idx]
            self.valid_region = (max(0, center - region_width), 
                               min(len(profile_smooth), center + region_width))
            
            self.tracked_point = (x[self.tracked_peaks[self.center_peak_idx]], 
                                y[self.tracked_peaks[self.center_peak_idx]])
            self.position_history.append(self.tracked_point)
            self.is_tracking = True
            return True
            
        return False
    
    def validate_peaks(self, peaks, heights, distances):
        """验证峰值的有效性"""
        if peaks is None or len(peaks) < self.min_peaks:
            return False
            
        if self.peak_heights is not None:
            peak_indices = np.searchsorted(peaks, self.tracked_peaks)
            if np.any(peak_indices >= len(heights)):
                return False
            selected_heights = heights[peak_indices]
            if np.max(np.abs(selected_heights - self.peak_heights)) > np.mean(self.peak_heights) * 0.3:
                return False
        
        if self.peak_distances is not None and distances is not None:
            peak_indices = np.searchsorted(peaks[:-1], self.tracked_peaks[:-1])
            if np.any(peak_indices >= len(distances)):
                return False
            selected_distances = distances[peak_indices]
            if np.max(np.abs(selected_distances - self.peak_distances)) > np.mean(self.peak_distances) * 0.3:
                return False
        
        return True
    
    def find_matching_peaks(self, peaks, profile_smooth, profile_peaks, x, y):
        """查找匹配的峰值组"""
        if len(peaks) < self.min_peaks:
            return None
            
        max_allowed_shift = self.wavelength * 0.5 if self.wavelength else float('inf')
        current_center = self.tracked_peaks[self.center_peak_idx]
        
        closest_peak_idx = None
        min_distance = float('inf')
        
        for i, peak in enumerate(peaks):
            distance = abs(peak - current_center)
            if distance < min_distance:
                min_distance = distance
                closest_peak_idx = i
        
        if min_distance <= max_allowed_shift:
            if closest_peak_idx is not None and closest_peak_idx > 0 and closest_peak_idx < len(peaks) - 1:
                peak_group = peaks[closest_peak_idx-1:closest_peak_idx+2]
                if self.valid_region[0] <= peak_group[1] <= self.valid_region[1]:
                    return peak_group
        
        return None
    
    def update_tracking(self, profile_smooth, profile_peaks, x, y):
        """更新追踪"""
        if profile_smooth is None:
            self.consecutive_failures += 1
            return
            
        peaks, heights, distances = self.find_peaks_with_features(profile_smooth)
        if peaks is None or not self.validate_peaks(peaks, heights, distances):
            self.consecutive_failures += 1
            if self.consecutive_failures >= self.max_failures:
                self.reset_tracking_state(profile_smooth, profile_peaks, x, y)
            return
            
        new_peaks = self.find_matching_peaks(peaks, profile_smooth, profile_peaks, x, y)
        if new_peaks is None:
            self.consecutive_failures += 1
            if self.consecutive_failures >= self.max_failures:
                self.reset_tracking_state(profile_smooth, profile_peaks, x, y)
            return
        
        self.consecutive_failures = 0
        
        old_center = self.tracked_peaks[self.center_peak_idx]
        new_center = new_peaks[self.center_peak_idx]
        movement = new_center - old_center
        
        if self.wavelength is not None and abs(movement) > self.wavelength * 0.5:
            self.total_fringes += 1 if movement > 0 else -1
        
        self.tracked_peaks = new_peaks
        new_point = (x[new_center], y[new_center])
        
        if len(self.position_history) >= self.position_history.maxlen:
            x_avg = int(np.mean([p[0] for p in self.position_history]))
            y_avg = int(np.mean([p[1] for p in self.position_history]))
            self.tracked_point = (x_avg, y_avg)
        else:
            self.tracked_point = new_point
            
        self.position_history.append(new_point)
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
            
            print("Click two points to select the reference line")
            first_frame = self.preprocess_frame(self.first_frame)
            self.display_image = cv2.cvtColor(first_frame, cv2.COLOR_GRAY2BGR)
            
            while len(self.points) < 2:
                cv2.imshow('Fringe Tracking', self.display_image)
                if cv2.waitKey(1) == ord('q'):
                    return
            
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            
            profile_smooth, profile_peaks, x, y = self.get_line_profile(first_frame)
            if not self.initialize_tracking(profile_smooth, profile_peaks, x, y):
                print("Failed to initialize tracking")
                return
                
            output_path = self.get_output_video_name(self.input_video_path)
            self.process_and_save_video(output_path)
            
            print(f"\nFinal fringe count: {self.total_fringes}")
            
        except Exception as e:
            print(f"\nError occurred: {str(e)}")
            raise
        finally:
            if hasattr(self, 'cap'):
                self.cap.release()
            
            for _ in range(4):
                cv2.destroyAllWindows()
                cv2.waitKey(1)

    def draw_overlay(self, frame):
        """在视频帧上叠加信息显示"""
        height, width = frame.shape[:2]
        font_scale = width / 600.0
        line_thickness = max(1, int(width/500))
        
        if len(self.points) == 2:
            cv2.line(frame, self.points[0], self.points[1], (0, 255, 0), line_thickness)
            
        if self.tracked_point is not None:
            circle_radius = max(3, int(width/200))
            cv2.circle(frame, self.tracked_point, circle_radius, (0, 0, 255), -1)
            cv2.circle(frame, self.tracked_point, circle_radius + 3, (255, 0, 0), line_thickness)
        
        text = f"Fringes: {self.total_fringes}"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]
        
        margin = int(width * 0.03)
        text_x = margin
        text_y = int(height * 0.15)
        
        padding = int(width * 0.02)
        bg_x1 = text_x - padding
        bg_y1 = text_y - text_size[1] - padding
        bg_x2 = text_x + text_size[0] + padding
        bg_y2 = text_y + padding
        
        cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
        cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 
                    font_scale, (0, 255, 0), line_thickness)
                    
        return frame

if __name__ == "__main__":
    tracker = RobustFringeTracker()
    tracker.load_video("示例视频.mov")
    tracker.run()
