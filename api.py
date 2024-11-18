# import cv2
# import numpy as np
# from scipy.signal import savgol_filter, find_peaks
# from collections import deque
# from fastapi import FastAPI, UploadFile, File, Form
# from fastapi.responses import FileResponse, JSONResponse
# from fastapi.middleware.cors import CORSMiddleware
# import os
# import tempfile
# import shutil

# app = FastAPI()

# # 配置CORS
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # 存储当前处理的视频路径和进度
# current_video_path = None
# processing_progress = {"total_frames": 0, "current_frame": 0, "is_processing": False}

# class RobustFringeTracker:
#     def __init__(self):
#         # 基本参数
#         self.points = []
#         self.window_length = 11
#         self.polyorder = 3
#         self.wavelength = None
#         self.total_fringes = 0
#         self.is_tracking = False
        
#         # 视频和显示相关
#         self.cap = None
#         self.first_frame = None
        
#         # 峰值追踪参数
#         self.tracked_peaks = None
#         self.center_peak_idx = None
#         self.valid_region = None
#         self.tracked_point = None
        
#         # 稳定性参数
#         self.min_peaks = 3
#         self.max_peak_shift = 10
#         self.position_history = deque(maxlen=5)
#         self.peak_heights = None
#         self.peak_distances = None

#         self.consecutive_failures = 0
#         self.max_failures = 30

#         self.video_writer = None
    
#     def reset_tracking_state(self, profile_smooth, profile_peaks, x, y):
#         saved_points = self.points
#         saved_fringes = self.total_fringes
#         saved_wavelength = self.wavelength
        
#         self.tracked_peaks = None
#         self.center_peak_idx = None
#         self.valid_region = None
#         self.tracked_point = None
#         self.peak_heights = None
#         self.peak_distances = None
#         self.position_history.clear()
#         self.is_tracking = False
#         self.consecutive_failures = 0
        
#         self.points = saved_points
#         self.total_fringes = saved_fringes
#         self.wavelength = saved_wavelength
        
#         return self.initialize_tracking(profile_smooth, profile_peaks, x, y)

#     def load_video(self, video_path, auto_mode=True, points=None):
#         """加载视频并设置基准线"""
#         self.cap = cv2.VideoCapture(video_path)
#         if not self.cap.isOpened():
#             raise ValueError("Cannot load video")
            
#         ret, self.first_frame = self.cap.read()
#         if not ret:
#             raise ValueError("Cannot read first frame")
            
#         self.first_frame = self.preprocess_frame(self.first_frame)
        
#         height, width = self.first_frame.shape
#         if auto_mode:
#             # 自动模式：选择一条竖线
#             self.points = [(int(width * 0.5), int(height * 0.2)), 
#                           (int(width * 0.5), int(height * 0.8))]
#         elif points:
#             # 手动模式：使用用户提供的点
#             self.points = points

#     def preprocess_frame(self, frame):
#         if len(frame.shape) == 3:
#             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         frame = cv2.equalizeHist(frame)
#         frame = cv2.GaussianBlur(frame, (3, 3), 0)
#         return frame
    
#     def get_line_profile(self, image):
#         if len(self.points) != 2:
#             return None, None, None
            
#         x1, y1 = self.points[0]
#         x2, y2 = self.points[1]
#         length = int(np.sqrt((x2-x1)**2 + (y2-y1)**2))
#         x = np.linspace(x1, x2, length).astype(np.int32)
#         y = np.linspace(y1, y2, length).astype(np.int32)
        
#         profile = image[y, x]
#         profile_smooth = savgol_filter(profile, window_length=7, polyorder=3)
#         profile_peaks = savgol_filter(profile, window_length=self.window_length, polyorder=self.polyorder)
        
#         return profile_smooth, profile_peaks, x, y
    
#     def find_peaks_with_features(self, profile):
#         if profile is None:
#             return None, None, None
            
#         peaks, properties = find_peaks(profile, 
#                                     distance=10,
#                                     prominence=5,
#                                     height=np.mean(profile) * 0.5)
        
#         if len(peaks) >= self.min_peaks:
#             heights = profile[peaks]
#             distances = np.diff(peaks)
#             return peaks, heights, distances
#         return None, None, None
    
#     def initialize_tracking(self, profile_smooth, profile_peaks, x, y):
#         peaks, heights, distances = self.find_peaks_with_features(profile_smooth)
#         if peaks is None:
#             return False
            
#         self.wavelength = np.mean(distances) if distances is not None else None
        
#         mid_idx = len(peaks) // 2
#         if mid_idx >= 1 and mid_idx < len(peaks) - 1:
#             self.tracked_peaks = peaks[mid_idx-1:mid_idx+2]
#             self.center_peak_idx = 1
#             self.peak_heights = heights[mid_idx-1:mid_idx+2]
#             self.peak_distances = distances[mid_idx-1:mid_idx+1] if len(distances) >= 2 else None
            
#             region_width = int(self.wavelength * 1) if self.wavelength else 50
#             center = self.tracked_peaks[self.center_peak_idx]
#             self.valid_region = (max(0, center - region_width), 
#                                min(len(profile_smooth), center + region_width))
            
#             self.tracked_point = (x[self.tracked_peaks[self.center_peak_idx]], 
#                                 y[self.tracked_peaks[self.center_peak_idx]])
#             self.position_history.append(self.tracked_point)
#             self.is_tracking = True
#             return True
            
#         return False

#     def validate_peaks(self, peaks, heights, distances):
#         if peaks is None or len(peaks) < self.min_peaks:
#             return False
            
#         if self.peak_heights is not None:
#             peak_indices = np.searchsorted(peaks, self.tracked_peaks)
#             if np.any(peak_indices >= len(heights)):
#                 return False
#             selected_heights = heights[peak_indices]
#             height_diff = np.abs(selected_heights - self.peak_heights)
#             if np.max(height_diff) > np.mean(self.peak_heights) * 0.3:
#                 return False
        
#         if self.peak_distances is not None and distances is not None:
#             peak_indices = np.searchsorted(peaks[:-1], self.tracked_peaks[:-1])
#             if np.any(peak_indices >= len(distances)):
#                 return False
#             selected_distances = distances[peak_indices]
#             dist_diff = np.abs(selected_distances - self.peak_distances)
#             if np.max(dist_diff) > np.mean(self.peak_distances) * 0.3:
#                 return False
        
#         return True
    
#     def find_matching_peaks(self, peaks, profile_smooth, profile_peaks, x, y):
#         if len(peaks) < self.min_peaks:
#             return None
            
#         best_match = None
#         min_diff = float('inf')
        
#         max_allowed_shift = self.wavelength * 0.5 if self.wavelength is not None else float('inf')
#         current_center = self.tracked_peaks[self.center_peak_idx]
        
#         closest_peak_idx = None
#         min_distance = float('inf')
        
#         for i, peak in enumerate(peaks):
#             distance = abs(peak - current_center)
#             if distance < min_distance:
#                 min_distance = distance
#                 closest_peak_idx = i
        
#         if min_distance <= max_allowed_shift:
#             if closest_peak_idx is not None and closest_peak_idx > 0 and closest_peak_idx < len(peaks) - 1:
#                 peak_group = peaks[closest_peak_idx-1:closest_peak_idx+2]
#                 if self.valid_region[0] <= peak_group[1] <= self.valid_region[1]:
#                     return peak_group
        
#         valid_matches = []
#         for i in range(len(peaks) - 2):
#             peak_group = peaks[i:i+3]
#             center_peak = peak_group[1]
#             if (self.valid_region[0] <= center_peak <= self.valid_region[1]):
#                 position_diff = abs(center_peak - current_center)
#                 if position_diff <= max_allowed_shift:
#                     valid_matches.append((position_diff, peak_group))
        
#         if valid_matches:
#             valid_matches.sort(key=lambda x: x[0])
#             return valid_matches[0][1]
        
#         for i in range(len(peaks) - 2):
#             peak_group = peaks[i:i+3]
#             position_diff = np.sum(np.abs(peak_group - self.tracked_peaks))
            
#             center_peak = peak_group[1]
#             if (self.valid_region[0] <= center_peak <= self.valid_region[1] and 
#                 position_diff < min_diff):
#                 min_diff = position_diff
#                 best_match = peak_group
                
#         return best_match
    
#     def update_tracking(self, profile_smooth, profile_peaks, x, y):
#         if profile_smooth is None:
#             self.consecutive_failures += 1
#             return
            
#         peaks, heights, distances = self.find_peaks_with_features(profile_smooth)
#         if peaks is None:
#             self.consecutive_failures += 1
#             return
            
#         if not self.validate_peaks(peaks, heights, distances):
#             self.consecutive_failures += 1
            
#             if self.consecutive_failures >= self.max_failures:
#                 self.reset_tracking_state(profile_smooth, profile_peaks, x, y)
#             return
            
#         new_peaks = self.find_matching_peaks(peaks, profile_smooth, profile_peaks, x, y)
        
#         if new_peaks is None:
#             self.consecutive_failures += 1
            
#             if self.consecutive_failures >= self.max_failures:
#                 self.reset_tracking_state(profile_smooth, profile_peaks, x, y)
#             return
        
#         self.consecutive_failures = 0
        
#         old_center = self.tracked_peaks[self.center_peak_idx]
#         new_center = new_peaks[self.center_peak_idx]
#         movement = new_center - old_center
        
#         if self.wavelength is not None:
#             if abs(movement) > self.wavelength * 0.5:
#                 if movement > 0:
#                     self.total_fringes += 1
#                 else:
#                     self.total_fringes -= 1
        
#         self.tracked_peaks = new_peaks
#         new_point = (x[new_center], y[new_center])
        
#         if len(self.position_history) >= self.position_history.maxlen:
#             x_avg = int(np.mean([p[0] for p in self.position_history]))
#             y_avg = int(np.mean([p[1] for p in self.position_history]))
#             self.tracked_point = (x_avg, y_avg)
#         else:
#             self.tracked_point = new_point
            
#         self.position_history.append(new_point)
        
#         self.peak_heights = heights[np.searchsorted(peaks, new_peaks)]
#         if len(distances) >= 2:
#             self.peak_distances = distances[np.searchsorted(peaks[:-1], new_peaks[:-1])]
    
#     def process_video(self, input_path, output_path, auto_mode=True, points=None):
#         """处理视频并返回结果"""
#         global processing_progress
        
#         self.load_video(input_path, auto_mode, points)
        
#         # 获取总帧数
#         total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
#         processing_progress["total_frames"] = total_frames
#         processing_progress["current_frame"] = 0
#         processing_progress["is_processing"] = True
        
#         # 初始化追踪
#         profile_smooth, profile_peaks, x, y = self.get_line_profile(self.first_frame)
#         if not self.initialize_tracking(profile_smooth, profile_peaks, x, y):
#             processing_progress["is_processing"] = False
#             raise ValueError("Failed to initialize tracking")
            
#         # 初始化视频写入器
#         fps = int(self.cap.get(cv2.CAP_PROP_FPS))
#         width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#         height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#         fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#         self.video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
#         frame_count = 0
#         while True:
#             ret, frame = self.cap.read()
#             if not ret:
#                 break
                
#             processed_frame = self.preprocess_frame(frame)
#             profile_smooth, profile_peaks, x, y = self.get_line_profile(processed_frame)
#             if profile_smooth is not None:
#                 self.update_tracking(profile_smooth, profile_peaks, x, y)
            
#             # 绘制追踪信息
#             frame_with_overlay = self.draw_overlay(frame)
#             self.video_writer.write(frame_with_overlay)
            
#             # 更新进度
#             frame_count += 1
#             processing_progress["current_frame"] = frame_count
        
#         # 清理资源
#         if self.video_writer is not None:
#             self.video_writer.release()
#         self.cap.release()
        
#         processing_progress["is_processing"] = False
#         return self.total_fringes

#     def draw_overlay(self, frame):
#         """在视频帧上绘制追踪信息"""
#         if len(self.points) == 2:
#             cv2.line(frame, self.points[0], self.points[1], (0, 255, 0), 1)
            
#         if self.tracked_point is not None:
#             cv2.circle(frame, self.tracked_point, 5, (0, 0, 255), -1)
#             cv2.circle(frame, self.tracked_point, 8, (255, 0, 0), 2)
            
#         text = f"Fringes: {self.total_fringes}"
#         cv2.putText(frame, text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
#         return frame

# @app.post("/process-video")
# async def process_video(
#     file: UploadFile = File(...),
#     auto_mode: bool = Form(True),
#     points: str = Form(None)
# ):
#     global current_video_path, processing_progress
    
#     # 重置进度
#     processing_progress = {"total_frames": 0, "current_frame": 0, "is_processing": False}
    
#     # 创建临时目录存储文件
#     temp_dir = tempfile.mkdtemp()
#     try:
#         # 保存上传的视频
#         input_path = os.path.join(temp_dir, "input.mp4")
#         with open(input_path, "wb") as buffer:
#             shutil.copyfileobj(file.file, buffer)
        
#         # 设置输出路径
#         output_path = os.path.join(temp_dir, "output.mp4")
        
#         # 处理视频
#         tracker = RobustFringeTracker()
#         try:
#             # 解析points参数（如果存在）
#             points_list = None
#             if points and not auto_mode:
#                 try:
#                     points_data = eval(points)
#                     if isinstance(points_data, list) and len(points_data) == 2:
#                         points_list = points_data
#                 except:
#                     return JSONResponse(
#                         status_code=400,
#                         content={"success": False, "error": "Invalid points format"}
#                     )

#             total_fringes = tracker.process_video(
#                 input_path, 
#                 output_path,
#                 auto_mode=auto_mode,
#                 points=points_list
#             )
            
#             # 更新当前视频路径
#             if current_video_path and os.path.exists(current_video_path):
#                 try:
#                     shutil.rmtree(os.path.dirname(current_video_path))
#                 except:
#                     pass
#             current_video_path = output_path
            
#             return {
#                 "success": True,
#                 "total_fringes": total_fringes
#             }
#         except Exception as e:
#             return {"success": False, "error": str(e)}
#     except Exception as e:
#         return {"success": False, "error": str(e)}

# @app.get("/processing-progress")
# async def get_processing_progress():
#     """获取视频处理进度"""
#     return processing_progress

# @app.get("/download-video")
# async def download_video():
#     global current_video_path
#     if current_video_path and os.path.exists(current_video_path):
#         return FileResponse(current_video_path, filename="processed_video.mp4")
#     return {"error": "Video not found"}

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)

import cv2
import numpy as np
from scipy.signal import savgol_filter, find_peaks
from collections import deque
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import tempfile
import shutil

app = FastAPI()

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 存储当前处理的视频路径和进度
current_video_path = None
processing_progress = {"total_frames": 0, "current_frame": 0, "is_processing": False}

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

        self.video_writer = None
    
    def reset_tracking_state(self, profile_smooth, profile_peaks, x, y):
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

    def load_video(self, video_path, auto_mode=True, points=None):
        """加载视频并设置基准线"""
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError("Cannot load video")
            
        ret, self.first_frame = self.cap.read()
        if not ret:
            raise ValueError("Cannot read first frame")
            
        self.first_frame = self.preprocess_frame(self.first_frame)
        
        height, width = self.first_frame.shape
        if auto_mode:
            # 自动模式：选择一条竖线
            self.points = [(int(width * 0.5), int(height * 0.2)), 
                          (int(width * 0.5), int(height * 0.8))]
        elif points:
            # 手动模式：使用用户提供的点
            self.points = points

    def preprocess_frame(self, frame):
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.equalizeHist(frame)
        frame = cv2.GaussianBlur(frame, (3, 3), 0)
        return frame
    
    def get_line_profile(self, image):
        if len(self.points) != 2:
            return None, None, None
            
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
        if peaks is None or len(peaks) < self.min_peaks:
            return False
            
        if self.peak_heights is not None:
            peak_indices = np.searchsorted(peaks, self.tracked_peaks)
            if np.any(peak_indices >= len(heights)):
                return False
            selected_heights = heights[peak_indices]
            height_diff = np.abs(selected_heights - self.peak_heights)
            if np.max(height_diff) > np.mean(self.peak_heights) * 0.3:
                return False
        
        if self.peak_distances is not None and distances is not None:
            peak_indices = np.searchsorted(peaks[:-1], self.tracked_peaks[:-1])
            if np.any(peak_indices >= len(distances)):
                return False
            selected_distances = distances[peak_indices]
            dist_diff = np.abs(selected_distances - self.peak_distances)
            if np.max(dist_diff) > np.mean(self.peak_distances) * 0.3:
                return False
        
        return True
    
    def find_matching_peaks(self, peaks, profile_smooth, profile_peaks, x, y):
        if len(peaks) < self.min_peaks:
            return None
            
        best_match = None
        min_diff = float('inf')
        
        max_allowed_shift = self.wavelength * 0.5 if self.wavelength is not None else float('inf')
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
        
        valid_matches = []
        for i in range(len(peaks) - 2):
            peak_group = peaks[i:i+3]
            center_peak = peak_group[1]
            if (self.valid_region[0] <= center_peak <= self.valid_region[1]):
                position_diff = abs(center_peak - current_center)
                if position_diff <= max_allowed_shift:
                    valid_matches.append((position_diff, peak_group))
        
        if valid_matches:
            valid_matches.sort(key=lambda x: x[0])
            return valid_matches[0][1]
        
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
        if profile_smooth is None:
            self.consecutive_failures += 1
            return
            
        peaks, heights, distances = self.find_peaks_with_features(profile_smooth)
        if peaks is None:
            self.consecutive_failures += 1
            return
            
        if not self.validate_peaks(peaks, heights, distances):
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
        
        if self.wavelength is not None:
            if abs(movement) > self.wavelength * 0.5:
                if movement > 0:
                    self.total_fringes += 1
                else:
                    self.total_fringes -= 1
        
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
    
    def process_video(self, input_path, output_path, auto_mode=True, points=None):
        """处理视频并返回结果"""
        global processing_progress
        
        self.load_video(input_path, auto_mode, points)
        
        # 获取总帧数
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        processing_progress["total_frames"] = total_frames
        processing_progress["current_frame"] = 0
        processing_progress["is_processing"] = True
        
        # 初始化追踪
        profile_smooth, profile_peaks, x, y = self.get_line_profile(self.first_frame)
        if not self.initialize_tracking(profile_smooth, profile_peaks, x, y):
            processing_progress["is_processing"] = False
            raise ValueError("Failed to initialize tracking")
            
        # 初始化视频写入器
        fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            processed_frame = self.preprocess_frame(frame)
            profile_smooth, profile_peaks, x, y = self.get_line_profile(processed_frame)
            if profile_smooth is not None:
                self.update_tracking(profile_smooth, profile_peaks, x, y)
            
            # 绘制追踪信息
            frame_with_overlay = self.draw_overlay(frame)
            self.video_writer.write(frame_with_overlay)
            
            # 更新进度
            frame_count += 1
            processing_progress["current_frame"] = frame_count
        
        # 清理资源
        if self.video_writer is not None:
            self.video_writer.release()
        self.cap.release()
        
        processing_progress["is_processing"] = False
        return self.total_fringes

    def draw_overlay(self, frame):
        """在视频帧上绘制追踪信息"""
        if len(self.points) == 2:
            cv2.line(frame, self.points[0], self.points[1], (0, 255, 0), 1)
            
        if self.tracked_point is not None:
            cv2.circle(frame, self.tracked_point, 5, (0, 0, 255), -1)
            cv2.circle(frame, self.tracked_point, 8, (255, 0, 0), 2)
            
        text = f"Fringes: {self.total_fringes}"
        cv2.putText(frame, text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return frame

@app.post("/process-video")
async def process_video(
    file: UploadFile = File(...),
    auto_mode: bool = Form(True),
    points: str = Form(None)
):
    global current_video_path, processing_progress
    
    # 重置进度
    processing_progress = {"total_frames": 0, "current_frame": 0, "is_processing": False}
    
    # 创建临时目录存储文件
    temp_dir = tempfile.mkdtemp()
    try:
        # 保存上传的视频
        input_path = os.path.join(temp_dir, "input.mp4")
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 设置输出路径
        output_path = os.path.join(temp_dir, "output.mp4")
        
        # 处理视频
        tracker = RobustFringeTracker()
        try:
            # 解析points参数（如果存在）
            points_list = None
            if points and not auto_mode:
                try:
                    points_data = eval(points)
                    if isinstance(points_data, list) and len(points_data) == 2:
                        points_list = points_data
                except:
                    return JSONResponse(
                        status_code=400,
                        content={"success": False, "error": "Invalid points format"}
                    )

            total_fringes = tracker.process_video(
                input_path, 
                output_path,
                auto_mode=auto_mode,
                points=points_list
            )
            
            # 更新当前视频路径
            if current_video_path and os.path.exists(current_video_path):
                try:
                    shutil.rmtree(os.path.dirname(current_video_path))
                except:
                    pass
            current_video_path = output_path
            
            return {
                "success": True,
                "total_fringes": total_fringes
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/processing-progress")
async def get_processing_progress():
    """获取视频处理进度"""
    return processing_progress

@app.get("/download-video")
async def download_video():
    global current_video_path
    if current_video_path and os.path.exists(current_video_path):
        return FileResponse(current_video_path, filename="processed_video.mp4")
    return {"error": "Video not found"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
