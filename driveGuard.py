import cv2
import numpy as np
import time
import pygame
import argparse
from scipy.spatial import distance as dist
import dlib
import os
from datetime import datetime

def calculate_ear(eye):
    """
    Calculate the Eye Aspect Ratio (EAR)
    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    """
    # Compute the euclidean distances between the two sets of
    # vertical eye landmarks
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    
    # Compute the euclidean distance between the horizontal
    # eye landmark
    C = dist.euclidean(eye[0], eye[3])
    
    # Compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    return ear

class DriveGuard:
    def __init__(self, ear_threshold=0.25, consecutive_frames=3, 
                drowsy_time_threshold=5.0, recovery_time_threshold=5.0, show_fps=True, dark_mode=True):
        """
        Initialize DriveGuard with detection parameters
        
        Args:
            ear_threshold: The threshold below which an eye is considered closed
            consecutive_frames: Number of consecutive frames eyes must be closed to count as a blink
            drowsy_time_threshold: Number of seconds with closed eyes to trigger an alert
            recovery_time_threshold: Number of seconds with open eyes to stop the alert
            show_fps: Whether to display FPS on the frame
            dark_mode: Whether to use dark mode UI
        """
        self.ear_threshold = ear_threshold
        self.consecutive_frames = consecutive_frames
        self.drowsy_time_threshold = drowsy_time_threshold
        self.recovery_time_threshold = recovery_time_threshold
        self.show_fps = show_fps
        self.dark_mode = dark_mode
        
        # UI colors based on theme
        if self.dark_mode:
            self.bg_color = (40, 44, 52)  # Dark background
            self.text_color = (236, 240, 241)  # Light text
            self.panel_color = (53, 59, 72)  # Dark panel
            self.accent_color = (52, 152, 219)  # Blue accent
        else:
            self.bg_color = (240, 240, 240)  # Light background
            self.text_color = (44, 62, 80)  # Dark text
            self.panel_color = (220, 220, 220)  # Light panel
            self.accent_color = (41, 128, 185)  # Blue accent
            
        # Status colors (consistent across themes)
        self.alert_color = (231, 76, 60)  # Red
        self.warning_color = (243, 156, 18)  # Orange
        self.safe_color = (46, 204, 113)  # Green
        
        # UI fonts and sizes
        self.font_face = cv2.FONT_HERSHEY_SIMPLEX
        self.title_font_scale = 0.8
        self.regular_font_scale = 0.6
        self.small_font_scale = 0.5
        self.font_thickness = 2
        self.thin_font_thickness = 1
        
        # Initialize dlib's face detector and facial landmark predictor
        self.detector = dlib.get_frontal_face_detector()
        
        # Download the predictor file if not already present
        predictor_path = "shape_predictor_68_face_landmarks.dat"
        if not os.path.exists(predictor_path):
            print("Landmark predictor file not found. Please download it from:")
            print("http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
            print("Extract and place in the same directory as this script.")
            raise FileNotFoundError(f"Could not find {predictor_path}")
            
        self.predictor = dlib.shape_predictor(predictor_path)
        
        # Define indices for left and right eyes according to facial landmark map
        self.left_eye_indices = list(range(36, 42))
        self.right_eye_indices = list(range(42, 48))
        
        # Initialize pygame for audio alerts
        pygame.mixer.init()
        
        # Try using a system beep if alert.wav is not available
        alert_sound_path = "alert.wav"
        if os.path.exists(alert_sound_path):
            self.alert_sound = pygame.mixer.Sound(alert_sound_path)
            self.use_tone = False
        else:
            # Create a simple tone as a fallback
            pygame.mixer.quit()
            pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
            self.use_tone = True
        
        # Drowsiness tracking variables
        self.drowsy_start_time = None
        self.recovery_start_time = None
        self.is_alarm_active = False
        self.last_alarm_time = 0
        self.frame_counter = 0
        self.closed_eyes_frames = 0
        
        # FPS calculation variables
        self.prev_frame_time = 0
        self.new_frame_time = 0
        self.fps_values = []
        self.avg_fps = 0
        
        # Status message and color
        self.status = "Monitoring"
        self.status_color = self.safe_color
        
        # Continuous monitoring variables
        self.eyes_closed = False
        self.eyes_open = False
        
        # EAR history for smoothing
        self.ear_history = []
        self.ear_history_size = 5  # Keep track of last 5 EAR values for smoothing
        
        # Statistics tracking
        self.detection_start_time = time.time()
        self.blink_count = 0
        self.drowsy_events = 0
        self.longest_drowsy_duration = 0
        self.total_drowsy_time = 0
        self.ear_values = []  # Store EAR values for statistics
        
        # Save frames of drowsy events (limit to avoid memory issues)
        self.max_saved_frames = 5
        self.saved_drowsy_frames = []
        
        # Create logs directory
        self.logs_dir = "drive_guard_logs"
        if not os.path.exists(self.logs_dir):
            os.makedirs(self.logs_dir)

    def play_alert(self):
        """Play the alert sound or generate a beep"""
        current_time = time.time()
        # Only play alert if it's been at least 1 second since the last alert
        if current_time - self.last_alarm_time > 1:
            self.last_alarm_time = current_time
            try:
                if self.use_tone:
                    # Generate a simple beep
                    duration = 1  # seconds
                    frequency = 1000  # Hz
                    pygame.mixer.Sound(pygame.sndarray.make_sound(
                        np.sin(2 * np.pi * np.arange(44100 * duration) * frequency / 44100).astype(np.float32)
                    )).play()
                else:
                    self.alert_sound.play()
            except Exception as e:
                print(f"Error playing alert: {e}")
    
    def shape_to_np(self, shape, dtype="int"):
        """Convert dlib's shape object to numpy array"""
        # Initialize the list of (x, y) coordinates
        coords = np.zeros((68, 2), dtype=dtype)
        
        # Loop over the 68 facial landmarks and convert them
        # to a 2-tuple of (x, y) coordinates
        for i in range(0, 68):
            coords[i] = (shape.part(i).x, shape.part(i).y)
            
        return coords

    def get_eye_landmarks(self, landmarks, eye_indices):
        """Extract eye landmarks from facial landmarks"""
        eye = []
        for i in eye_indices:
            eye.append(landmarks[i])
        return eye
    
    def visualize_facial_landmarks(self, frame, landmarks, color=(0, 255, 0), with_numbers=False):
        """Draw facial landmarks on the frame"""
        for i, (x, y) in enumerate(landmarks):
            # Draw the landmark
            cv2.circle(frame, (x, y), 1, color, -1)
            
            # Draw landmark number if requested
            if with_numbers:
                cv2.putText(frame, str(i), (x, y), self.font_face, 0.3, (0, 255, 255), 1)
        
        # Draw lines connecting the facial landmarks for left and right eyes
        for eye_indices in [self.left_eye_indices, self.right_eye_indices]:
            # Convert to tuples for drawing
            pts = np.array([landmarks[i] for i in eye_indices], np.int32)
            cv2.polylines(frame, [pts], True, color, 2)
    
    def get_smooth_ear(self, ear):
        """Get smoothed EAR value using a moving average"""
        self.ear_history.append(ear)
        if len(self.ear_history) > self.ear_history_size:
            self.ear_history.pop(0)
        return sum(self.ear_history) / len(self.ear_history)
    
    def draw_rounded_rectangle(self, frame, top_left, bottom_right, color, radius=15, thickness=-1):
        """Draw a rounded rectangle on the frame"""
        x1, y1 = top_left
        x2, y2 = bottom_right
        
        # Draw the main rectangle
        cv2.rectangle(frame, (x1+radius, y1), (x2-radius, y2), color, thickness)
        cv2.rectangle(frame, (x1, y1+radius), (x2, y2-radius), color, thickness)
        
        # Draw the four corner circles
        cv2.circle(frame, (x1+radius, y1+radius), radius, color, thickness)
        cv2.circle(frame, (x2-radius, y1+radius), radius, color, thickness)
        cv2.circle(frame, (x1+radius, y2-radius), radius, color, thickness)
        cv2.circle(frame, (x2-radius, y2-radius), radius, color, thickness)
    
    def draw_progress_bar(self, frame, x, y, width, height, value, max_value, color_low, color_high, bg_color):
        """Draw a progress bar with gradient color based on value"""
        # Background
        cv2.rectangle(frame, (x, y), (x + width, y + height), bg_color, -1)
        
        # Progress
        if max_value > 0:  # Avoid division by zero
            progress_width = int((value / max_value) * width)
            # Calculate color based on progress
            t = value / max_value
            r = int(color_low[0] * (1-t) + color_high[0] * t)
            g = int(color_low[1] * (1-t) + color_high[1] * t)
            b = int(color_low[2] * (1-t) + color_high[2] * t)
            color = (r, g, b)
            cv2.rectangle(frame, (x, y), (x + progress_width, y + height), color, -1)
            
        # Border
        cv2.rectangle(frame, (x, y), (x + width, y + height), self.text_color, 1)
    
    def draw_ui_panel(self, frame, x, y, width, height, title=None, footer=None):
        """Draw a panel with optional title and footer"""
        # Draw panel background
        self.draw_rounded_rectangle(frame, (x, y), (x + width, y + height), self.panel_color)
        
        # Draw title if provided
        if title:
            cv2.putText(frame, title, (x + 10, y + 25), self.font_face, 
                      self.title_font_scale, self.accent_color, self.font_thickness)
            # Draw separator line
            cv2.line(frame, (x + 5, y + 35), (x + width - 5, y + 35), 
                   self.accent_color, 1)
        
        # Draw footer if provided
        if footer:
            # Draw separator line
            cv2.line(frame, (x + 5, y + height - 35), (x + width - 5, y + height - 35), 
                   self.accent_color, 1)
            cv2.putText(frame, footer, (x + 10, y + height - 15), self.font_face, 
                      self.small_font_scale, self.text_color, self.thin_font_thickness)
    
    def draw_status_panel(self, frame, status, color, icon=None):
        """Draw a status panel with icon and message"""
        frame_height, frame_width = frame.shape[:2]
        panel_width = frame_width - 20
        panel_height = 80
        panel_x = 10
        panel_y = frame_height - panel_height - 10
        
        # Draw panel background
        self.draw_rounded_rectangle(frame, (panel_x, panel_y), 
                                 (panel_x + panel_width, panel_y + panel_height), color)
        
        # Draw icon or status circle
        if icon is not None:
            # If an icon is provided, draw it
            frame[panel_y+15:panel_y+15+icon.shape[0], panel_x+15:panel_x+15+icon.shape[1]] = icon
            text_x = panel_x + 15 + icon.shape[1] + 10
        else:
            # Draw a status circle
            cv2.circle(frame, (panel_x + 30, panel_y + 40), 15, (255, 255, 255), -1)
            text_x = panel_x + 60
        
        # Draw status text
        cv2.putText(frame, status, (text_x, panel_y + 45), self.font_face, 
                  self.title_font_scale, (255, 255, 255), self.font_thickness)
    
    def save_drowsy_frame(self, frame):
        """Save a frame during a drowsy event"""
        if len(self.saved_drowsy_frames) < self.max_saved_frames:
            # Create a copy to avoid reference issues
            self.saved_drowsy_frames.append(frame.copy())
            
            # Save to disk as well
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(f"{self.logs_dir}/drowsy_event_{timestamp}.jpg", frame)
    
    def draw_statistics_panel(self, frame):
        """Draw a panel showing monitoring statistics"""
        frame_height, frame_width = frame.shape[:2]
        stats_x = frame_width - 210
        stats_y = 10
        stats_width = 200
        stats_height = 180
        
        # Calculate session duration
        session_duration = time.time() - self.detection_start_time
        hours, remainder = divmod(session_duration, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        # Calculate average EAR
        avg_ear = sum(self.ear_values) / max(1, len(self.ear_values))
        
        # Draw statistics panel
        self.draw_ui_panel(frame, stats_x, stats_y, stats_width, stats_height, 
                         "Session Statistics")
        
        # Draw statistics
        metrics = [
            f"Duration: {int(hours)}h {int(minutes)}m {int(seconds)}s",
            f"Blinks: {self.blink_count}",
            f"Drowsy Events: {self.drowsy_events}",
            f"Longest Drowsy: {self.longest_drowsy_duration:.1f}s",
            f"Avg EAR: {avg_ear:.3f}",
            f"FPS: {self.avg_fps:.1f}"
        ]
        
        for i, metric in enumerate(metrics):
            cv2.putText(frame, metric, (stats_x + 10, stats_y + 55 + (i * 20)), 
                      self.font_face, self.small_font_scale, self.text_color, self.thin_font_thickness)
    
    def process_frame(self, frame):
        """
        Process a frame to detect drowsiness
        
        Args:
            frame: The video frame to process
            
        Returns:
            The processed frame with detection overlays
        """
        current_time = time.time()
        self.frame_counter += 1
        
        # Create a clean copy of the frame for display
        original_frame = frame.copy()
        
        # Create UI background
        if self.dark_mode:
            # Darken the frame
            frame = cv2.addWeighted(frame, 0.7, np.zeros(frame.shape, frame.dtype), 0, 0)
        
        # Calculate FPS
        self.new_frame_time = current_time
        if self.prev_frame_time > 0:
            fps = 1 / (self.new_frame_time - self.prev_frame_time)
            self.fps_values.append(fps)
            # Keep only last 10 FPS values for averaging
            if len(self.fps_values) > 10:
                self.fps_values.pop(0)
            self.avg_fps = sum(self.fps_values) / len(self.fps_values)
        self.prev_frame_time = self.new_frame_time
        
        # Display FPS if enabled
        if self.show_fps:
            cv2.putText(frame, f"FPS: {int(self.avg_fps)}", (10, 30), self.font_face, 
                      self.regular_font_scale, self.text_color, self.thin_font_thickness)
        
        # Convert to grayscale for detection
        gray = cv2.cvtColor(original_frame, cv2.COLOR_BGR2GRAY)
        
        # Apply histogram equalization to improve detection in different lighting
        gray = cv2.equalizeHist(gray)
        
        # Detect faces using dlib
        faces = self.detector(gray, 0)
        
        # Draw main UI panel
        main_panel_x = 10
        main_panel_y = 10
        main_panel_width = 300
        main_panel_height = 160
        
        self.draw_ui_panel(frame, main_panel_x, main_panel_y, main_panel_width, main_panel_height,
                         "DriveGuard Monitor", f"v2.0 - {datetime.now().strftime('%H:%M:%S')}")
        
        if len(faces) == 0:
            # No face detected
            self.status = "No face detected"
            self.status_color = self.alert_color
            
            # Reset timers when no face is detected
            self.eyes_closed = False
            self.drowsy_start_time = None
            
            # Keep alarm active if it was already active
            if not self.is_alarm_active:
                self.recovery_start_time = None
                self.eyes_open = False
                
            # Display no face message
            cv2.putText(frame, "No face detected", (main_panel_x + 10, main_panel_y + 70), 
                      self.font_face, self.regular_font_scale, self.alert_color, self.font_thickness)
            
            # Draw status icon
            self.draw_status_panel(frame, "NO FACE DETECTED", self.alert_color)
        else:
            # Sort faces by area to get the largest face (likely the driver)
            if len(faces) > 1:
                faces = sorted(faces, key=lambda rect: (rect.right() - rect.left()) * (rect.bottom() - rect.top()), reverse=True)
            
            face = faces[0]
            
            # Draw face rectangle
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            cv2.rectangle(frame, (x, y), (x+w, y+h), self.accent_color, 2)
            
            # Get facial landmarks
            landmarks = self.predictor(gray, face)
            landmarks = self.shape_to_np(landmarks)
            
            # Visualize landmarks
            self.visualize_facial_landmarks(frame, landmarks, color=self.accent_color)
            
            # Get left and right eye coordinates
            left_eye = self.get_eye_landmarks(landmarks, self.left_eye_indices)
            right_eye = self.get_eye_landmarks(landmarks, self.right_eye_indices)
            
            # Calculate EAR for both eyes
            left_ear = calculate_ear(left_eye)
            right_ear = calculate_ear(right_eye)
            
            # Average the EAR between both eyes
            avg_ear = (left_ear + right_ear) / 2.0
            
            # Get smoothed EAR
            smooth_ear = self.get_smooth_ear(avg_ear)
            self.ear_values.append(smooth_ear)
            
            # Display EAR values in the main panel
            cv2.putText(frame, f"Eye Aspect Ratio:", (main_panel_x + 10, main_panel_y + 70), 
                      self.font_face, self.regular_font_scale, self.text_color, self.thin_font_thickness)
            
            # Draw EAR progress bar
            bar_x = main_panel_x + 10
            bar_y = main_panel_y + 80
            bar_width = main_panel_width - 20
            bar_height = 20
            
            # Draw EAR bar
            self.draw_progress_bar(
                frame, bar_x, bar_y, bar_width, bar_height, 
                smooth_ear, 0.4,  # 0.4 is max expected EAR
                self.alert_color, self.safe_color,
                (self.bg_color if self.dark_mode else self.panel_color)
            )
            
            # Add EAR threshold line
            threshold_x = bar_x + int(bar_width * (self.ear_threshold / 0.4))
            cv2.line(frame, (threshold_x, bar_y - 5), (threshold_x, bar_y + bar_height + 5), 
                   self.warning_color, 2)
            
            # Show EAR value
            cv2.putText(frame, f"{smooth_ear:.2f}", (main_panel_x + main_panel_width - 50, main_panel_y + 70), 
                      self.font_face, self.regular_font_scale, self.text_color, self.font_thickness)
            
            # Check if eyes are closed based on the EAR threshold
            if smooth_ear < self.ear_threshold:
                self.closed_eyes_frames += 1
                
                # Eyes are considered closed after consecutive frames
                if self.closed_eyes_frames >= self.consecutive_frames:
                    if not self.eyes_closed:
                        self.eyes_closed = True
                        self.drowsy_start_time = current_time
                        self.eyes_open = False
                        self.recovery_start_time = None
                        self.blink_count += 1  # Count as a blink
                
                # Calculate how long the eyes have been closed
                if self.eyes_closed and self.drowsy_start_time:
                    drowsy_duration = current_time - self.drowsy_start_time
                    
                    # Display drowsy duration in main panel
                    cv2.putText(frame, f"Eyes Closed: {drowsy_duration:.1f}s/{self.drowsy_time_threshold:.1f}s", 
                              (main_panel_x + 10, main_panel_y + 120), 
                              self.font_face, self.regular_font_scale, self.warning_color, self.thin_font_thickness)
                    
                    # Draw drowsy duration progress bar
                    self.draw_progress_bar(
                        frame, main_panel_x + 10, main_panel_y + 130, main_panel_width - 20, 10,
                        drowsy_duration, self.drowsy_time_threshold,
                        self.safe_color, self.alert_color,
                        (self.bg_color if self.dark_mode else self.panel_color)
                    )
                    
                    # Update longest drowsy duration
                    self.longest_drowsy_duration = max(self.longest_drowsy_duration, drowsy_duration)
                    
                    # Check if the eyes have been closed for too long
                    if drowsy_duration >= self.drowsy_time_threshold:
                        # Trigger the alarm if it's not already active
                        if not self.is_alarm_active:
                            self.is_alarm_active = True
                            self.play_alert()
                            self.drowsy_events += 1
                            self.save_drowsy_frame(original_frame)
                            
                            # Log drowsy event
                            with open(f"{self.logs_dir}/drowsy_events.log", "a") as log_file:
                                log_file.write(f"{datetime.now()}: Drowsy event detected - Duration: {drowsy_duration:.1f}s\n")
                        elif current_time - self.last_alarm_time > 2:
                            # Play alert every 2 seconds while drowsy
                            self.play_alert()
                        
                        self.status = "DROWSINESS DETECTED!"
                        self.status_color = self.alert_color
                        
                        # Update total drowsy time
                        self.total_drowsy_time += (current_time - self.last_alarm_time)
                        
                        # Draw the drowsiness alert panel
                        self.draw_status_panel(frame, "DROWSINESS DETECTED!", self.alert_color)
            else:
                # Reset the counter for closed eyes frames
                self.closed_eyes_frames = 0
                
                # Eyes are open
                if self.eyes_closed:
                    self.eyes_closed = False
                    self.drowsy_start_time = None
                
                if self.is_alarm_active:
                    # Start recovery timer if eyes open and alarm is active
                    if not self.eyes_open:
                        self.eyes_open = True
                        self.recovery_start_time = current_time
                    
                    # Calculate recovery duration
                    if self.recovery_start_time:
                        recovery_duration = current_time - self.recovery_start_time
                        
                        # Display recovery duration in main panel
                        cv2.putText(frame, f"Recovery: {recovery_duration:.1f}s/{self.recovery_time_threshold:.1f}s", 
                                  (main_panel_x + 10, main_panel_y + 120), 
                                  self.font_face, self.regular_font_scale, self.safe_color, self.thin_font_thickness)
                        
                        # Draw recovery progress bar
                        self.draw_progress_bar(
                            frame, main_panel_x + 10, main_panel_y + 130, main_panel_width - 20, 10,
                            recovery_duration, self.recovery_time_threshold,
                            self.alert_color, self.safe_color,
                            (self.bg_color if self.dark_mode else self.panel_color)
                        )
                        
                        # Check if eyes have been open long enough to deactivate the alarm
                        if recovery_duration >= self.recovery_time_threshold:
                            self.is_alarm_active = False
                            self.recovery_start_time = None
                            self.eyes_open = False
                            
                            # Log recovery
                            with open(f"{self.logs_dir}/drowsy_events.log", "a") as log_file:
                                log_file.write(f"{datetime.now()}: Recovered from drowsy state\n")
                
                self.status = "Monitoring - Eyes Open"
                self.status_color = self.safe_color
                
                # Display status in main panel
                cv2.putText(frame, "Eyes: OPEN", (main_panel_x + 10, main_panel_y + 120), 
                          self.font_face, self.regular_font_scale, self.safe_color, self.font_thickness)
                
                # Draw the drowsiness alert panel if alarm is active, otherwise normal panel
                if self.is_alarm_active:
                    self.draw_status_panel(frame, "REMAIN ALERT!", self.warning_color)
                else:
                    self.draw_status_panel(frame, "MONITORING - ALERT", self.safe_color)
        
        # Draw statistics panel
        self.draw_statistics_panel(frame)
        
        return frame
    
    def run(self, camera_index=0):
        """
        Run the DriveGuard system using the specified camera
        
        Args:
            camera_index: Index of the camera to use (default: 0)
        """
        try:
            # Initialize webcam
            cap = cv2.VideoCapture(camera_index)
            
            if not cap.isOpened():
                print("Error: Could not open camera.")
                return
            
            # Try to set higher resolution for better detection
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            
            print("=" * 50)
            print("Enhanced DriveGuard v2.0 is running")
            print("=" * 50)
            print(f"Theme: {'Dark Mode' if self.dark_mode else 'Light Mode'}")
            print(f"Eyes are considered closed when EAR < {self.ear_threshold}")
            print(f"Drowsiness detection after {self.drowsy_time_threshold} seconds of closed eyes")
            print(f"Alarm will stop after {self.recovery_time_threshold} seconds of open eyes")
            print("=" * 50)
            print("Press 'q' to quit, 's' to take a screenshot, 't' to toggle theme")
            print("=" * 50)
            
            # Create window with custom properties
            window_name = 'DriveGuard v2.0 - Driver Sleep Detection'
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 1280, 720)
            
            while True:
                # Read frame from camera
                ret, frame = cap.read()
                
                if not ret:
                    print("Error: Failed to capture video frame.")
                    break
                
                # Process the frame
                processed_frame = self.process_frame(frame)
                
                # Display the processed frame
                cv2.imshow(window_name, processed_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):  # Quit
                    break
                elif key == ord('s'):  # Screenshot
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%")
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    screenshot_path = f"{self.logs_dir}/screenshot_{timestamp}.jpg"
                    cv2.imwrite(screenshot_path, processed_frame)
                    print(f"Screenshot saved to {screenshot_path}")
                elif key == ord('t'):  # Toggle theme
                    self.dark_mode = not self.dark_mode
                    # Update UI colors based on new theme
                    if self.dark_mode:
                        self.bg_color = (40, 44, 52)  # Dark background
                        self.text_color = (236, 240, 241)  # Light text
                        self.panel_color = (53, 59, 72)  # Dark panel
                        self.accent_color = (52, 152, 219)  # Blue accent
                    else:
                        self.bg_color = (240, 240, 240)  # Light background
                        self.text_color = (44, 62, 80)  # Dark text
                        self.panel_color = (220, 220, 220)  # Light panel
                        self.accent_color = (41, 128, 185)  # Blue accent
                    print(f"Theme switched to {'Dark Mode' if self.dark_mode else 'Light Mode'}")
            
            # Clean up
            cap.release()
            cv2.destroyAllWindows()
            
            # Save session statistics
            self.save_session_stats()
            
        except Exception as e:
            print(f"Error in DriveGuard: {e}")
            import traceback
            traceback.print_exc()
    
    def save_session_stats(self):
        """Save session statistics to a log file"""
        session_duration = time.time() - self.detection_start_time
        hours, remainder = divmod(session_duration, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        avg_ear = sum(self.ear_values) / max(1, len(self.ear_values))
        
        with open(f"{self.logs_dir}/session_stats.log", "a") as stats_file:
            stats_file.write("\n" + "="*50 + "\n")
            stats_file.write(f"Session ended at: {datetime.now()}\n")
            stats_file.write(f"Duration: {int(hours)}h {int(minutes)}m {int(seconds)}s\n")
            stats_file.write(f"Blinks detected: {self.blink_count}\n")
            stats_file.write(f"Drowsy events: {self.drowsy_events}\n")
            stats_file.write(f"Longest drowsy duration: {self.longest_drowsy_duration:.1f}s\n")
            stats_file.write(f"Total drowsy time: {self.total_drowsy_time:.1f}s\n")
            stats_file.write(f"Average EAR: {avg_ear:.3f}\n")
            stats_file.write(f"Average FPS: {self.avg_fps:.1f}\n")
            stats_file.write("="*50 + "\n")
        
        print(f"Session statistics saved to {self.logs_dir}/session_stats.log")


# Main function to run the application
def main():
    """Main function to parse arguments and run DriveGuard"""
    parser = argparse.ArgumentParser(description='DriveGuard - Driver Drowsiness Detection System')
    parser.add_argument('--camera', type=int, default=0, help='Camera index (default: 0)')
    parser.add_argument('--ear', type=float, default=0.25, help='EAR threshold (default: 0.25)')
    parser.add_argument('--frames', type=int, default=3, help='Consecutive frames (default: 3)')
    parser.add_argument('--drowsy', type=float, default=5.0, help='Drowsy time threshold in seconds (default: 5.0)')
    parser.add_argument('--recovery', type=float, default=5.0, help='Recovery time threshold in seconds (default: 5.0)')
    parser.add_argument('--no-fps', action='store_false', dest='show_fps', help='Hide FPS counter')
    parser.add_argument('--light-mode', action='store_false', dest='dark_mode', help='Use light mode interface')
    
    args = parser.parse_args()
    
    # Create DriveGuard instance
    drive_guard = DriveGuard(
        ear_threshold=args.ear,
        consecutive_frames=args.frames,
        drowsy_time_threshold=args.drowsy,
        recovery_time_threshold=args.recovery,
        show_fps=args.show_fps,
        dark_mode=args.dark_mode
    )
    
    # Run the system
    drive_guard.run(camera_index=args.camera)


if __name__ == "__main__":
    main()