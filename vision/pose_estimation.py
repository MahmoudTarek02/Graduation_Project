import cv2
import numpy as np
from ultralytics import YOLO

class PersonWristDetector:
    def __init__(self, model_path='yolo11n-pose.pt'):
        """
        Initialize the YOLOv11 pose estimation model
        """
        try:
            self.model = YOLO(model_path)
            print(f"Loaded model: {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Trying to download model...")
            self.model = YOLO('yolo11n-pose.pt')  # This will download if not available
    
    def detect_poses(self, frame):
        """
        Detect persons and their keypoints in the frame
        """
        try:
            # Use pose estimation specifically
            results = self.model(frame, verbose=False, conf=0.5)
            return results
        except Exception as e:
            print(f"Detection error: {e}")
            return []
    
    def get_wrist_coordinates(self, keypoints, person_idx=0):
        """
        Extract left and right wrist coordinates from pose keypoints
        Returns: (left_wrist, right_wrist) as (x, y) tuples
        """
        if keypoints is None or len(keypoints) <= person_idx:
            return None, None
        
        person_keypoints = keypoints[person_idx]
        
        # YOLO pose keypoint indices: 
        # 9 = left wrist, 10 = right wrist
        left_wrist_idx = 9
        right_wrist_idx = 10
        
        left_wrist = None
        right_wrist = None
        
        # Debug: print available keypoints
        print(f"Available keypoints: {len(person_keypoints)}")
        
        if len(person_keypoints) > left_wrist_idx:
            left_wrist_kp = person_keypoints[left_wrist_idx]
            print(f"Left wrist confidence: {left_wrist_kp[2]:.2f}")
            if left_wrist_kp[2] > 0.3:  # Lower confidence threshold
                left_wrist = (int(left_wrist_kp[0]), int(left_wrist_kp[1]))
                print(f"Left wrist detected at: {left_wrist}")
        
        if len(person_keypoints) > right_wrist_idx:
            right_wrist_kp = person_keypoints[right_wrist_idx]
            print(f"Right wrist confidence: {right_wrist_kp[2]:.2f}")
            if right_wrist_kp[2] > 0.3:  # Lower confidence threshold
                right_wrist = (int(right_wrist_kp[0]), int(right_wrist_kp[1]))
                print(f"Right wrist detected at: {right_wrist}")
        
        return left_wrist, right_wrist
    
    def crop_wrist_region(self, frame, wrist_point, crop_size=200):
        """
        Crop a square region around the wrist point
        """
        if wrist_point is None:
            return None
        
        x, y = wrist_point
        h, w = frame.shape[:2]
        
        # Calculate crop boundaries
        half_size = crop_size // 2
        x1 = max(0, x - half_size)
        y1 = max(0, y - half_size)
        x2 = min(w, x + half_size)
        y2 = min(h, y + half_size)
        
        # Ensure we have a valid crop
        if x2 - x1 < 50 or y2 - y1 < 50:  # Minimum size
            return None
        
        wrist_crop = frame[y1:y2, x1:x2]
        return wrist_crop
    
    def draw_all_keypoints(self, frame, keypoints, person_idx=0):
        """
        Draw all pose keypoints on the frame for visualization
        """
        if keypoints is None or len(keypoints) <= person_idx:
            return frame
        
        person_keypoints = keypoints[person_idx]
        
        # Keypoint names for YOLO pose model
        keypoint_names = [
            "nose", "left_eye", "right_eye", "left_ear", "right_ear",
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_hip", "right_hip",
            "left_knee", "right_knee", "left_ankle", "right_ankle"
        ]
        
        # Draw all keypoints
        for i, kp in enumerate(person_keypoints):
            x, y, conf = kp
            if conf > 0.3:  # Lower confidence threshold
                color = (0, 255, 0)  # Green for most keypoints
                if i == 9:  # Left wrist - Blue
                    color = (255, 0, 0)
                elif i == 10:  # Right wrist - Red
                    color = (0, 0, 255)
                
                cv2.circle(frame, (int(x), int(y)), 6, color, -1)
                cv2.putText(frame, f"{i}", (int(x) + 10, int(y)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        return frame
    
    def process_frame(self, frame):
        """
        Process a single frame and return the three views
        """
        # Detect persons and poses
        results = self.detect_poses(frame)
        
        full_view = frame.copy()
        left_wrist_view = None
        right_wrist_view = None
        
        print(f"Number of detections: {len(results)}")
        
        if len(results) > 0 and hasattr(results[0], 'keypoints') and results[0].keypoints is not None:
            keypoints = results[0].keypoints.data.cpu().numpy()
            print(f"Number of people detected: {len(keypoints)}")
            
            if len(keypoints) > 0:
                # Get wrist coordinates for the first person
                left_wrist, right_wrist = self.get_wrist_coordinates(keypoints, 0)
                
                # Draw all keypoints on full view
                full_view = self.draw_all_keypoints(full_view, keypoints, 0)
                
                # Highlight wrists with larger circles
                if left_wrist:
                    cv2.circle(full_view, left_wrist, 12, (255, 0, 0), -1)
                    cv2.putText(full_view, "LEFT WRIST", 
                               (left_wrist[0] + 15, left_wrist[1]), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    # Crop wrist region
                    left_wrist_view = self.crop_wrist_region(frame, left_wrist, 250)
                
                if right_wrist:
                    cv2.circle(full_view, right_wrist, 12, (0, 0, 255), -1)
                    cv2.putText(full_view, "RIGHT WRIST", 
                               (right_wrist[0] + 15, right_wrist[1]), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    # Crop wrist region
                    right_wrist_view = self.crop_wrist_region(frame, right_wrist, 250)
        
        return full_view, left_wrist_view, right_wrist_view

def create_display_layout(full_view, left_wrist, right_wrist, display_width=800):
    """
    Create a combined display with all three views
    """
    # Resize full view for display
    h, w = full_view.shape[:2]
    scale_factor = display_width / w
    new_height = int(h * scale_factor)
    new_width = display_width
    
    full_view_resized = cv2.resize(full_view, (new_width, new_height))
    
    # Create placeholder for missing wrist views
    wrist_size = 250
    placeholder = np.zeros((wrist_size, wrist_size, 3), dtype=np.uint8)
    cv2.putText(placeholder, "No Wrist Detected", (20, 125), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(placeholder, "Make sure wrists are visible", (10, 150), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Prepare wrist views
    def prepare_wrist_view(wrist_img, size, label):
        if wrist_img is None:
            display_img = placeholder.copy()
        else:
            display_img = cv2.resize(wrist_img, (size, size))
        
        # Add label
        color = (255, 0, 0) if "LEFT" in label else (0, 0, 255)
        cv2.putText(display_img, label, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return display_img
    
    left_display = prepare_wrist_view(left_wrist, wrist_size, "LEFT WRIST")
    right_display = prepare_wrist_view(right_wrist, wrist_size, "RIGHT WRIST")
    
    # Create wrist views row
    wrist_row = np.hstack([left_display, right_display])
    
    # Resize wrist row to match full view width
    wrist_row = cv2.resize(wrist_row, (new_width, wrist_size))
    
    # Create final layout
    layout = np.vstack([full_view_resized, wrist_row])
    
    return layout

def main():
    """
    Main function to run the person and wrist detection
    """
    # Initialize detector with pose model
    detector = PersonWristDetector('yolo11n-pose.pt')
    
    # Initialize video capture
    
    # cap = cv2.VideoCapture("Graduation_Project\\vision\\tracking\\Videos\\Security_camera.mp4")
    cap = cv2.VideoCapture(r"D:\Graduation_Project\vision\tracking\Videos\Security_camera.mp4")

    #cap = cv2.VideoCapture(0)  # Use webcam
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("Starting person and wrist detection...")
    print("TIPS:")
    print("- Make sure you're facing the camera")
    print("- Keep your arms visible and not crossed")
    print("- Ensure good lighting")
    print("- Stand reasonably close to the camera")
    print("Press 'q' to quit")
    print("Press 's' to save current frames")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        frame_count += 1
        print(f"\n--- Frame {frame_count} ---")
        
        # Process frame
        full_view, left_wrist, right_wrist = detector.process_frame(frame)
        
        # Create combined display
        display = create_display_layout(full_view, left_wrist, right_wrist, 800)
        
        # Show results
        cv2.imshow('Person & Wrist Detection', display)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Save current frames
            cv2.imwrite('full_view.jpg', full_view)
            if left_wrist is not None:
                cv2.imwrite('left_wrist.jpg', left_wrist)
            if right_wrist is not None:
                cv2.imwrite('right_wrist.jpg', right_wrist)
            print("Frames saved!")
    
    cap.release()
    cv2.destroyAllWindows()

# Alternative testing function with static image
def test_with_image():
    """
    Test the detection with a static image first
    """
    detector = PersonWristDetector('yolo11n-pose.pt')
    
    # Create a test image or use a file
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(test_image, "TEST - Run with webcam for real detection", 
                (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    full_view, left_wrist, right_wrist = detector.process_frame(test_image)
    
    cv2.imshow('Test', full_view)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Uncomment the line below to test with static image first
    # test_with_image()
    
    main()