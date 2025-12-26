import cv2
import torch
import numpy as np
from ultralytics import YOLO
from boxmot import ByteTrack
from pathlib import Path
from reid import REID
import operator
from datetime import datetime
import time


class PersonDetector:
    """Detects persons in frames using YOLO"""
    def __init__(self, model_path="yolov8n.pt", device="cuda"):
        print(f"Initializing Person Detector with model: {model_path}")
        self.model = YOLO(model_path)
        self.device = device
        print(f"Person Detector loaded successfully")

    def detect(self, frame):
        results = self.model(frame, verbose=False)[0]

        boxes = []
        for b in results.boxes:
            if b.cls[0] != 0:  # Only keep person class (class 0)
                continue
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            conf = float(b.conf[0])
            cls = int(b.cls[0])
            boxes.append([x1, y1, x2, y2, conf, cls])

        return np.array(boxes)


class PersonTracker:
    """Tracks persons across frames using ByteTrack"""
    def __init__(self, device="cuda"):
        print(f"Initializing Person Tracker on device: {device}")
        self.tracker = ByteTrack(device=device, half=False)
        print("Person Tracker initialized")

    def update(self, detections, frame):
        # INPUT: M x (x, y, x, y, conf, class)
        # OUTPUT: M x (x, y, x, y, id, conf, class, index)
        return self.tracker.update(detections, frame)

    def draw(self, frame):
        self.tracker.plot_results(frame, show_trajectories=True)


class PoseEstimator:
    """Estimates body pose keypoints using YOLO pose model"""
    def __init__(self, model_path='yolo11n-pose.pt'):
        print(f"Initializing Pose Estimator with model: {model_path}")
        try:
            self.model = YOLO(model_path)
            print(f"Pose Estimator loaded: {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Trying to download model...")
            self.model = YOLO('yolo11n-pose.pt')
    
    def estimate_pose(self, crop):
        """
        Estimate pose on a cropped person image
        Returns: keypoints array of shape (17, 3) where each row is (x, y, confidence)
        """
        if crop is None or crop.size == 0:
            return None
        
        try:
            results = self.model(crop, verbose=False, conf=0.3)
            
            if len(results) > 0 and hasattr(results[0], 'keypoints') and results[0].keypoints is not None:
                keypoints = results[0].keypoints.data.cpu().numpy()
                if len(keypoints) > 0:
                    return keypoints[0]  # Return first person's keypoints
            
            return None
        except Exception as e:
            print(f"Pose estimation error: {e}")
            return None
    
    def get_wrist_positions(self, keypoints):
        """
        Extract wrist positions from keypoints
        Returns: (left_wrist, right_wrist) as (x, y) tuples or None
        Keypoint indices: 9 = left wrist, 10 = right wrist
        """
        if keypoints is None or len(keypoints) < 11:
            return None, None
        
        left_wrist = None
        right_wrist = None
        
        # Left wrist (index 9)
        if keypoints[9][2] > 0.3:  # confidence threshold
            left_wrist = (int(keypoints[9][0]), int(keypoints[9][1]))
        
        # Right wrist (index 10)
        if keypoints[10][2] > 0.3:  # confidence threshold
            right_wrist = (int(keypoints[10][0]), int(keypoints[10][1]))
        
        return left_wrist, right_wrist
    
    def draw_skeleton(self, frame, keypoints, bbox_offset=(0, 0), color=(0, 255, 0)):
        """
        Draw pose skeleton on frame
        bbox_offset: (x_offset, y_offset) to adjust keypoints to original frame coordinates
        """
        if keypoints is None:
            return frame
        
        x_offset, y_offset = bbox_offset
        
        # YOLO pose skeleton connections
        skeleton = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
            (5, 11), (6, 12), (11, 12),  # Torso
            (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
        ]
        
        # Draw connections
        for start_idx, end_idx in skeleton:
            if start_idx < len(keypoints) and end_idx < len(keypoints):
                start_point = keypoints[start_idx]
                end_point = keypoints[end_idx]
                
                if start_point[2] > 0.3 and end_point[2] > 0.3:
                    start_pos = (int(start_point[0]) + x_offset, int(start_point[1]) + y_offset)
                    end_pos = (int(end_point[0]) + x_offset, int(end_point[1]) + y_offset)
                    cv2.line(frame, start_pos, end_pos, color, 2)
        
        # Draw keypoints
        for i, kp in enumerate(keypoints):
            if kp[2] > 0.3:
                x, y = int(kp[0]) + x_offset, int(kp[1]) + y_offset
                
                # Different colors for wrists
                if i == 9:  # Left wrist
                    point_color = (255, 0, 0)  # Blue
                elif i == 10:  # Right wrist
                    point_color = (0, 0, 255)  # Red
                else:
                    point_color = color
                
                cv2.circle(frame, (x, y), 4, point_color, -1)
        
        return frame


class ReIDFusionManager:
    """Manages person re-identification and ID fusion"""
    def __init__(self, threshold=600):
        print(f"Initializing ReID Fusion Manager with threshold: {threshold}")
        self.reid = REID()
        self.threshold = threshold

        self.images_by_id = {}  # crops
        self.feats = {}         # extracted features
        self.exist_ids = set()
        self.final_fuse_id = {}
        print("ReID Fusion Manager initialized")

    def process(self, res, frame):
        # Collect crops and features
        for row in res:
            tid = int(row[4])
            x1, y1, x2, y2 = map(int, row[:4])

            crop = frame[y1:y2, x1:x2]
            if crop is None or crop.size == 0:
                continue

            if tid not in self.images_by_id:
                self.images_by_id[tid] = []
                self.feats[tid] = []

            self.images_by_id[tid].append(crop)
            self.feats[tid].append(self.reid._feature(crop))

        # Start ID-fusion logic
        if len(res) == 0:
            return self.final_fuse_id

        current_ids = set(res[:, 4].astype(int))

        if len(self.exist_ids) == 0:
            for i in current_ids:
                self.final_fuse_id[i] = [i]
            self.exist_ids = current_ids
            print(f"Initial IDs detected: {self.final_fuse_id}")
            return self.final_fuse_id

        new_ids = current_ids - self.exist_ids

        for nid in new_ids:
            if len(self.images_by_id[nid]) < 10:
                continue

            # Add to exist_ids only when we have enough images
            self.exist_ids.add(nid)

            dis = []
            unpickable = []

            # Build unpickable IDs
            for i in current_ids:
                for key, group in self.final_fuse_id.items():
                    if i in group:
                        unpickable += group

            candidates = (self.exist_ids - set(unpickable)) & set(self.final_fuse_id.keys())
            
            print(f'exist_ids: {self.exist_ids}, unpickable: {unpickable}')

            for oid in candidates:
                d = np.median(
                    self.reid.compute_distance(
                        torch.cat(self.feats[nid], 0),
                        torch.cat(self.feats[oid], 0)
                    )
                )
                print(f'nid {nid}, oid {oid}, distance {d:.2f}')
                dis.append([oid, d])

            if not dis:
                self.final_fuse_id[nid] = [nid]
                print(f"New ID {nid} created (no candidates)")
                continue

            dis.sort(key=operator.itemgetter(1))

            if dis[0][1] < self.threshold:
                combined_id = dis[0][0]
                self.images_by_id[combined_id] += self.images_by_id[nid]
                self.final_fuse_id[combined_id].append(nid)
                print(f"ID {nid} merged with ID {combined_id} (distance: {dis[0][1]:.2f})")
            else:
                self.final_fuse_id[nid] = [nid]
                print(f"New ID {nid} created (distance too large: {dis[0][1]:.2f})")

        print(f"Current ID fusion: {self.final_fuse_id}")
        
        return self.final_fuse_id


class IntegratedTrackingSystem:
    """Main system integrating detection, tracking, pose estimation and ReID"""
    def __init__(self, src=0, draw_poses=True):
        # Print startup info
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        print(datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-5])
        print("="*60)
        
        self.cap = cv2.VideoCapture(src)
        self.device = device
        self.draw_poses = draw_poses

        # Initialize all components
        self.detector = PersonDetector(device=self.device)
        self.tracker = PersonTracker(device=self.device)
        self.pose_estimator = PoseEstimator()
        self.reid_manager = ReIDFusionManager()
        
        print("="*60)
        print("Integrated System initialization complete!")
        print("="*60)

    def process_frame(self, frame):
        """
        Process a single frame through the entire pipeline
        Returns: 
            - frame: visualized frame
            - fuse_ids: current ID fusion mapping
            - tracks_with_poses: list of dicts with track info and pose data
        """
        # Step 1: Detect persons
        detections = self.detector.detect(frame)
        
        # Step 2: Track persons
        tracks = self.tracker.update(detections, frame)
        
        # Step 3: Estimate pose for each tracked person
        tracks_with_poses = []
        
        for track in tracks:
            x1, y1, x2, y2 = map(int, track[:4])
            track_id = int(track[4])
            
            # Crop person from frame
            person_crop = frame[y1:y2, x1:x2]
            
            # Estimate pose on the crop
            pose_keypoints = self.pose_estimator.estimate_pose(person_crop)
            
            # Get wrist positions if pose was detected
            left_wrist, right_wrist = None, None
            if pose_keypoints is not None:
                left_wrist, right_wrist = self.pose_estimator.get_wrist_positions(pose_keypoints)
            
            # Store track with pose data
            track_data = {
                'bbox': (x1, y1, x2, y2),
                'track_id': track_id,
                'confidence': float(track[5]),
                'pose_keypoints': pose_keypoints,
                'left_wrist': left_wrist,
                'right_wrist': right_wrist
            }
            tracks_with_poses.append(track_data)
            
            # Draw pose skeleton if enabled
            if self.draw_poses and pose_keypoints is not None:
                self.pose_estimator.draw_skeleton(
                    frame, 
                    pose_keypoints, 
                    bbox_offset=(x1, y1),
                    color=(0, 255, 0)
                )
        
        # Step 4: ReID fusion
        fuse_ids = self.reid_manager.process(tracks, frame)
        
        # Step 5: Draw tracking visualization
        self.tracker.draw(frame)
        
        return frame, fuse_ids, tracks_with_poses

    def start(self):
        """Start the integrated tracking and pose estimation system"""
        print("\nStarting integrated tracking system...")
        print("Press 'q' to quit")
        print("Press 'p' to toggle pose visualization\n")
        
        with torch.inference_mode():
            frame_count = 0
            while True:
                t1 = time.time()
                ok, frame = self.cap.read()
                if not ok:
                    print("\nEnd of video stream")
                    break

                # Process frame through entire pipeline
                disp, fuse_ids, tracks_with_poses = self.process_frame(frame)
                
                frame_count += 1
                t2 = time.time()
                fps = 1 / (t2 - t1)
                
                # Display info on frame
                info_text = f"Frame: {frame_count} | FPS: {fps:.2f} | Persons: {len(tracks_with_poses)}"
                cv2.putText(disp, info_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                print(f"Frame: {frame_count}, FPS: {fps:.2f}, Tracked: {len(tracks_with_poses)}", end='\r')
                
                cv2.imshow("Integrated Tracking + Pose System", disp)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print(f"\nUser quit after {frame_count} frames")
                    break
                elif key == ord('p'):
                    self.draw_poses = not self.draw_poses
                    print(f"\nPose visualization: {'ON' if self.draw_poses else 'OFF'}")

        self.cap.release()
        cv2.destroyAllWindows()
        
        print("\n" + "="*60)
        print("Final Fused IDs:")
        print(self.reid_manager.final_fuse_id)
        print(f"Total unique persons detected: {len(self.reid_manager.final_fuse_id)}")
        print("="*60)


if __name__ == "__main__":
    system = IntegratedTrackingSystem(src="Videos\\Security_camera.mp4", draw_poses=True)
    system.start()