import cv2
import json
import numpy as np

class ZoneSetupTool:
    """Interactive tool to define shelf zones by clicking on the frame"""
    
    def __init__(self, video_source):
        self.video_source = video_source
        self.zones = {}  # {zone_id: [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]}
        self.current_zone_id = 1
        self.current_points = []
        self.frame = None
        self.display_frame = None
        
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks to define zone corners"""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Add point
            self.current_points.append((x, y))
            print(f"Point {len(self.current_points)}: ({x}, {y})")
            
            # Draw point on display
            cv2.circle(self.display_frame, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(self.display_frame, str(len(self.current_points)), 
                       (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # If we have 4 points, draw the zone
            if len(self.current_points) == 4:
                self.draw_zone(self.display_frame, self.current_points, self.current_zone_id)
            
            cv2.imshow('Zone Setup', self.display_frame)
    
    def draw_zone(self, frame, points, zone_id):
        """Draw a zone on the frame"""
        if len(points) < 2:
            return
        
        # Draw lines between points
        pts = np.array(points, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], True, (0, 255, 255), 2)
        
        # Fill with semi-transparent color
        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts], (0, 255, 255))
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        # Add zone label
        center_x = int(np.mean([p[0] for p in points]))
        center_y = int(np.mean([p[1] for p in points]))
        cv2.putText(frame, f"Zone {zone_id}", (center_x - 30, center_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    def redraw_all_zones(self):
        """Redraw all defined zones on the display frame"""
        self.display_frame = self.frame.copy()
        
        for zone_id, points in self.zones.items():
            self.draw_zone(self.display_frame, points, zone_id)
        
        # Draw current points if any
        for i, point in enumerate(self.current_points):
            cv2.circle(self.display_frame, point, 5, (0, 255, 0), -1)
            cv2.putText(self.display_frame, str(i + 1), 
                       (point[0] + 10, point[1]), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (0, 255, 0), 2)
        
        if len(self.current_points) >= 2:
            # Draw lines between current points
            for i in range(len(self.current_points) - 1):
                cv2.line(self.display_frame, self.current_points[i], 
                        self.current_points[i + 1], (0, 255, 0), 2)
    
    def setup_zones(self):
        """Main function to run the interactive zone setup"""
        # Read first frame
        cap = cv2.VideoCapture(self.video_source)
        ret, self.frame = cap.read()
        cap.release()
        
        if not ret:
            print("Error: Could not read video")
            return None
        
        self.display_frame = self.frame.copy()
        
        # Create window and set mouse callback
        cv2.namedWindow('Zone Setup')
        cv2.setMouseCallback('Zone Setup', self.mouse_callback)
        
        print("\n" + "="*70)
        print("ZONE SETUP TOOL - INSTRUCTIONS")
        print("="*70)
        print("HOW TO DEFINE A ZONE (SHELF ROW):")
        print("1. Click on the TOP-LEFT corner of the shelf row")
        print("2. Click on the TOP-RIGHT corner")
        print("3. Click on the BOTTOM-RIGHT corner")
        print("4. Click on the BOTTOM-LEFT corner")
        print("\nThe 4 clicks define one zone in a clockwise order.")
        print("\nKEYBOARD COMMANDS:")
        print("  ENTER - Save current zone and start next zone")
        print("  'u'   - Undo last point")
        print("  'r'   - Reset current zone (clear all 4 points)")
        print("  'd'   - Delete last saved zone")
        print("  's'   - Save all zones to file and exit")
        print("  'q'   - Quit without saving")
        print("="*70)
        print(f"\nDefining Zone {self.current_zone_id}...")
        print("Click on the 4 corners of the shelf row (clockwise from top-left)\n")
        
        cv2.imshow('Zone Setup', self.display_frame)
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == 13:  # ENTER - Save current zone
                if len(self.current_points) == 4:
                    self.zones[self.current_zone_id] = self.current_points.copy()
                    print(f"✓ Zone {self.current_zone_id} saved: {self.current_points}")
                    self.current_zone_id += 1
                    self.current_points = []
                    self.redraw_all_zones()
                    cv2.imshow('Zone Setup', self.display_frame)
                    print(f"\nDefining Zone {self.current_zone_id}...")
                    print("Click on the 4 corners (clockwise from top-left)\n")
                else:
                    print(f"✗ Need 4 points to save zone. Current points: {len(self.current_points)}")
            
            elif key == ord('u'):  # Undo last point
                if self.current_points:
                    removed = self.current_points.pop()
                    print(f"✓ Removed point: {removed}")
                    self.redraw_all_zones()
                    cv2.imshow('Zone Setup', self.display_frame)
                else:
                    print("✗ No points to undo")
            
            elif key == ord('r'):  # Reset current zone
                if self.current_points:
                    print(f"✓ Reset current zone (removed {len(self.current_points)} points)")
                    self.current_points = []
                    self.redraw_all_zones()
                    cv2.imshow('Zone Setup', self.display_frame)
                else:
                    print("✗ No points to reset")
            
            elif key == ord('d'):  # Delete last saved zone
                if self.zones:
                    deleted_zone = self.current_zone_id - 1
                    if deleted_zone in self.zones:
                        del self.zones[deleted_zone]
                        self.current_zone_id -= 1
                        print(f"✓ Deleted Zone {deleted_zone}")
                        self.redraw_all_zones()
                        cv2.imshow('Zone Setup', self.display_frame)
                else:
                    print("✗ No zones to delete")
            
            elif key == ord('s'):  # Save and exit
                if self.current_points:
                    print("\n⚠ Warning: You have unsaved points. Press ENTER to save them first, or 'q' to discard.")
                else:
                    if self.zones:
                        self.save_zones()
                        print("\n✓ Zones saved successfully!")
                        break
                    else:
                        print("\n✗ No zones defined. Press 'q' to quit without saving.")
            
            elif key == ord('q'):  # Quit without saving
                print("\nQuitting without saving...")
                break
        
        cv2.destroyAllWindows()
        return self.zones
    
    def save_zones(self, filename='shelf_zones.json'):
        """Save zones to a JSON file"""
        with open(filename, 'w') as f:
            json.dump(self.zones, f, indent=2)
        print(f"\n✓ Zones saved to {filename}")
        print("\nSaved zones:")
        for zone_id, points in self.zones.items():
            print(f"  Zone {zone_id}: {points}")
    
    @staticmethod
    def load_zones(filename='shelf_zones.json'):
        """Load zones from a JSON file"""
        try:
            with open(filename, 'r') as f:
                zones = json.load(f)
            # Convert string keys back to integers
            zones = {int(k): v for k, v in zones.items()}
            print(f"✓ Loaded {len(zones)} zones from {filename}")
            return zones
        except FileNotFoundError:
            print(f"✗ File {filename} not found")
            return {}


def visualize_zones(video_source, zones_file='shelf_zones.json'):
    """Visualize saved zones on video"""
    zones = ZoneSetupTool.load_zones(zones_file)
    
    if not zones:
        print("No zones to visualize")
        return
    
    cap = cv2.VideoCapture(video_source)
    
    print("\nVisualizing zones... Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Draw all zones
        for zone_id, points in zones.items():
            pts = np.array(points, np.int32)
            pts = pts.reshape((-1, 1, 2))
            
            # Draw zone boundaries
            cv2.polylines(frame, [pts], True, (0, 255, 255), 2)
            
            # Fill with semi-transparent color
            overlay = frame.copy()
            cv2.fillPoly(overlay, [pts], (0, 255, 255))
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
            
            # Add zone label
            center_x = int(np.mean([p[0] for p in points]))
            center_y = int(np.mean([p[1] for p in points]))
            cv2.putText(frame, f"Zone {zone_id}", (center_x - 30, center_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.imshow('Zones Visualization', frame)
        
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Setup zones interactively
    video_path = "Videos\\Security_camera.mp4"
    
    print("Starting Zone Setup Tool...")
    setup_tool = ZoneSetupTool(video_path)
    zones = setup_tool.setup_zones()
    
    # Optionally visualize the saved zones
    if zones:
        print("\n" + "="*70)
        response = input("Do you want to visualize the zones on the video? (y/n): ")
        if response.lower() == 'y':
            visualize_zones(video_path)