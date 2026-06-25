import cv2
import numpy as np
import time
from collections import Counter, deque
from shelf_checkout import ShelfItemMonitor
from config import SHELF_ITEM_MODEL_PATH, SHELF_ITEM_CONFIDENCE_THRESHOLD
from hand_tracker import MediaPipeHandTracker

def main():
    print("=" * 60)
    print("Standalone Shelf Item Detection & Counting Tester")
    print("  (With Occlusion Prevention & Cart Assignment Simulation)")
    print("=" * 60)
    print(f"Loading YOLO model: {SHELF_ITEM_MODEL_PATH}...")
    
    # Initialize ShelfItemMonitor
    monitor = ShelfItemMonitor(
        model_path=SHELF_ITEM_MODEL_PATH,
        conf_threshold=SHELF_ITEM_CONFIDENCE_THRESHOLD,
        show_preview=False  # We will handle previewing ourselves in this script
    )
    monitor._ensure_models()
    
    # Initialize MediaPipe Hand Tracker
    print("Loading MediaPipe Hand Tracker...")
    hand_tracker = MediaPipeHandTracker()
    
    # Ask the user for the camera index or video file path
    print("\nEnter camera index (e.g. 0, 1) or path to video file:")
    source_input = input("Source [default: 0]: ").strip()
    if not source_input:
        source = 0
    else:
        try:
            source = int(source_input)
        except ValueError:
            source = source_input
            
    print(f"\nOpening source: {source}...")
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Could not open source {source}")
        return

    print("\nSource opened successfully!")
    print("Instructions:")
    print("  - Place items on the shelf or remove them.")
    print("  - Place your hand in front of the camera (occluding items if you wish).")
    print("  - Cart assignment will pause while your hand is visible.")
    print("  - Once you remove your hand, wait 15 frames for counts to settle.")
    print("  - The console will commit and print the final cart changes.")
    print("  - Press 'q' in the window to quit.")
    print("-" * 60)
    
    # Rolling window parameters for count stabilization
    WINDOW_SIZE = 15
    counts_history = deque(maxlen=WINDOW_SIZE)
    
    # State machine variables
    current_stable_counts = {}
    last_stable_inventory = {}
    in_interaction = False
    settle_counter = 0
    REQUIRED_SETTLE_FRAMES = 15
    prev_frame_stable_counts = {}
    
    window_name = "Standalone Shelf Detector Tester"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # Track FPS
    frame_count = 0
    start_time = time.time()
    
    while True:
        ok, frame = cap.read()
        if not ok:
            print("\nEnd of video source reached.")
            break
            
        # Detect items
        detections = monitor._detect_items(frame)
        
        # Detect hands
        hands = hand_tracker.detect(frame)
        hand_present = len(hands) > 0
        
        # Count items in this frame
        frame_counts = Counter()
        for det in detections:
            cls = int(det[5])
            class_name = monitor._class_names.get(cls, str(cls))
            frame_counts[class_name] += 1
            
        # Add to history
        counts_history.append(frame_counts)
        
        # Calculate rolling stable counts (average rounded to nearest int)
        rolling_stable_counts = {}
        if counts_history:
            totals = Counter()
            for f_c in counts_history:
                for cls, count in f_c.items():
                    totals[cls] += count
            n = len(counts_history)
            for cls, total in totals.items():
                val = round(total / n)
                if val > 0:
                    rolling_stable_counts[cls] = val
                    
        # State machine logic
        status_text = "Stable (Ready)"
        status_color = (0, 255, 0)
        
        if hand_present:
            status_text = "Interacting (Hand Present - Cart Paused)"
            status_color = (0, 100, 255)  # Orange/Red in BGR
            if not in_interaction:
                in_interaction = True
                last_stable_inventory = dict(current_stable_counts)
                print(f"\n[{time.strftime('%H:%M:%S')}] [SHOPPER INTERACTING] Hand entered shelf. Pausing cart updates...")
                print(f"  Baseline Inventory: {last_stable_inventory}")
            settle_counter = 0
            
        elif in_interaction:
            # Hand was removed, waiting for counts to settle
            status_text = f"Settling counts ({settle_counter}/{REQUIRED_SETTLE_FRAMES})"
            status_color = (255, 255, 0)  # Cyan/Yellow in BGR
            
            # Check if rolling counts have stabilized (same as previous frame)
            if rolling_stable_counts == prev_frame_stable_counts:
                settle_counter += 1
            else:
                settle_counter = 0
                
            if settle_counter >= REQUIRED_SETTLE_FRAMES:
                # Settle duration met! Commit cart changes
                in_interaction = False
                settle_counter = 0
                
                all_keys = set(last_stable_inventory.keys()).union(rolling_stable_counts.keys())
                cart_changes = []
                for cls in all_keys:
                    prev = last_stable_inventory.get(cls, 0)
                    curr = rolling_stable_counts.get(cls, 0)
                    diff = curr - prev
                    if diff != 0:
                        if diff < 0:
                            cart_changes.append(f"Taken: {abs(diff)}x {cls}")
                        else:
                            cart_changes.append(f"Returned: {diff}x {cls}")
                            
                timestamp = time.strftime("%H:%M:%S")
                if cart_changes:
                    print(f"\n[{timestamp}] [CART ASSIGNMENT COMMIT] Cart updated:")
                    for change in cart_changes:
                        print(f"  * {change}")
                else:
                    print(f"\n[{timestamp}] [CART ASSIGNMENT COMMIT] Cart unchanged (occlusion cleared successfully).")
                    
                current_stable_counts = dict(rolling_stable_counts)
                print(f"  Current Shelf Inventory: {dict(current_stable_counts) if current_stable_counts else 'Empty'}\n")
                
        else:
            # Idle/Stable state
            current_stable_counts = dict(rolling_stable_counts)
            
        prev_frame_stable_counts = dict(rolling_stable_counts)
            
        # Draw HUD & bounding boxes on frame
        preview = frame.copy()
        
        # Draw detections
        for row in detections:
            x1, y1, x2, y2 = map(int, row[:4])
            conf = float(row[4])
            cls = int(row[5])
            class_name = monitor._class_names.get(cls, str(cls))
            cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 210, 80), 2)
            cv2.putText(
                preview,
                f"{class_name} {conf:.2f}",
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )
            
        # Draw hand markers if present
        for hand in hands:
            for lm in hand["landmarks"]:
                cv2.circle(preview, lm, 3, (0, 0, 255), -1)
            cv2.circle(preview, hand["wrist"], 8, (0, 140, 255), -1)
            
        # Draw current stable count HUD
        cv2.rectangle(preview, (0, 0), (450, 160), (20, 20, 20), -1)
        cv2.putText(
            preview,
            "Stable Shelf Inventory:",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (50, 255, 50),
            2
        )
        
        cv2.putText(
            preview,
            f"Status: {status_text}",
            (10, 145),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            status_color,
            2
        )
        
        y_offset = 55
        if not current_stable_counts:
            cv2.putText(
                preview,
                "No items on shelf",
                (15, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (180, 180, 180),
                1
            )
        else:
            for cls, qty in current_stable_counts.items():
                cv2.putText(
                    preview,
                    f"- {cls}: {qty}",
                    (15, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2
                )
                y_offset += 25
                
        # Draw FPS
        frame_count += 1
        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0
        cv2.putText(
            preview,
            f"FPS: {fps:.1f}",
            (preview.shape[1] - 100, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2
        )
        
        cv2.imshow(window_name, preview)
        
        # Break loop on 'q' key
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("\nUser requested exit.")
            break
            
    cap.release()
    hand_tracker.close()
    cv2.destroyAllWindows()
    print("=" * 60)
    print("Testing session ended.")
    print("=" * 60)

if __name__ == "__main__":
    main()
