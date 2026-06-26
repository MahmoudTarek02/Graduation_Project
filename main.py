from __future__ import annotations

import argparse
import time
from collections import Counter, deque
from pathlib import Path

import cv2

from config import MAX_CAMERA_INDEX_TO_PROBE, SHELF_ITEM_CONFIDENCE_THRESHOLD, SHELF_ITEM_MODEL_PATH
from hand_tracker import MediaPipeHandTracker
from shelf_checkout import ShelfItemMonitor


def _parse_source(value: str) -> int | str:
    try:
        return int(value)
    except ValueError:
        return str(Path(value).expanduser())


class CameraSelector:
    WINDOW_NAME = "Camera Preview - Y: select, N: next, Q: quit"

    def __init__(self, max_index: int = MAX_CAMERA_INDEX_TO_PROBE):
        self.max_index = max_index

    def choose_camera(self, label: str, excluded: set[int] | None = None) -> int:
        excluded = excluded or set()
        print(f"\nSearching camera indices for {label}...")
        for index in range(self.max_index + 1):
            if index in excluded:
                continue
            print(f"[CameraSelector] Checking camera {index} for {label}...")
            result = self._preview_and_confirm(index, label)
            if result is True:
                print(f"[CameraSelector] {label} -> camera {index}")
                return index
            if result is None:
                raise RuntimeError("Camera selection aborted by user.")
        raise RuntimeError(f"No available cameras found for {label}.")

    def _preview_and_confirm(self, index: int, label: str) -> bool | None:
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            cap.release()
            return False

        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)
        result = False
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            preview = frame.copy()
            self._draw_hud(
                preview,
                [
                    f"{label}: camera {index}",
                    "Press Y to select this camera",
                    "Press N to try the next camera",
                    "Press Q to abort",
                ],
            )
            cv2.imshow(self.WINDOW_NAME, preview)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("y"), ord("Y")):
                result = True
                break
            if key in (ord("n"), ord("N")):
                break
            if key in (ord("q"), ord("Q"), 27):
                result = None
                break

        cap.release()
        cv2.destroyWindow(self.WINDOW_NAME)
        return result

    @staticmethod
    def _draw_hud(frame, lines):
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 110), (20, 20, 20), -1)
        for i, line in enumerate(lines):
            y = 28 + i * 24
            cv2.putText(
                frame,
                line,
                (12, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (240, 240, 240),
                2,
            )


class ForwardCameraSession:
    WINDOW_NAME = "Forward Camera Trigger"

    def __init__(self, source: int | str):
        self.source = source
        self.cap: cv2.VideoCapture | None = None
        self.active = False

    def start(self) -> bool:
        if self.active and self.cap is not None and self.cap.isOpened():
            return True

        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            self.cap.release()
            self.cap = None
            print(f"[ForwardCamera] Could not open source {self.source}")
            return False

        self.active = True
        print("[ForwardCamera] Triggered by hand detection.")
        return True

    def show_frame(self) -> None:
        if not self.active or self.cap is None:
            return

        ok, frame = self.cap.read()
        if not ok:
            self.stop()
            return

        preview = frame.copy()
        cv2.rectangle(preview, (0, 0), (preview.shape[1], 70), (20, 20, 20), -1)
        cv2.putText(
            preview,
            f"Forward camera active | source {self.source}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (240, 240, 240),
            2,
        )
        cv2.putText(
            preview,
            "Stops when no hand is detected on shelf camera",
            (12, 56),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (120, 220, 255),
            2,
        )
        cv2.imshow(self.WINDOW_NAME, preview)

    def stop(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        if self.active:
            print("[ForwardCamera] Stopped.")
        self.active = False
        try:
            cv2.destroyWindow(self.WINDOW_NAME)
        except Exception:
            pass


def _rolling_counts(history: deque[Counter]) -> dict[str, int]:
    if not history:
        return {}

    totals = Counter()
    for frame_counts in history:
        for cls, count in frame_counts.items():
            totals[cls] += count

    n = len(history)
    stable_counts: dict[str, int] = {}
    for cls, total in totals.items():
        value = round(total / n)
        if value > 0:
            stable_counts[cls] = value
    return stable_counts


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Shelf camera always on; trigger forward camera only while a hand is detected"
    )
    parser.add_argument("shelf_source", nargs="?", help="Item-facing camera index or video path")
    parser.add_argument("forward_source", nargs="?", help="Forward-facing camera index or video path")
    parser.add_argument(
        "--settle-frames",
        type=int,
        default=5,
        help="Consecutive stable frames required after the hand leaves before cart updates commit",
    )
    args = parser.parse_args()

    if args.shelf_source is None and args.forward_source is None:
        selector = CameraSelector()
        forward_source = selector.choose_camera(label="Forward-facing camera")
        shelf_source = selector.choose_camera(label="Item-facing camera", excluded={forward_source})
    elif args.shelf_source is not None and args.forward_source is not None:
        shelf_source = _parse_source(args.shelf_source)
        forward_source = _parse_source(args.forward_source)
    else:
        parser.error("Provide both shelf_source and forward_source, or provide neither to choose cameras interactively.")

    print("=" * 60)
    print("Shelf Camera + Hand-Triggered Forward Camera")
    print("=" * 60)
    print(f"Loading YOLO model: {SHELF_ITEM_MODEL_PATH}...")

    monitor = ShelfItemMonitor(
        model_path=SHELF_ITEM_MODEL_PATH,
        conf_threshold=SHELF_ITEM_CONFIDENCE_THRESHOLD,
        show_preview=False,
    )
    monitor._ensure_models()

    print("Loading MediaPipe Hand Tracker...")
    hand_tracker = MediaPipeHandTracker()
    forward_session = ForwardCameraSession(forward_source)

    print(f"\nOpening item-facing source: {shelf_source}...")
    cap = cv2.VideoCapture(shelf_source)
    if not cap.isOpened():
        print(f"Error: Could not open source {shelf_source}")
        hand_tracker.close()
        return 1

    print("\nSources ready.")
    print("Instructions:")
    print("  - Item-facing camera stays on continuously.")
    print("  - When a hand is detected, the forward-facing camera opens.")
    print("  - When no hand is detected, the forward-facing camera stops.")
    print("  - Press 'q' in either window to quit.")
    print("-" * 60)

    window_size = 15
    required_settle_frames = max(1, args.settle_frames)
    counts_history: deque[Counter] = deque(maxlen=window_size)
    current_stable_counts: dict[str, int] = {}
    last_stable_inventory: dict[str, int] = {}
    prev_frame_stable_counts: dict[str, int] = {}
    in_interaction = False
    settle_counter = 0

    shelf_window_name = "Shelf Cart Monitor"
    cv2.namedWindow(shelf_window_name, cv2.WINDOW_NORMAL)
    frame_count = 0
    start_time = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("\nEnd of video source reached.")
                break

            detections = monitor._detect_items(frame)
            hands = hand_tracker.detect(frame)
            hand_present = len(hands) > 0

            if hand_present:
                forward_session.start()
            else:
                forward_session.stop()

            frame_counts = Counter()
            for det in detections:
                cls = int(det[5])
                class_name = monitor._class_names.get(cls, str(cls))
                frame_counts[class_name] += 1

            counts_history.append(frame_counts)
            rolling_stable_counts = _rolling_counts(counts_history)

            status_text = "Stable (Ready)"
            status_color = (0, 255, 0)

            if hand_present:
                status_text = "Interacting (Hand Present - Forward Camera Active)"
                status_color = (0, 100, 255)
                if not in_interaction:
                    in_interaction = True
                    last_stable_inventory = dict(current_stable_counts)
                    timestamp = time.strftime("%H:%M:%S")
                    print(f"\n[{timestamp}] [SHOPPER INTERACTING] Hand entered shelf.")
                    print(f"  Baseline Inventory: {last_stable_inventory}")
                settle_counter = 0
            elif in_interaction:
                status_text = f"Settling counts ({settle_counter}/{required_settle_frames})"
                status_color = (255, 255, 0)

                if rolling_stable_counts == prev_frame_stable_counts:
                    settle_counter += 1
                else:
                    settle_counter = 0

                if settle_counter >= required_settle_frames:
                    in_interaction = False
                    settle_counter = 0
                    timestamp = time.strftime("%H:%M:%S")
                    all_keys = set(last_stable_inventory.keys()).union(rolling_stable_counts.keys())
                    cart_changes = []

                    for cls in sorted(all_keys):
                        prev = last_stable_inventory.get(cls, 0)
                        curr = rolling_stable_counts.get(cls, 0)
                        diff = curr - prev
                        if diff < 0:
                            cart_changes.append(f"Taken: {abs(diff)}x {cls}")
                        elif diff > 0:
                            cart_changes.append(f"Returned: {diff}x {cls}")

                    if cart_changes:
                        print(f"\n[{timestamp}] [CART ASSIGNMENT COMMIT] Cart updated:")
                        for change in cart_changes:
                            print(f"  * {change}")
                    else:
                        print(f"\n[{timestamp}] [CART ASSIGNMENT COMMIT] Cart unchanged.")

                    current_stable_counts = dict(rolling_stable_counts)
                    print(f"  Current Shelf Inventory: {current_stable_counts if current_stable_counts else 'Empty'}\n")
            else:
                current_stable_counts = dict(rolling_stable_counts)

            prev_frame_stable_counts = dict(rolling_stable_counts)

            preview = frame.copy()
            monitor._draw_detections(preview, detections)

            for hand in hands:
                for landmark in hand["landmarks"]:
                    cv2.circle(preview, landmark, 3, (0, 0, 255), -1)
                cv2.circle(preview, hand["wrist"], 8, (0, 140, 255), -1)

            cv2.rectangle(preview, (0, 0), (500, 160), (20, 20, 20), -1)
            cv2.putText(
                preview,
                "Stable Shelf Inventory:",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (50, 255, 50),
                2,
            )
            cv2.putText(
                preview,
                f"Status: {status_text}",
                (10, 145),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                status_color,
                2,
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
                    1,
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
                        2,
                    )
                    y_offset += 25

            frame_count += 1
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0.0
            cv2.putText(
                preview,
                f"FPS: {fps:.1f}",
                (preview.shape[1] - 100, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
            )

            cv2.imshow(shelf_window_name, preview)
            forward_session.show_frame()

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("\nUser requested exit.")
                break
    finally:
        cap.release()
        hand_tracker.close()
        monitor.close()
        forward_session.stop()
        cv2.destroyAllWindows()

    print("=" * 60)
    print("Session ended.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
