from __future__ import annotations

import argparse
import time
from collections import Counter, deque
from pathlib import Path

import cv2

from config import (
    MAX_CAMERA_INDEX_TO_PROBE,
    SHELF_ITEM_CONFIDENCE_THRESHOLD,
    SHELF_ITEM_MODEL_PATH,
)
from hand_tracker import MediaPipeHandTracker
from shelf_checkout import ShelfItemMonitor
from shelf_identity_resolver import ShelfIdentityResolver


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
            lines = [
                f"{label}: camera {index}",
                "Press Y to select this camera",
                "Press N to try the next camera",
                "Press Q to abort",
            ]
            self._draw_hud(preview, lines)
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


def _rolling_counts(history: deque[Counter], class_names: dict) -> dict[str, int]:
    del class_names
    if not history:
        return {}

    totals = Counter()
    for frame_counts in history:
        for cls, count in frame_counts.items():
            totals[cls] += count

    n = len(history)
    stable_counts = {}
    for cls, total in totals.items():
        value = round(total / n)
        if value > 0:
            stable_counts[cls] = value
    return stable_counts


def _resolve_cart_update(
    resolver: ShelfIdentityResolver,
    forward_source: int | str,
    actor_side: str,
    class_name: str,
    quantity: int,
) -> None:
    result = resolver.resolve_event(
        {
            "class_name": class_name,
            "quantity": quantity,
            "frame_taken": 0,
            "actor_side": actor_side,
        },
        source=forward_source,
    )

    print(
        f"    forward_result: event={result.event_type} "
        f"class={class_name} "
        f"quantity={quantity} "
        f"person_id={result.person_id} "
        f"score={result.score} "
        f"method={result.selection_method} "
        f"person_count={result.person_count}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run shelf-camera cart detection and trigger the forward camera on committed cart updates"
    )
    parser.add_argument("shelf_source", nargs="?", help="Item-facing camera index or video path")
    parser.add_argument("forward_source", nargs="?", help="Forward-facing camera index or video path")
    parser.add_argument(
        "--actor-side",
        default="unknown-side",
        help="Optional shelf-side hint attached to generated forward-camera events",
    )
    parser.add_argument(
        "--settle-frames",
        type=int,
        default=5,
        help="Number of consecutive stable frames required after the hand leaves before committing cart updates",
    )
    args = parser.parse_args()

    if args.shelf_source is None and args.forward_source is None:
        selector = CameraSelector()
        forward_source = selector.choose_camera(label="Forward-facing camera")
        shelf_source = selector.choose_camera(
            label="Item-facing camera",
            excluded={forward_source},
        )
    elif args.shelf_source is not None and args.forward_source is not None:
        shelf_source = _parse_source(args.shelf_source)
        forward_source = _parse_source(args.forward_source)
    else:
        parser.error("Provide both shelf_source and forward_source, or provide neither to choose cameras interactively.")

    print("=" * 60)
    print("Shelf Cart Monitor + Forward Camera Trigger")
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
    resolver = ShelfIdentityResolver(show_trigger_preview=True)

    print(f"\nOpening item-facing source: {shelf_source}...")
    cap = cv2.VideoCapture(shelf_source)
    if not cap.isOpened():
        print(f"Error: Could not open source {shelf_source}")
        hand_tracker.close()
        return 1

    print("\nSources ready.")
    print("Instructions:")
    print("  - Interact with items in front of the item-facing camera.")
    print("  - While your hand is present, cart updates are paused.")
    print("  - After your hand leaves and counts settle, cart changes are committed.")
    print("  - Each committed change triggers the forward-facing camera.")
    print("  - Press 'q' in the window to quit.")
    print("-" * 60)

    window_size = 15
    required_settle_frames = max(1, args.settle_frames)
    counts_history: deque[Counter] = deque(maxlen=window_size)
    current_stable_counts: dict[str, int] = {}
    last_stable_inventory: dict[str, int] = {}
    prev_frame_stable_counts: dict[str, int] = {}
    in_interaction = False
    settle_counter = 0

    window_name = "Shelf Cart Monitor"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
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

            frame_counts = Counter()
            for det in detections:
                cls = int(det[5])
                class_name = monitor._class_names.get(cls, str(cls))
                frame_counts[class_name] += 1

            counts_history.append(frame_counts)
            rolling_stable_counts = _rolling_counts(counts_history, monitor._class_names)

            status_text = "Stable (Ready)"
            status_color = (0, 255, 0)

            if hand_present:
                status_text = "Interacting (Hand Present - Cart Paused)"
                status_color = (0, 100, 255)
                if not in_interaction:
                    in_interaction = True
                    last_stable_inventory = dict(current_stable_counts)
                    timestamp = time.strftime("%H:%M:%S")
                    print(f"\n[{timestamp}] [SHOPPER INTERACTING] Hand entered shelf. Pausing cart updates...")
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
                    committed_changes: list[tuple[str, int]] = []

                    for cls in sorted(all_keys):
                        prev = last_stable_inventory.get(cls, 0)
                        curr = rolling_stable_counts.get(cls, 0)
                        diff = curr - prev
                        if diff == 0:
                            continue
                        quantity = -diff
                        committed_changes.append((cls, quantity))

                    if committed_changes:
                        print(f"\n[{timestamp}] [CART ASSIGNMENT COMMIT] Cart updated:")
                        for cls, quantity in committed_changes:
                            if quantity > 0:
                                print(f"  * Taken: {quantity}x {cls}")
                            else:
                                print(f"  * Returned: {abs(quantity)}x {cls}")

                        print("  Triggering forward-facing camera...")
                        for cls, quantity in committed_changes:
                            _resolve_cart_update(
                                resolver=resolver,
                                forward_source=forward_source,
                                actor_side=args.actor_side,
                                class_name=cls,
                                quantity=quantity,
                            )
                    else:
                        print(f"\n[{timestamp}] [CART ASSIGNMENT COMMIT] Cart unchanged (occlusion cleared successfully).")

                    current_stable_counts = dict(rolling_stable_counts)
                    current_inventory = dict(current_stable_counts) if current_stable_counts else "Empty"
                    print(f"  Current Shelf Inventory: {current_inventory}\n")
            else:
                current_stable_counts = dict(rolling_stable_counts)

            prev_frame_stable_counts = dict(rolling_stable_counts)

            preview = frame.copy()
            monitor._draw_detections(preview, detections)

            for hand in hands:
                for landmark in hand["landmarks"]:
                    cv2.circle(preview, landmark, 3, (0, 0, 255), -1)
                cv2.circle(preview, hand["wrist"], 8, (0, 140, 255), -1)

            cv2.rectangle(preview, (0, 0), (450, 160), (20, 20, 20), -1)
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

            cv2.imshow(window_name, preview)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\nUser requested exit.")
                break
    finally:
        cap.release()
        hand_tracker.close()
        monitor.close()
        cv2.destroyAllWindows()

    print("=" * 60)
    print("Session ended.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
