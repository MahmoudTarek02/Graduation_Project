"""
hand_tracker.py
---------------
Abstract HandTrackerBase defines the contract.
MediaPipeHandTracker is the default implementation using the
MediaPipe Tasks API (mediapipe >= 0.10).

To swap models later, subclass HandTrackerBase and implement detect().
"""

from abc import ABC, abstractmethod
from pathlib import Path
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


# MediaPipe landmark indices
_WRIST      = 0
_INDEX_TIP  = 8
_MIDDLE_TIP = 12

# Auto-download URL for the official hand landmarker model
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)
_DEFAULT_MODEL_PATH = Path(__file__).parent / "hand_landmarker.task"


def _ensure_model(path: Path) -> str:
    if not path.exists():
        print(f"[MediaPipeHandTracker] Downloading hand landmarker model to {path} ...")
        urllib.request.urlretrieve(_MODEL_URL, path)
        print("[MediaPipeHandTracker] Download complete.")
    return str(path)


class HandTrackerBase(ABC):
    """
    Contract for all hand-tracking backends.

    detect(frame) must return a list of hand dicts, each with:
        'wrist'      : (x, y) pixel coords of the wrist joint
        'fingertip'  : (x, y) pixel coords of the reach point
        'handedness' : 'Left' or 'Right'
        'landmarks'  : list of (x, y) for all 21 joints
    """

    @abstractmethod
    def detect(self, frame) -> list[dict]:
        pass


class MediaPipeHandTracker(HandTrackerBase):
    """
    MediaPipe Hand Landmarker implementation (Tasks API, mediapipe >= 0.10).

    The model file is auto-downloaded on first run if not present.

    Parameters
    ----------
    model_path          : Path to .task model file (auto-downloaded if None)
    max_hands           : Max simultaneous hands
    detection_confidence: Min score to accept a detection
    tracking_confidence : Min score to keep tracking between frames
    use_fingertip       : 'middle' | 'index'  — landmark used as reach point
    """

    def __init__(
        self,
        model_path: str | None = None,
        max_hands: int = 4,
        detection_confidence: float = 0.5,
        tracking_confidence: float = 0.5,
        use_fingertip: str = "middle",
    ):
        resolved_path = _ensure_model(
            Path(model_path) if model_path else _DEFAULT_MODEL_PATH
        )

        base_options = mp_python.BaseOptions(model_asset_path=resolved_path)
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            # VIDEO mode gives temporal smoothing across frames
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_confidence,
            min_hand_presence_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )

        self._landmarker = mp_vision.HandLandmarker.create_from_options(options)
        self._tip_idx = _MIDDLE_TIP if use_fingertip == "middle" else _INDEX_TIP
        self._frame_ts_ms = 0   # monotonic timestamp required by VIDEO mode

        print(
            f"[MediaPipeHandTracker] init — max_hands={max_hands}, "
            f"det={detection_confidence}, trk={tracking_confidence}, "
            f"tip={'middle' if use_fingertip == 'middle' else 'index'}, "
            f"model={resolved_path}"
        )

    # ------------------------------------------------------------------
    def detect(self, frame) -> list[dict]:
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # VIDEO mode requires a strictly increasing timestamp (ms)
        self._frame_ts_ms += 33          # ~30 fps; exact gap doesn't matter
        result = self._landmarker.detect_for_video(mp_image, self._frame_ts_ms)

        hands = []
        for lms, handed in zip(result.hand_landmarks, result.handedness):
            def _px(lm):
                return (int(lm.x * w), int(lm.y * h))

            all_lm = [_px(lm) for lm in lms]

            hands.append(
                {
                    "wrist"      : all_lm[_WRIST],
                    "fingertip"  : all_lm[self._tip_idx],
                    "handedness" : handed[0].category_name,   # 'Left' | 'Right'
                    "landmarks"  : all_lm,
                }
            )

        return hands

    def close(self):
        self._landmarker.close()