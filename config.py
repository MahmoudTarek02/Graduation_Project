from pathlib import Path


# Back-camera detector model path used for person detection.
BACK_DETECTOR_MODEL_PATH = "yolo26n.pt"

# Shelf item detector model path used to recognize products on the shelf.
SHELF_ITEM_MODEL_PATH = "my_model_new.pt"

# Pose model path used on shelf cameras to estimate wrists for multi-person attribution.
SHELF_POSE_MODEL_PATH = "yolo26s-pose.pt"

# ReID model weights path used to merge tracker IDs that belong to the same person.
REID_WEIGHTS_PATH = "osnet_x0_25_msmt1.pth"

# MediaPipe hand landmarker model path used for back-camera hand tracking.
HAND_LANDMARKER_MODEL_PATH = Path(__file__).parent / "hand_landmarker.task"

# Maximum camera index to probe during manual camera selection.
MAX_CAMERA_INDEX_TO_PROBE = 8

# Number of frames to analyze from a live shelf camera after a shelf event.
LIVE_SHELF_ANALYSIS_FRAMES = 60

# Fallback FPS used when a live camera does not report a valid frame rate.
DEFAULT_LIVE_CAMERA_FPS = 30.0

# Detection confidence threshold for shelf item detection.
SHELF_ITEM_CONFIDENCE_THRESHOLD = 0.35

# Detection confidence threshold for shelf pose estimation.
SHELF_POSE_CONFIDENCE_THRESHOLD = 0.35

# Maximum pixel distance between a detected wrist and an item box to count as a touch.
SHELF_HAND_TOUCH_MARGIN = 60

# Number of frames to keep the last wrist-to-item touch association alive.
SHELF_HAND_TOUCH_TTL_FRAMES = 50

# COCO keypoint index used for the left wrist in the pose model output.
POSE_LEFT_WRIST_INDEX = 9

# COCO keypoint index used for the right wrist in the pose model output.
POSE_RIGHT_WRIST_INDEX = 10

# Number of frames an item must remain visible before it is treated as stable on the shelf.
SHELF_STABLE_FRAMES = 15

# Number of frames a stable item can disappear before it is marked as taken in offline/video mode.
SHELF_DISAPPEAR_THRESHOLD = 90

# Number of frames a disappeared item stays eligible for shelf-side re-identification in offline/video mode.
SHELF_REID_WINDOW_FRAMES = 10

# Minimum IoU to help match a reappearing shelf item with a previous lost item.
SHELF_REID_IOU_THRESHOLD = 0.25

# Maximum center distance in pixels for shelf-side item re-identification.
SHELF_REID_MAX_DIST = 200

# Number of warmup frames to discard after opening a shelf camera session.
SHELF_CAMERA_WARMUP_FRAMES = 5

# Number of extra live frames to discard before starting shelf analysis to reduce stale camera buffers.
SHELF_LIVE_BUFFER_FLUSH_FRAMES = 3

# Whether to show the live shelf-analysis preview window.
SHELF_SHOW_PREVIEW = True

# Number of hands the back-camera MediaPipe tracker will try to track at once.
BACK_HAND_MAX_HANDS = 6

# Minimum confidence for initial hand detection on the back camera.
BACK_HAND_DETECTION_CONFIDENCE = 0.5

# Minimum confidence for continuing hand tracking on the back camera.
BACK_HAND_TRACKING_CONFIDENCE = 0.5

# Which fingertip to use as the reach point in back-camera hand tracking.
BACK_HAND_FINGERTIP = "middle"

# Number of frames to wait before the same person can trigger the same shelf zone again.
SHELF_EVENT_COOLDOWN_FRAMES = 60

# Fractional horizontal/vertical expansion applied when matching a hand to a person box on the back camera.
SHELF_EVENT_PERSON_BBOX_EXPAND = 0.5

# Number of frames a shelf-entry banner remains visible on the back-camera display.
SHELF_EVENT_BANNER_DURATION = 60

# Distance threshold used to decide whether a new tracker ID should merge into an existing ReID identity.
REID_MATCH_THRESHOLD = 600

# ReID backbone architecture name.
REID_MODEL_NAME = "osnet_x0_25"

# Input image height for the ReID preprocessing pipeline.
REID_IMAGE_HEIGHT = 256

# Input image width for the ReID preprocessing pipeline.
REID_IMAGE_WIDTH = 128

# Distance metric used when comparing ReID embeddings.
REID_DISTANCE_METRIC = "euclidean"
