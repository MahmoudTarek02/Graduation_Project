import cv2
import torch
import numpy as np
from ultralytics import YOLO
from boxmot import ByteTrack
from pathlib import Path
from reid import REID
import operator

reid = REID()