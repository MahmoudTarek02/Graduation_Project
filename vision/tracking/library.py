import cv2
import torch
import numpy as np
from pathlib import Path
from boxmot import BoostTrack, ByteTrack
from torchvision.models.detection import (
    fasterrcnn_mobilenet_v3_large_320_fpn,
    FasterRCNN_MobileNet_V3_Large_320_FPN_Weights as Weights
)
from reid import REID
import operator


# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Load detector with pretrained weights and preprocessing transforms
weights = Weights.DEFAULT
detector = fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights, box_score_thresh=0.5)
detector.to(device).eval()
transform = weights.transforms()

# Initialize tracker
# tracker = BoostTrack(reid_weights=Path('osnet_x0_25_msmt17.pt'), device=device, half=False)
tracker = tracker = ByteTrack(device=device, half=False)

# Start video capture
cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture('videos/init/Double1.mp4')

# ReID Stuff !!
images_by_id = dict() # Shuold be removed from code
reid = REID()

threshold = 600    # De el simillarity threshold ben el features fel reid
exist_ids = set()
final_fuse_id = dict()

# print(f'Total IDs = {len(images_by_id)}')
feats = dict() # features for each ID for every image in images_by_id

with torch.inference_mode():
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Convert frame to RGB and prepare for detector
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).to(torch.uint8)
        input_tensor = transform(tensor).to(device)


        # Run detection
        output = detector([input_tensor])[0]
        scores = output['scores'].cpu().numpy()
        keep = scores >= 0.5


        # Prepare detections for tracking
        boxes = output['boxes'][keep].cpu().numpy()
        labels = output['labels'][keep].cpu().numpy()
        filtered_scores = scores[keep]
        detections = np.concatenate([boxes, filtered_scores[:, None], labels[:, None]], axis=1)

        # Update tracker and draw results
        #   INPUT:  M X (x, y, x, y, conf, cls)
        #   OUTPUT: M X (x, y, x, y, tracking_id, confidence, object class, detection index in input frame)
        res = tracker.update(detections, frame)

        # Rest is REID logic
        for row in res:
            id = int(row[4])
            x1, y1, x2, y2 = map(int, row[:4])

            if id not in images_by_id:
                images_by_id[id] = []
                feats[id] = []

            crop = frame[y1:y2, x1:x2]
            images_by_id[id].append(crop) #Should be removed from code
            feats[id].append(reid._feature(crop))

        

        if len(res) > 0:
            f = res[:, 4].tolist() # list of tracking ids in current frame
            f = set(f)
            for id in f:
                if len(exist_ids) == 0:
                    for i in f:
                        final_fuse_id[i] = [i]
                    exist_ids = exist_ids or f

                else:
                    new_ids = f - exist_ids
                    for nid in new_ids:
                        dis = []
                        # print(f"F: {f}")
                        # print('Processing nid {}'.format(nid))
                        # print(f'currently in images_by_id {list(images_by_id.keys())}')
                        if len(images_by_id[nid]) < 10:
                            # exist_ids.add(nid) Neglect until more images added
                            continue

                        unpickable = []
                        for i in f:
                            for key, item in final_fuse_id.items():
                                if i in item:
                                    unpickable += final_fuse_id[key]
                        
                        print('exist_ids {} unpickable {}'.format(exist_ids, unpickable))
                        for oid in (exist_ids - set(unpickable)) & set(final_fuse_id.keys()):
                            tmp = np.median(reid.compute_distance(torch.cat(feats[nid], 0), torch.cat(feats[oid], 0)))
                            print('nid {}, oid {}, tmp {}'.format(nid, oid, tmp))
                            dis.append([oid, tmp])
                        exist_ids.add(nid)

                                            
                        if not dis:
                            final_fuse_id[nid] = [nid]
                            continue

                        dis.sort(key=operator.itemgetter(1))
                        if dis[0][1] < threshold:
                            combined_id = dis[0][0]
                            images_by_id[combined_id] += images_by_id[nid]
                            final_fuse_id[combined_id].append(nid)

                        else:
                            final_fuse_id[nid] = [nid]

            print(final_fuse_id)
        displayed = frame.copy()

        tracker.plot_results(displayed, show_trajectories=True)

        # Show output

        cv2.imshow('BoXMOT + Torchvision', displayed)

        if cv2.waitKey(1) & 0xFF == ord('q'):

            break 


# Clean up
cap.release()
cv2.destroyAllWindows()

for track_id, crops in images_by_id.items():
    for i, crop in enumerate(crops):
        # skip invalid crops
        if crop is None or crop.size == 0:
            continue
        
        win_name = f"Haha"
        cv2.imshow(win_name, crop)
        # wait for a key, press any key to go to next crop  
        key = cv2.waitKey(1)
        
        # if you want "q" to quit early
        if key == ord('q'):
            break
    
    # close all windows from previous ID
    cv2.destroyAllWindows()