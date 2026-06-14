# Automatic Checkout Pipeline Redesign Survey

## Context

The current pipeline combines:

- back-camera person detection with YOLO + ByteTrack
- appearance-based person ReID to merge tracker IDs
- hand-to-person association by expanded bounding-box proximity
- zone-entry events from the back camera
- on-demand shelf-camera analysis bursts to infer which item disappeared

That works as a prototype, but it is computationally expensive and structurally brittle for an Amazon Go style system.

## What The Current Code Is Doing

- `main.py:82` extracts a ReID feature for every tracked crop on every frame and stores the full history per track.
- `main.py:109` only tries to merge identities after a new tracker ID has accumulated at least 10 crops, then compares it against prior IDs using full embedding histories.
- `shelf_event_detector.py:242` associates a hand to a person mostly by wrist proximity to expanded person boxes.
- `main.py:322` pauses normal reasoning after each shelf event and launches shelf-camera analysis.
- `shelf_checkout.py:149` runs a second live analysis loop on the shelf camera.
- `shelf_checkout.py:497` uses another lightweight re-association heuristic for shelf items based on IoU and center distance.

## Why This Fails

### 1. Appearance ReID is solving the wrong problem

For checkout, you do not really need "who is this person visually across arbitrary views?" as the primary primitive. You need:

- which shopper owned the interaction episode
- which shelf zone the episode happened in
- which product left the shelf
- whether the product stayed with that shopper until exit

Back-only then front-only observations are exactly where generic person ReID becomes unreliable. Clothing embeddings are sensitive to viewpoint, occlusion, lighting, carts, bags, and partial crops.

### 2. Identity is being inferred too late

The current system first tracks locally, then tries to fuse IDs after fragmentation. That means identity errors accumulate before the shelf event is even interpreted.

### 3. The pipeline duplicates perception work

One event on the back camera can trigger an additional multi-frame shelf-camera inference session. That scales poorly with shopper count and event frequency.

### 4. Person ownership is heuristic-heavy

Wrist-to-person and shelf-side actor matching are both heuristic:

- hand ownership from expanded person boxes
- shelf actor matching from left/right side hints

Those are fragile in crowded scenes.

## Better Replacement Directions

## Option A: Global multi-camera tracking with calibrated geometry

Replace "back-camera identity + ReID + event-triggered shelf analysis" with a synchronized multi-camera pipeline:

- calibrate all cameras to one store coordinate system
- detect people continuously in every camera
- project feet/contact points to the floor plane
- run a global multi-camera tracker in world coordinates
- detect shelf interactions continuously, not in burst mode
- detect product removal from shelf cameras continuously
- bind each removal to the nearest active hand/contact event in time and 3D/2D geometry

### Why it is better

- identity comes from motion continuity across cameras, not mostly from appearance
- front/back view changes matter much less after geometric calibration
- no need to re-open a second analysis loop per event
- much better scaling for multiple simultaneous shoppers

### Costs

- requires camera calibration and synchronization
- more engineering up front
- best when you control camera placement

### Best fit

This is the strongest architecture if the goal is a serious autonomous checkout system.

## Option B: Trackless transaction-centric pipeline

Instead of maintaining strong person identity everywhere, build short-lived interaction episodes:

- entrance detector gives shopper tokens
- per-zone occupancy tracks "someone is in front of shelf Z"
- hand/shelf contact starts an interaction episode
- product disappearance is attached to that episode
- episode follows the nearest shopper token until exit
- final basket is resolved at exit gates using trajectory continuity and item possession state

### Why it is better

- avoids global ReID as the center of the system
- much cheaper than multi-camera global MOT
- robust to front/back view mismatch because identity is mostly temporal and local

### Costs

- weaker if many shoppers cross tightly and repeatedly
- needs clean entrance/exit instrumentation
- harder to support free roaming with long occlusions

### Best fit

This is the best practical redesign if you want to keep complexity moderate.

## Option C: Keep tracking, remove heavy ReID

If you want the least disruptive upgrade:

- keep ByteTrack or move to BoT-SORT/OC-SORT
- remove per-frame deep ReID extraction from `main.py`
- replace global person ReID with a lighter identity state:
  - motion continuity
  - floor position continuity
  - coarse appearance signatures like color histograms
  - optional face embedding only when visible and high-confidence
- keep shelf reasoning continuous instead of event-triggered

### Why it is better

- much cheaper immediately
- simpler to retrofit into current code
- removes the worst failure mode from back/front appearance mismatch

### Costs

- still heuristic
- still weaker than calibrated multi-camera tracking
- may break under occlusion-heavy scenes

### Best fit

Good intermediate step for a student project or MVP.

## Recommended Architecture

Recommended target: **Option A if this is a serious final system, Option B if you need a strong project result with manageable implementation risk.**

My recommendation for this repo specifically is a hybrid of A and B:

1. Remove person ReID as the primary identity mechanism.
2. Replace event-triggered shelf analysis with continuous shelf-camera processing.
3. Introduce a shared store-map coordinate system for all cameras.
4. Maintain shopper state as short tracklets linked by geometry and time, not just appearance.
5. Treat basket updates as event fusion:
   - person-near-shelf
   - hand contact
   - product removed
   - product remains absent
6. Resolve ambiguous ownership with delayed commitment instead of instant assignment.

## Concrete New Pipeline

### Stage 1: Store calibration

- calibrate each camera with homography to a common floor plane
- define shelf polygons and entrance/exit zones in world coordinates

### Stage 2: Continuous perception

- run person detection in all relevant cameras
- run item detection or shelf-state detection in shelf cameras continuously
- run pose/hands only in shelf-near regions, not full frame everywhere

### Stage 3: Global association

- convert each shopper track to world-position observations
- fuse observations across cameras using time + position gating
- use appearance only as a tie-breaker, not primary evidence

### Stage 4: Interaction graph

Build event nodes:

- `person_near_zone`
- `hand_reaches_zone`
- `item_moved`
- `item_removed`
- `person_leaves_zone`
- `person_exits_store`

Connect nodes by temporal overlap and geometric consistency. Basket assignment comes from the graph, not a single heuristic at one frame.

### Stage 5: Delayed basket commit

When ownership is ambiguous:

- mark item assignment as provisional
- wait for next evidence:
  - same shopper carries item away
  - second camera confirms possession
  - competing shopper leaves empty-handed

This is much more stable than updating cart immediately on one event.

## Model Suggestions

Use model families by role, not one heavy model for everything:

- person detection: YOLOv8n/s or RT-DETR if accuracy matters more than speed
- single-camera tracking: ByteTrack or BoT-SORT
- pose near shelves: lightweight pose model only on ROI
- shelf state:
  - product detector if SKUs are visually distinct
  - or shelf occupancy / empty-slot detection if SKU recognition is hard
- appearance:
  - optional compact embedding only for tie-breaking
  - not mandatory as a first-class identity source

## What To Remove From This Repo First

Highest-value changes:

1. Remove `ReIDManager` from the critical path.
2. Stop launching `ShelfItemMonitor.analyze_source()` in response to each event.
3. Replace one-shot cart updates with provisional interaction records.
4. Add a central event buffer that fuses:
   - back-camera person track state
   - shelf-camera item state
   - hand contact state

## Migration Plan

### Phase 1: Low-risk cleanup

- keep current detectors and trackers
- remove deep ReID feature extraction from every frame
- store only short per-track summaries
- make shelf processing continuous
- output provisional events instead of direct cart writes

### Phase 2: Geometry-aware association

- calibrate cameras
- move from image-space left/right matching to world-space proximity matching
- associate shopper ownership using trajectory + zone dwell time

### Phase 3: Global transaction engine

- add delayed assignment and confidence scores
- keep ambiguity sets instead of forcing one owner immediately
- finalize basket when enough evidence accumulates or at exit

## Bottom Line

If you keep the current architecture and only swap in a better ReID model, you will improve accuracy a bit but not solve the core issue.

The real fix is to move from:

**person appearance matching -> event trigger -> second camera re-analysis -> immediate cart update**

to:

**continuous multi-camera evidence fusion -> interaction graph -> delayed confident basket assignment**

That redesign directly addresses both of your concerns:

- much less compute wasted on repeated ReID and repeated shelf re-analysis
- much less sensitivity to a shopper first appearing from the back and later from the front
