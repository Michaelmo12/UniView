# Algorithm Pipeline Data Flow

**Complete guide to understanding how data flows through the UniView algorithm pipeline.**

---

## Overview

The pipeline processes frames through 5 stages (currently 4 implemented):

```
TCP Streams → Ingestion → Detection → Features → Fusion → [Reconstruction] → [Tracking]
```

Each stage:
- **Receives** input data structures
- **Processes** the data
- **Generates** new data structures
- **Passes** everything downstream

**Key Principle:** Each stage only stores what it **generates**, not what it **receives**. This avoids data duplication.

---

## Stage 1: Ingestion

### Input
- Raw TCP packets from 8 drone streams (ports 15000-15007)
- Binary protocol: header + JPEG frame + calibration data

### Processing
1. **TCPReceiver**: Decode binary packets → DroneFrame
2. **FrameSynchronizer**: Group frames by frame_num → SynchronizedFrameSet

### Output: `SynchronizedFrameSet`

```python
sync_set = SynchronizedFrameSet(
    frame_num=42,        # Frame sequence number
    timestamp=1.4,       # Synchronized timestamp (seconds)
    frames={             # Dictionary: drone_id → DroneFrame
        1: DroneFrame(
            drone_id=1,
            frame_num=42,
            timestamp=1.4,
            frame=np.ndarray,  # (1080, 1920, 3) BGR image
            calibration=CameraCalibration(
                K=np.ndarray,              # (3, 3) intrinsic matrix
                R=np.ndarray,              # (3, 3) rotation matrix
                t=np.ndarray,              # (3, 1) translation vector
                dist=np.ndarray,           # (5,) distortion coefficients
                projection_matrix=np.ndarray  # (3, 4) P = K[R|t]
            )
        ),
        2: DroneFrame(...),
        3: DroneFrame(...),
        ...
        8: DroneFrame(...)
    }
)
```

**What it contains:**
- ✅ Full RGB images (for visualization, feature extraction)
- ✅ Camera calibration (K, R, t, distortion, projection matrices)
- ✅ Timestamps (for synchronization)
- ❌ NO detection information
- ❌ NO features

**Memory:** ~45 MB per frame set (8 cameras × 1920×1080×3 bytes)

---

## Stage 2: Detection (YOLO)

### Input
- `SynchronizedFrameSet` (from Stage 1)

### Processing
1. **BatchDetector**: Run YOLOv11 on all frames in parallel
2. Extract bounding boxes for "person" class (class_id=0)
3. Assign local_id to each detection (0, 1, 2, ...)

### Output: `dict[int, DetectionSet]`

```python
detection_sets = {
    1: DetectionSet(
        drone_id=1,
        frame_num=42,
        inference_time=0.125,  # seconds
        detections=[
            Detection(
                bbox=BoundingBox(x1=400, y1=300, x2=500, y2=550),
                class_id=0,              # person
                confidence=0.95,
                drone_id=1,
                frame_num=42,
                local_id=0,              # First detection in this camera
                features=None            # Not extracted yet
            ),
            Detection(
                bbox=BoundingBox(x1=700, y1=350, x2=800, y2=600),
                class_id=0,
                confidence=0.87,
                drone_id=1,
                frame_num=42,
                local_id=1,              # Second detection in this camera
                features=None
            ),
            # ... more detections
        ]
    ),
    2: DetectionSet(...),
    3: DetectionSet(...),
    ...
}
```

**What it contains:**
- ✅ Bounding boxes (x1, y1, x2, y2)
- ✅ Class IDs and confidences
- ✅ local_id (unique within each camera)
- ✅ Metadata (drone_id, frame_num)
- ❌ NO features yet (features=None)
- ❌ NO images (only bbox coordinates)
- ❌ NO calibration

**Memory:** ~200 bytes per detection (BoundingBox + metadata)

**Typical numbers:** 30-40 detections per camera, 8 cameras = 240-320 detections per frame

---

## Stage 3: Feature Extraction (WCH)

### Input
- `SynchronizedFrameSet` (for images)
- `dict[int, DetectionSet]` (for bounding boxes)

### Processing
1. **WCHExtractor**: For each detection:
   - Crop image using bbox
   - Extract WCH (Weighted Color Histogram) descriptor
   - Store in `detection.features` (modifies Detection in-place)
   - Create `PersonFeatures` object (bundles detection + projection_matrix)

### Output (TWO things):

#### 3A. Modified `DetectionSet` (features populated)
```python
detection_sets = {
    1: DetectionSet(
        detections=[
            Detection(
                bbox=BoundingBox(...),
                confidence=0.95,
                drone_id=1,
                frame_num=42,
                local_id=0,
                features=np.ndarray  # ← NOW POPULATED! (96-dim WCH)
            ),
            # ...
        ]
    ),
    # ...
}
```

#### 3B. Features dict: `dict[int, list[PersonFeatures]]`
```python
features_dict = {
    1: [
        PersonFeatures(
            drone_id=1,
            frame_num=42,
            local_id=0,
            wch=np.ndarray,              # (96,) L2-normalized histogram
            bbox_center=(450.0, 425.0),  # Computed from bbox
            projection_matrix=np.ndarray,# (3, 4) from calibration
            confidence=0.95
        ),
        PersonFeatures(
            drone_id=1,
            frame_num=42,
            local_id=1,
            wch=np.ndarray,
            bbox_center=(750.0, 475.0),
            projection_matrix=np.ndarray,
            confidence=0.87
        ),
        # ... more features
    ],
    2: [...],
    3: [...],
    # ...
}
```

**What PersonFeatures contains:**
- ✅ WCH feature vector (96-dim)
- ✅ Bounding box center (for epipolar geometry)
- ✅ Projection matrix (for computing fundamental matrix)
- ✅ Metadata (drone_id, frame_num, local_id)
- ⚠️ **DUPLICATES data from Detection** (drone_id, frame_num, local_id, confidence)

**Memory:** ~456 bytes per PersonFeatures (with duplication)

**Why two outputs?**
- **DetectionSet**: Used by downstream stages that need bbox, confidence
- **features_dict**: Used by fusion for geometric + appearance matching

---

## Stage 4: Fusion (Cross-Camera Matching)

### Input (THREE things)
```python
# 1. Detection sets (has bboxes, features)
detection_sets: dict[int, DetectionSet]

# 2. Projection matrices (extracted from sync_set)
projection_matrices = {
    drone_id: sync_set.frames[drone_id].calibration.projection_matrix
    for drone_id in detection_sets.keys()
}

# 3. Features (has WCH, bbox_center, projection_matrix)
features_dict: dict[int, list[PersonFeatures]]
```

### Processing

#### Step 1: For each camera pair (i, j)

**1.1 Compute Fundamental Matrix**
```python
P_i = projection_matrices[i]  # (3, 4)
P_j = projection_matrices[j]  # (3, 4)
F_ij = compute_fundamental_matrix(P_i, P_j)  # (3, 3)
```

**1.2 Epipolar Filtering (Geometric constraint)**
```python
# For all detection pairs from cameras i and j
candidates = []
for feat_i in features_dict[i]:
    for feat_j in features_dict[j]:
        # Compute symmetric epipolar distance
        distance = compute_epipolar_distance(
            feat_i.bbox_center,
            feat_j.bbox_center,
            F_ij
        )

        if distance <= 5.0:  # epipolar_threshold
            candidates.append((idx_i, idx_j, distance))

# Result: 90% of pairs rejected (geometric impossibility)
```

**1.3 Appearance Matching (WCH similarity)**
```python
# Build similarity matrix for candidates only
similarity_matrix = compute_wch_similarity(
    [features_dict[i][idx_i].wch for idx_i, _, _ in candidates],
    [features_dict[j][idx_j].wch for _, idx_j, _ in candidates]
)

# Hungarian algorithm: optimal 1-to-1 assignment
matches = hungarian_assignment(similarity_matrix)

# Filter by appearance threshold
confirmed_matches = [
    (idx_i, idx_j, similarity)
    for (idx_i, idx_j, similarity) in matches
    if similarity >= 0.7  # appearance_threshold
]
```

**1.4 Create CrossCameraMatch objects**
```python
pairwise_matches = []
for idx_i, idx_j, similarity in confirmed_matches:
    match = CrossCameraMatch(
        drone_id_a=i,
        drone_id_b=j,
        local_id_a=idx_i,
        local_id_b=idx_j,
        epipolar_distance=candidates_dict[(idx_i, idx_j)],
        appearance_score=similarity,
        is_valid=True
    )
    pairwise_matches.append(match)
```

#### Step 2: Merge all pairwise matches into groups (BFS clustering)

```python
# Build adjacency graph
graph = defaultdict(set)
for match in all_pairwise_matches:
    a = (match.drone_id_a, match.local_id_a)
    b = (match.drone_id_b, match.local_id_b)
    graph[a].add(b)
    graph[b].add(a)

# Find connected components (BFS)
match_groups = []
visited = set()

for node in graph:
    if node in visited:
        continue

    # BFS to find all connected detections
    component = bfs(graph, node)
    visited.update(component)

    # Create MatchGroup
    group = MatchGroup(
        detections=list(component),  # [(drone_id, local_id), ...]
        mean_appearance_score=compute_mean_score(component, matches)
    )
    match_groups.append(group)
```

**Example of transitive closure:**
```
Pairwise matches:
- Camera 1, Detection 0 ↔ Camera 2, Detection 3 (similarity=0.85)
- Camera 2, Detection 3 ↔ Camera 4, Detection 1 (similarity=0.78)

BFS finds connected component:
{(1, 0), (2, 3), (4, 1)}

Result: One MatchGroup with 3 cameras, 3 detections
```

### Output: `FusionResult`

```python
fusion_result = FusionResult(
    frame_num=42,
    match_groups=[
        MatchGroup(
            detections=[(1, 0), (2, 3), (4, 1)],  # Person A in 3 cameras
            mean_appearance_score=0.815
        ),
        MatchGroup(
            detections=[(2, 5), (3, 2)],          # Person B in 2 cameras
            mean_appearance_score=0.892
        ),
        MatchGroup(
            detections=[(1, 2)],                   # Person C in 1 camera (unmatched)
            mean_appearance_score=0.0
        ),
        # ... more groups
    ],
    total_detections=307,  # Total input detections
    total_matches=61       # Unique persons found
)
```

**What it contains:**
- ✅ Match groups (which detections represent the same person)
- ✅ Appearance scores
- ✅ Indices: (drone_id, local_id) tuples
- ❌ NO bounding boxes
- ❌ NO images
- ❌ NO calibration
- ❌ NO 2D points

**Memory:** ~100 bytes per match group (lightweight - just indices)

---

## Stage 5: Reconstruction (3D Triangulation) [PHASE 3 - NOT YET IMPLEMENTED]

### Input (THREE things needed)
```python
# 1. Which detections to triangulate
fusion_result: FusionResult

# 2. Where are the detections in 2D images?
detection_sets: dict[int, DetectionSet]

# 3. What are the camera parameters?
sync_set: SynchronizedFrameSet  # (for projection matrices)
```

### Processing
```python
reconstruction_results = []

for group in fusion_result.match_groups:
    if group.num_cameras < 2:
        continue  # Can't triangulate with 1 camera

    # Collect 2D points and projection matrices
    points_2d = []
    projection_matrices = []

    for drone_id, local_id in group.detections:
        # Get 2D point from detection
        detection = detection_sets[drone_id].detections[local_id]
        point_2d = detection.bbox.center  # (x, y)
        points_2d.append(point_2d)

        # Get projection matrix from calibration
        P = sync_set.frames[drone_id].calibration.projection_matrix
        projection_matrices.append(P)

    # Triangulate 3D position using DLT (Direct Linear Transform)
    point_3d = triangulate_dlt(points_2d, projection_matrices)

    reconstruction_results.append(
        Person3D(
            position=point_3d,           # (x, y, z) in world coordinates
            group=group,                 # Which detections were used
            reprojection_error=compute_reprojection_error(...)
        )
    )
```

### Output: `ReconstructionResult`
```python
ReconstructionResult(
    frame_num=42,
    persons=[
        Person3D(
            position=np.array([2.3, 1.5, 0.0]),  # meters
            group=MatchGroup(...),
            reprojection_error=1.2  # pixels
        ),
        # ... more 3D persons
    ]
)
```

---

## Complete Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│ STAGE 1: INGESTION                                                  │
├─────────────────────────────────────────────────────────────────────┤
│ Input:  TCP packets (binary)                                        │
│ Output: SynchronizedFrameSet                                        │
│         ├─ frames: {drone_id → DroneFrame}                          │
│         │   ├─ frame: (1080, 1920, 3) image                         │
│         │   └─ calibration: K, R, t, dist, P                        │
│         ├─ frame_num: 42                                            │
│         └─ timestamp: 1.4s                                          │
│                                                                      │
│ Memory: ~45 MB                                                      │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ STAGE 2: DETECTION (YOLO)                                           │
├─────────────────────────────────────────────────────────────────────┤
│ Input:  SynchronizedFrameSet                                        │
│ Output: detection_sets: {drone_id → DetectionSet}                  │
│         └─ detections: [Detection, Detection, ...]                 │
│             ├─ bbox: (x1, y1, x2, y2)                               │
│             ├─ confidence: 0.95                                     │
│             ├─ local_id: 0, 1, 2, ...                               │
│             └─ features: None (not extracted yet)                   │
│                                                                      │
│ Typical: 30-40 detections/camera × 8 cameras = 240-320 detections  │
│ Memory: ~200 bytes/detection                                        │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ STAGE 3: FEATURE EXTRACTION (WCH)                                   │
├─────────────────────────────────────────────────────────────────────┤
│ Input:  SynchronizedFrameSet (images) + detection_sets (bboxes)    │
│ Output: 1) Modified detection_sets (features populated)            │
│            └─ detection.features = 96-dim WCH vector                │
│         2) features_dict: {drone_id → [PersonFeatures, ...]}       │
│            └─ PersonFeatures:                                       │
│                ├─ wch: (96,) WCH vector                             │
│                ├─ bbox_center: (x, y)                               │
│                ├─ projection_matrix: (3, 4)                         │
│                └─ metadata: drone_id, frame_num, local_id           │
│                                                                      │
│ Memory: ~456 bytes/PersonFeatures (with duplication)                │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ STAGE 4: FUSION (CROSS-CAMERA MATCHING)                             │
├─────────────────────────────────────────────────────────────────────┤
│ Input:  detection_sets + projection_matrices + features_dict        │
│                                                                      │
│ Processing:                                                         │
│   For each camera pair (i, j):                                     │
│     1. Compute F_ij (fundamental matrix)                            │
│     2. Epipolar filter → candidates (90% reduction)                 │
│     3. WCH similarity → confirmed matches (Hungarian)               │
│     4. Create CrossCameraMatch objects                              │
│                                                                      │
│   Merge all pairwise matches → MatchGroups (BFS clustering)        │
│                                                                      │
│ Output: FusionResult                                                │
│         ├─ match_groups: [MatchGroup, MatchGroup, ...]             │
│         │   └─ detections: [(drone_id, local_id), ...]             │
│         ├─ total_detections: 307                                    │
│         └─ total_matches: 61 (unique persons)                       │
│                                                                      │
│ Memory: ~100 bytes/group (lightweight - just indices)               │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ STAGE 5: RECONSTRUCTION (3D TRIANGULATION) [PHASE 3 - TODO]        │
├─────────────────────────────────────────────────────────────────────┤
│ Input:  fusion_result + detection_sets + sync_set                  │
│                                                                      │
│ For each match_group:                                              │
│   1. Get 2D points from detection_sets (bbox.center)                │
│   2. Get projection matrices from sync_set (calibration.P)          │
│   3. Triangulate 3D position (DLT algorithm)                        │
│                                                                      │
│ Output: ReconstructionResult                                        │
│         └─ persons: [Person3D, Person3D, ...]                       │
│             ├─ position: (x, y, z) in world coords                  │
│             ├─ group: MatchGroup (which detections)                 │
│             └─ reprojection_error: pixels                           │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Why You Need Multiple Data Structures

### Example: Triangulating Person A in Frame 42

**From FusionResult:**
```python
group = MatchGroup(detections=[(1, 0), (2, 3), (4, 1)])
# "Person A is detection 0 in cam 1, detection 3 in cam 2, detection 1 in cam 4"
```

**But to triangulate, you need:**

**1. 2D image points** (from `detection_sets`)
```python
point_1 = detection_sets[1].detections[0].bbox.center  # (450, 425)
point_2 = detection_sets[2].detections[3].bbox.center  # (520, 380)
point_4 = detection_sets[4].detections[1].bbox.center  # (600, 410)
```

**2. Projection matrices** (from `sync_set`)
```python
P_1 = sync_set.frames[1].calibration.projection_matrix  # (3, 4)
P_2 = sync_set.frames[2].calibration.projection_matrix  # (3, 4)
P_4 = sync_set.frames[4].calibration.projection_matrix  # (3, 4)
```

**3. Triangulate**
```python
points_2d = [point_1, point_2, point_4]
P_matrices = [P_1, P_2, P_4]
point_3d = triangulate_dlt(points_2d, P_matrices)
# Result: (2.3, 1.5, 0.0) meters in world coordinates
```

**FusionResult alone is NOT enough** - it only tells you WHICH detections to combine.

---

## Key Takeaways

1. **Each stage generates new data, doesn't duplicate inputs**
   - Ingestion: frames + calibration
   - Detection: bounding boxes
   - Features: WCH vectors (stored in Detection.features)
   - Fusion: match groups (indices only)

2. **Downstream stages need data from MULTIPLE previous stages**
   - Fusion needs: detection_sets + projection_matrices + features_dict
   - Reconstruction needs: fusion_result + detection_sets + sync_set

3. **FusionResult is lightweight**
   - Only stores indices: (drone_id, local_id)
   - Actual data lives in detection_sets and sync_set
   - Avoids duplicating images, bboxes, calibration

4. **Memory hierarchy**
   - SynchronizedFrameSet: ~45 MB (images)
   - DetectionSets: ~64 KB (307 detections × 200 bytes)
   - PersonFeatures: ~140 KB (307 × 456 bytes with duplication)
   - FusionResult: ~6 KB (61 groups × 100 bytes)

5. **Data flow pattern**
   ```
   Heavy data (images, calibration) → Generated once in ingestion
                                    → Passed by reference to all stages

   Light data (indices, scores)    → Generated in each stage
                                    → Small memory footprint
   ```

---

## Practical Usage Pattern

```python
# Run pipeline on one frame
sync_set = receive_and_sync_frames()           # Stage 1
detection_sets = detector.process(sync_set)     # Stage 2
features_dict = extractor.extract_all(sync_set, detection_sets)  # Stage 3

# Extract projection matrices
projection_matrices = {
    drone_id: sync_set.frames[drone_id].calibration.projection_matrix
    for drone_id in detection_sets.keys()
}

# Fusion
fusion_result = matcher.match_frame(
    detection_sets,
    projection_matrices,
    features_dict
)  # Stage 4

# Reconstruction (Phase 3 - future)
reconstruction_result = reconstructor.triangulate(
    fusion_result,      # Which detections to triangulate
    detection_sets,     # Get 2D points
    sync_set           # Get projection matrices
)  # Stage 5

# You need to KEEP all three:
# - sync_set (calibration, images)
# - detection_sets (bboxes, features)
# - fusion_result (match groups)
```

---

## FAQ

**Q: Why not store projection_matrix in Detection?**

A: That would couple the detection module to camera calibration. Detection should only know about bounding boxes, not camera geometry. Plus it would add 48 bytes × 307 detections = 14.7 KB duplication per frame.

**Q: Why does PersonFeatures duplicate data from Detection?**

A: Legacy design from Phase 1. Originally PersonFeatures was meant to be a standalone feature representation. In practice, it creates memory waste. Could be optimized to just store a reference to Detection + projection_matrix.

**Q: Can I discard sync_set after fusion?**

A: NO! You need it for Phase 3 (reconstruction) to get projection matrices. You also need it if you want to visualize detections on the original images.

**Q: Why not just have one big data structure with everything?**

A: Separation of concerns. Each module owns its output. Makes testing easier and reduces coupling between stages.

---

**Last updated:** 2026-02-16
**Author:** Phase 2 implementation documentation
