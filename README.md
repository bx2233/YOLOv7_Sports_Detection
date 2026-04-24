YOLOv7 Object Detection (Soccer Game)

**Course:** GR5293 Applied Machine Learning for Computer Vision

## Overview

This assignment uses the pre-trained YOLOv7 model to perform three object-detection tasks: detecting players in a football video, detecting the sports ball in the same video, and running detection on a webcam capture to surface a clear mis-detection. All code was run in a Google Colab notebook on an NVIDIA Tesla T4 GPU, using the `yolov7-object-tracking` repository by Rizwan Munawar (https://github.com/RizwanMunawar/yolov7-object-tracking), which wraps the original YOLOv7 detector (https://github.com/WongKinYiu/yolov7) together with a SORT-based object tracker.

## Files in this submission

| File | Purpose |
| --- | --- |
| `task1_players.mp4` | Task 1 — players detected in the football clip |
| `task2_ball.mp4` | Task 2 — sports ball detected in the football clip |
| `task3_webcam.jpg` | Task 3 — webcam capture showing a mis-detection |
| `Homework4_YOLOv7.ipynb` | Full Colab notebook with every command and its terminal output |
| `README.md` | This file |

## Explanation of the command line

The core command for Task 1 was:

```bash
python detect_and_track.py \
  --weights yolov7.pt \
  --source football_clip.mp4 \
  --classes 0 \
  --device 0 \
  --name task1_players \
  --no-trace
```

For Task 2:

```bash
python detect.py \
  --weights yolov7.pt \
  --source football_clip.mp4 \
  --classes 32 \
  --conf-thres 0.15 \
  --device 0 \
  --name task2_ball \
  --no-trace
```

For Task 3 (on a single webcam frame captured through Colab's JavaScript bridge, since the Colab VM has no direct webcam access):

```bash
python detect.py \
  --weights yolov7.pt \
  --source webcam.jpg \
  --device 0 \
  --name task3_webcam \
  --no-trace
```

The flags mean:

- `--weights yolov7.pt` loads the official YOLOv7 weights pre-trained on the COCO dataset (80 classes).
- `--source` is the input — an MP4 for the football tasks, a JPG for the webcam task. `0` would be used for a live camera stream on a local machine.
- `--classes 0` restricts output to class ID 0 (`person`) so the Task 1 video only boxes players. `--classes 32` restricts output to `sports ball` for Task 2.
- `--conf-thres 0.15` lowers the minimum confidence from the default `0.25`. The sports ball is small and fast-moving, so it rarely exceeds the default threshold; lowering it trades some false positives for many more true detections.
- `--device 0` puts inference on the first CUDA device (the T4 GPU).
- `--name` sets the subfolder under `runs/detect/` where the annotated output is saved.
- `--no-trace` skips the TorchScript tracing step and avoids a compatibility warning under the current PyTorch version on Colab (torch 2.10.0+cu128).
- `detect_and_track.py` (Task 1) wraps `detect.py` and adds SORT-based ID labels on top of each bounding box, matching the look of the reference image in the homework PDF. `detect.py` (Task 2 and Task 3) draws only class + confidence labels, which is sufficient when we are not tracking identities.

Before running YOLO, `ffmpeg` was used to trim the 5-minute source video down to a 45-second clip with clear action, keeping inference time reasonable while staying comfortably above the 30-second minimum required by the assignment:

```bash
ffmpeg -y -ss 00:01:00 -i football.mp4 -t 45 -c:v libx264 -an football_clip.mp4
```

One extra patch was needed after `pip install -r requirements.txt`. The `sort.py` tracker in the repo uses `np.int`, which was removed in NumPy 1.24. Replacing `np.int)` with `int)` in-place and pinning `numpy<2` makes the tracker run cleanly:

```bash
sed -i 's/np\.int)/int)/g' sort.py
pip install -q 'numpy<2' 'opencv-python-headless' 'filterpy' 'scikit-image' 'lap'
```

## Problems observed in the results

**Task 1 (players).** Spectators and staff on the sidelines are sometimes labeled as `person` along with the actual players on the pitch, since they are all human shapes and YOLO has no concept of "on or off the field." When players are partially occluded by other players during crowded play, two close bounding boxes can merge into one, or one player disappears for a few frames. When the SORT tracker is enabled, these brief occlusions also cause ID switches: the same player picks up a new integer ID after re-appearing, because the tracker lost the association.

**Task 2 (sports ball).** Detection is intermittent. In any given frame the ball may be picked up with confidence ~0.2–0.5, but it is also frequently missed, especially when the ball is against a busy background such as players' jerseys, when it is motion-blurred from a fast kick, or when it is very small relative to the frame (football broadcast shots are wide). A few false positives also appear — round objects like a white logo on the advertising boards or a player's head silhouette occasionally fire the `sports ball` class at low confidence.

**Task 3 (webcam).** The clear mis-detection is a **framed painting on the wall, boxed as `person` with confidence 0.71**. The painting depicts a person in traditional robes, and YOLO treats it exactly like a real standing person. Additional reasonable detections in the same frame are two `potted plant` boxes and one `vase` box, so the model is not broken — it has simply confused a 2D depiction of a person with a real one.

## Reasons and potential improvements

The reasons for these problems follow naturally from how the model is trained.

The **spectator problem** happens because COCO does not have a "football player" class — it has only `person` — so the model cannot distinguish athletes from anyone else. A fix is domain adaptation: fine-tune on the Kaggle football-analysis dataset, which has pitch-specific classes (player, referee, goalkeeper, ball), or add a simple spatial filter that ignores detections outside a rough pitch polygon.

The **occlusion and ID-switch problems** in tracking are a well-known limitation of SORT, which uses only an IoU-based Kalman filter and no visual appearance features. Swapping SORT for Deep SORT or ByteTrack, both of which incorporate an appearance embedding, keeps identities stable across short occlusions.

The **missed-ball problem** is mostly a small-object problem. At the default 640-pixel inference resolution, a ball that occupies ~10×10 pixels in the broadcast view ends up as just a handful of pixels after the downsample inside the network, which destroys the features YOLO relies on. The most direct fix is `--img-size 1280` (or `1920`) together with a heavier weights file like `yolov7-w6.pt` or `yolov7-e6e.pt`, which are trained for 1280-px input. A stronger fix is tiled inference (e.g. SAHI), which slices each frame into overlapping patches, runs detection on each patch, and merges the results — this was designed exactly for tiny objects in wide scenes.

The **false positives** on round logos can be reduced by raising `--conf-thres` back toward 0.25 once you also bring in the higher-resolution weights; with a stronger backbone the genuine ball detections rise in confidence while spurious ones stay low, so the threshold can do its job.

The **painting-as-person mis-detection** in Task 3 is a fundamental limitation: YOLO learns from 2D images, so it has no way to know whether person-shaped pixels belong to a real 3D person or a printed flat surface. Ways to mitigate: provide hard-negative training examples containing framed pictures and posters, use a monocular depth estimator (e.g. MiDaS) and suppress detections that sit on a perfectly flat depth plane, or chain YOLO with a classifier that filters out anything falling inside a detected picture-frame boundary.

## Understanding of YOLO and YOLOv7

YOLO ("You Only Look Once") is a family of single-stage object detectors introduced by Redmon et al. The defining idea is that detection is a single regression problem: one forward pass through a fully convolutional network simultaneously outputs class probabilities and bounding-box coordinates for every object in the image. This is in contrast to two-stage detectors like Faster R-CNN, which first propose regions of interest and then classify them; by collapsing the two stages into one, YOLO gives up a small amount of accuracy in exchange for a very large speedup, making real-time inference possible.

Architecturally, a YOLO network divides the input image into a grid. Each grid cell is responsible for predicting a fixed number of bounding boxes (*anchor boxes*) along with a class-probability vector. Each predicted box carries four geometry values (center x, center y, width, height), one objectness score (how confident the cell is that any object is present), and one score per class. After the forward pass, a non-maximum-suppression (NMS) step removes overlapping duplicate boxes by keeping only the highest-scoring box in each cluster of overlaps, using the `--iou-thres` parameter (default 0.45) to decide when two boxes count as overlapping.

YOLOv7, released in 2022 by Wang, Bochkovskiy, and Liao (https://arxiv.org/abs/2207.02696), made several architectural and training contributions on top of this base. The backbone uses an **Extended-ELAN (E-ELAN)** block, which improves the efficiency of feature aggregation by controlling the gradient path length as the network deepens. Training uses **auxiliary heads with coarse-to-fine label assignment** — an extra output head receives a looser target assignment during training, which provides a richer learning signal without adding any cost at inference. Finally, the model applies **re-parameterization at inference time**: branches that exist only during training are mathematically folded into a single convolution for deployment, so the deployed model is smaller and faster than the trained one. The result is that YOLOv7 achieves higher accuracy than comparable-cost YOLOv5 and YOLOR variants on the COCO benchmark while running at real-time speeds on a single GPU.

The checkpoint used here, `yolov7.pt`, is the standard-size variant trained on MS COCO 2017 with 80 classes. For the football clips, the relevant COCO class IDs are `0` (person) and `32` (sports ball). Because COCO is a general-purpose benchmark and not football-specific, the off-the-shelf model works reasonably well for the coarse task of "find humans" but struggles with the fine-grained and small-object aspects described in the problems section above.

## References

- YOLOv7 paper — https://arxiv.org/abs/2207.02696
- YOLOv7 official repo — https://github.com/WongKinYiu/yolov7
- YOLOv7 + SORT tracking wrapper (used in this homework) — https://github.com/RizwanMunawar/yolov7-object-tracking
- SORT tracker — https://github.com/abewley/sort
- Football video dataset — https://www.kaggle.com/datasets/venkatkumar001/football-analysis
