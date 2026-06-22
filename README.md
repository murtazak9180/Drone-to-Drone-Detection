# Efficient Heatmap-Guided Sparse Detection for Drone-to-Drone Perception on NPS-Drones

Vision-based drone-to-drone detection is critical for aerial see-and-avoid, coordinated swarming, and collision prevention. State-of-the-art methods such as **TransVisDrone** and **YOLOMG** reach AP@0.5 ≈ 0.95 on NPS-Drones but at substantial computational cost. This project studies the **accuracy–throughput trade-off** with reproducible spatial baselines and a custom **heatmap-guided sparse detector** inspired by CenterNet and YOLOMG-style motion fusion.

---

## Summary

We benchmark six off-the-shelf detectors and iteratively build a lightweight temporal model:

- **Best spatial baseline:** YOLO12s — AP@0.5 **0.854**, **117.7 FPS**
- **Highest spatial AP@0.5:** RT-DETRv1 — **0.913** at 22.3 FPS (high FPPI: 78.6)
- **Best custom model (Phase 3):** AP@0.5 **0.772**, F1 **0.812**, **49.0 FPS** on $T{=}3$ test triplets

Phase 3 approaches YOLO12s accuracy while keeping a sparse three-frame pipeline at real-time speed.

---

## Dataset: NPS-Drones

Public benchmark of 50 videos of a delta-wing aircraft filmed from another airborne platform ([Li et al., 2016](https://arxiv.org/abs/1602.00775)).

| Split | Videos | Frames |
|-------|--------|--------|
| Train | #01–#36 | 32,220 |
| Val   | #37–#40 | 3,753  |
| Test  | #41–#50 | 12,355 |

- **Format:** YOLO axis-aligned boxes, single class `drone`
- **Input size:** 640×640
- **Source:** Public Kaggle release (see notebooks for dataset paths)

### Evaluation protocols

| Setting | Used by | Test samples |
|---------|---------|--------------|
| Every 4th test frame (skip-rate 4) | Spatial baselines, TransVisDrone, Dogfight | 3,089 |
| Overlapping $T{=}3$ triplets | Custom heatmap detectors | 12,335 |

> Direct AP@0.5 comparison across protocols is approximate.

---

## Spatial baselines

All baselines share: AdamW (`lr₀=3e-5`), 100 epochs, 640² input, mosaic + HSV aug, early stopping (patience 30), Tesla T4, Ultralytics v8.4.

**Test results** (every 4th frame, 640², Tesla T4):

| Model | AP50 | Prec. | Rec. | F1 | FPS | FPPI |
|-------|------|-------|------|-----|-----|------|
| YOLO11n | 0.673 | 0.799 | 0.567 | 0.663 | 261.6 | 5.23 |
| YOLO11s | 0.729 | 0.820 | 0.601 | 0.694 | 154.2 | 4.15 |
| YOLO11m | 0.718 | 0.832 | 0.588 | 0.688 | 55.2 | 3.71 |
| **YOLO12s** | **0.854** | **0.917** | 0.812 | **0.862** | **117.7** | 3.27 |
| RT-DETRv1 | 0.913 | 0.917 | 0.870 | 0.893 | 22.3 | 78.57 |

### Published references (NPS test)

| Method | AP50 | F1 | FPS | Notes |
|--------|------|-----|-----|-------|
| TransVisDrone (1280², τ=5) | 0.95 | 0.915 | 24.6 | RTX A6000 |
| TransVisDrone (640²) | 0.91 | 0.869 | 87.7 | RTX A6000 |
| YOLOMG-640 | 0.92 | — | 133 | Authors' hardware |
| Dogfight | 0.89 | 0.92 | ~1.0 | Patch-based |

---

## Custom heatmap-guided detector

Sparse, CenterNet-style design: predict a class heatmap, extract top-$K$ peaks, refine boxes with ROIAlign — instead of dense anchors over the full feature map.

### Shared pipeline

1. **Input:** $T{=}3$ consecutive RGB frames (640²) with temporally consistent augmentations
2. **Backbone:** EfficientNet-B0 + per-level **TemporalFusion** (1×1 conv over concatenated frames)
3. **Neck:** 3-layer BiFPN → P1 (stride 4, 160×160) … P4
4. **Heatmap head:** Gaussian focal loss targets; top-$K{=}50$ peaks at inference
5. **Refinement:** Multi-level ROIAlign (P1–P3) + MLP → objectness + box deltas

### Architecture evolution

| Module | Base (v1) | Phase 1–2 | Phase 3 |
|--------|-----------|-----------|---------|
| Input | RGB only | RGB + motion diff maps | RGB + motion diff maps |
| Heatmap targets | Fixed Gaussian σ=2.0 | Fixed Gaussian σ=2.0 | Adaptive σ = max(min(w,h)/3, 0.5) |
| Anchor | Fixed 8×8 px | Data-driven (~5.9 px median GT) | Data-driven (~5.9 px) |
| ROI grid | 7×7 | 11×11 | 11×11 |
| Box loss | Smooth L1 | Smooth L1 | DIoU |
| Epoch subsampling | Full set | Full set | 35% random triplets / epoch |

**v1** (`without-temporal(1).ipynb`) — appearance-only, no motion difference maps.  
**v2** (`experiments/hestmap-with temporal.ipynb`) — motion encoder + three anchor sizes (8 / 16 / 24 px).  
**Phases 1–3** — single data-driven anchor line with progressive training improvements.

Regenerate the v1 architecture diagram:

```bash
python diagrams/draw_v1_architecture.py
```

### Custom test results ($T{=}3$ triplets, Tesla T4)

| Model | AP50 | AP50-95 | AP25 | F1 | FPS |
|-------|------|---------|------|-----|-----|
| Base (v1) | 0.609 | 0.198 | 0.760 | 0.753 | 49.2 |
| Multi-anchor (v2) | 0.649 | 0.216 | 0.812 | 0.748 | 41.3 |
| Phase 1 | 0.633 | 0.200 | 0.800 | 0.736 | 47.4 |
| **Phase 3** | **0.772** | **0.307** | **0.891** | **0.812** | **49.0** |

Phase 2 test evaluation is not yet complete and is omitted from the table above.

---

## References

1. Ashraf et al., *Dogfight: Detecting Drones from Drones Videos*, arXiv 2021.
2. Sangam et al., *TransVisDrone: Spatiotemporal Transformer for Vision-based Drone-to-Drone Detection*, CVPR 2023.
3. Guo et al., *YOLOMG: Tiny Drone Detection with Motion Difference*, 2025.
4. Law & Deng, *CornerNet: Detecting Objects as Paired Keypoints*, ECCV 2018.
5. Li et al., *Vision-based Autonomous Obstacle Avoidance for Small UAVs*, ICUAS 2016 (NPS-Drones).

---

## License

Academic / course project use. NPS-Drones dataset terms apply to the data.
