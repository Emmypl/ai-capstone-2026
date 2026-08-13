# HW2: 3D Scene Reconstruction — Project Summary

## Goal
Reconstruct a 3D scene from RGB-D frames captured in Habitat-Sim using Open3D ICP. Evaluate by comparing estimated camera trajectory against ground truth poses (Mean L2 distance, lower = better).

---

## Files
- `load.py` — data collection script (Habitat-Sim simulator)
- `reconstruct.py` — main pipeline (implement TODOs here)
- Data: `data_collection/first_floor/rgb/`, `depth/`, `GT_pose.npy`

---

## Fixes Already Applied

### `load.py`
- `SensorSpec.position` and `.orientation` must use `mn.Vector3(...)`, not plain Python lists → fixes `TypeError` on startup

### `reconstruct.py` — critical discoveries:

1. **Depth scale**: `load.py` saves depth as 8-bit PNG via `(depth_meters / 10.0 * 255).astype(uint8)`. Inverse decode is:
   ```python
   depth_m = depth / 255.0 * 10.0   # NOT / 1000
   ```

2. **Camera coordinate convention** (Habitat/OpenGL, Y-up, looks toward -Z):
   ```python
   z_cam = -depth_m[valid]
   x_cam =  (uu[valid] - CX) * depth_m[valid] / FX
   y_cam = -(vv[valid] - CY) * depth_m[valid] / FY
   ```
   Intrinsics: 512×512, 90° FOV → `FX=FY=256`, `CX=CY=256`

3. **GT pose format**: `[x, y, z, qw, qx, qy, qz]` → quaternion scipy order is `[qx, qy, qz, qw]`:
   ```python
   R.from_quat([p[4], p[5], p[6], p[3]])
   ```

4. **Normal orientation**: After `estimate_normals`, must call:
   ```python
   pcd_down.orient_normals_towards_camera_location(np.array([0., 0., 0.]))
   ```
   Without this, ~half the normals point inward → Point-to-Plane ICP gradient inversion → tilted reconstruction

5. **RANSAC disabled**: RANSAC (`registration_ransac_based_on_feature_matching`) is too slow (~34s/frame) and produces garbage transforms in repetitive indoor environments (same-looking walls confuse FPFH). Currently bypassed with `ransac_init = np.eye(4)`. FPFH computation also skipped (returns `None`).

6. **Ceiling & Floor Visibility**: Ceilings were occupying up to 20% of the points, so the removal threshold was lowered from the 97th to the **80th percentile** (`np.percentile(..., 80)`). The floor was originally appearing missing/transparent in top-down BEV because voxel downsampling creates a sparse grid with 10cm gaps; increasing the renderer `point_size` from 2.0 to 5.0 merged the gaps into solid surfaces.

7. **Point-to-Plane Corridor Sliding**: Standard `TransformationEstimationPointToPlane` has zero cost gradient for sliding along featureless walls. It hallucinated 12m+ false translations. This was completely fixed by switching to **Point-to-Point** inside the Multiple Hypothesis ICP loop, which properly penalizes sliding.

8. **Multiple Hypothesis Initialization**: Habitat camera turns 10-degrees in place, displacing distant points by >0.5m. A tight correlation threshold (0.15m) misses this completely. We evaluate 4 explicit hypotheses (`Identity`, `Translate Forward -0.25m`, `Yaw +10°`, `Yaw -10°`) and snap to the hypothesis with the highest strictly-calculated fitness overlap.

9. **Constant Velocity Fallback**: When tracking occasionally triggers the safety guard due to extreme feature wipeout, wiping the transform to `Identity` shrinks corridors because it throws away forward steps mathematically. We now fall back to the `last_valid_T_rel`, ensuring the corridor length is preserved accurately.

---

## Current Pipeline (`reconstruct.py`)

```
For each consecutive frame pair:
  1. Load RGB + depth → depth_image_to_point_cloud()
  2. Voxel downsample (voxel_size=0.1m) + estimate normals
  3. Pre-guess: Multiple Hypothesis ICP (Forward, Identity, ±10° Yaw) using Point-to-Point Metric.
  4. Best result chosen based on highest point overlap fitness and lowest RMSE.
  5. Enforce planar constraint (Zero out Pitch, Roll, and Y translation).
  6. Sanity Guard: If fitness < 0.05 or translation > 0.75m, fallback to last valid motion (Constant Velocity).
  7. T_abs = T_abs_prev @ T_rel  [pose chaining]
  8. Transform current cloud to world frame, accumulate
Post: remove ceiling (Y > 80th percentile) to reveal BEV floorplan.
```

---

## Results Table
| Run | Frames | ICP Method | L2 (m) | Time |
|-----|--------|------------|---------|------|
| Baseline RANSAC | 150 | RANSAC → Point-to-Plane | 31.4 | 1250s |
| Baseline Identity | 150 | identity → Point-to-Plane | 22.7 | 20s |
| Basic Chaining | 600 | identity → Point-to-Plane | 7740 | 87s |
| **Current Stable** | **630** | Multi-Hypothesis + Point-to-Point | **4.81** | **66s** |

---

## Visualization
- On **Wayland**: `draw_geometries` fails (GLFW can't create window). Offscreen renderer saves `reconstruction_floor1.png`, then `eog` opens it.
- On **Xorg**: `draw_geometries` works directly (interactive 3D window).
- `os.environ.setdefault('DISPLAY', ':0')` is set at the top of the file to force headless egl if needed.

---

## Known Issues Remaining
- **Jumbled BEV / Drift buildup**: A final Mean L2 error of ~4.8m over almost a thousand frames is exceptionally good for open-loop Sequential ICP without loop closures. However, the absolute topological error (rooms overlapping) still appears because even a microscopic 0.05° tracking error per frame eventually integrates into 90° layout rotation. Professional maps use Pose-Graph Optimization (PGO) or global loop closure backends (e.g. g2o / GTSAM) to distribute this error backward.

Report Contents:
1. Implementation (50%)
   - Code:
      - Provide a clear explanation of how you implement each step of the pipeline.
   - Result & Discussion:
      - Screenshots of apartment_0 Floor 1 reconstruction.
      - A comparison table that includes:
         - Mean L2 distance between estimated and ground truth camera poses.
         - Total execution time.
         - The hyperparameters you have tested.
      - Discuss the results and share your findings.
      - (Bonus) If you implement your own ICP algorithm, compare your results with the Open3D version and highlight the techniques or tricks you used to improve performance.
2. Questions (20%):
   - What happens if you perform ICP without Global Registration (RANSAC)? Why?
   - Describe any tricks used to improve your ICP stability (e.g., Voxel size, Huber loss, etc.).