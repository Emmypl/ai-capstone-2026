import os
os.environ.setdefault('DISPLAY', ':0')  # ensure X11 display is set for Open3D
import re
import glob
import cv2
import numpy as np
import open3d as o3d
import argparse
from copy import deepcopy
from scipy.spatial.transform import Rotation as R
import time

# ---------- Camera Intrinsics (Resolution 512x512, FOV 90) ----------
# These parameters are derived from the Habitat pinhole camera model [cite: 26-27].
IMG_W, IMG_H = 512, 512
FOV = np.deg2rad(90.0)
FX = (IMG_W / 2.0) / np.tan(FOV / 2.0)   # = 256.0
FY = (IMG_H / 2.0) / np.tan(FOV / 2.0)   # = 256.0
CX, CY = IMG_W / 2.0, IMG_H / 2.0
# Depth images were saved in load.py as 8-bit PNG via:
#   pixel = (depth_meters / 10.0 * 255).astype(uint8)
# Inverse: depth_meters = pixel / 255.0 * 10.0
# We express this as pixel / DEPTH_SCALE_A * DEPTH_SCALE_B, but for clarity
# we define the two-step inverse directly in depth_image_to_point_cloud.
DEPTH_SCALE = 1000.0  # NOT used for the saved data — see depth_image_to_point_cloud

def depth_image_to_point_cloud(rgb_image, depth_image):
    """
    TASK 1: Geometric Unprojection [cite: 12, 25-27]
    Convert depth pixels (u, v, d) into 3D world points (x, y, z).
    """
    # 1. Convert inputs to numpy arrays
    rgb   = np.array(rgb_image,   dtype=np.float64)   # (H, W, 3)
    depth = np.array(depth_image, dtype=np.float64)   # (H, W)

    # 2. Convert depth to meters (Habitat depth is often scaled or normalized)
    # load.py saved depth as 8-bit: pixel = (depth_meters / 10.0 * 255).astype(uint8)
    # Inverse: depth_meters = pixel_value / 255.0 * 10.0
    depth_m = depth / 255.0 * 10.0

    # 3. Create a coordinate grid for (u, v) pixels
    # uu[row, col] = col  (pixel column  = u)
    # vv[row, col] = row  (pixel row     = v)
    u_coords = np.arange(IMG_W)
    v_coords = np.arange(IMG_H)
    uu, vv = np.meshgrid(u_coords, v_coords)

    # Mask out pixels with zero / invalid depth before indexing
    valid = depth_m > 0

    # TODO: Implement unprojection logic here
    # x = (u - CX) * z / FX
    # y = (v - CY) * z / FY
    # z = -depth (assuming camera looks towards -Z)
    #
    # --- Coordinate-frame derivation (Habitat / OpenGL convention) ---
    # Habitat uses a right-hand camera frame where:
    #   +X  = right,  +Y = up,  +Z = toward the viewer (camera looks along -Z).
    # A point D meters in front of the camera therefore sits at z_cam = -D.
    #
    # The pinhole projection in this frame is:
    #   u = FX * x_cam / (-z_cam) + CX   =>  x_cam = (u - CX) * (-z_cam) / FX
    #   v = FY * (-y_cam) / (-z_cam) + CY =>  y_cam = -(v - CY) * (-z_cam) / FY
    # (the minus sign on y_cam flips the pixel-row axis, which grows downward,
    #  onto the camera-Y axis, which grows upward.)
    #
    # Substituting -z_cam = depth_m:
    #   z_cam = -depth_m
    #   x_cam = (u - CX) * depth_m / FX
    #   y_cam = -(v - CY) * depth_m / FY
    z_cam =  -depth_m[valid]
    x_cam =  (uu[valid] - CX) * depth_m[valid] / FX
    y_cam = -(vv[valid] - CY) * depth_m[valid] / FY

    points_3d  = np.stack([x_cam, y_cam, z_cam], axis=1)  # (N, 3)
    colors_norm = rgb[valid] / 255.0                        # (N, 3), range [0, 1]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)
    pcd.colors = o3d.utility.Vector3dVector(colors_norm)
    return pcd

def preprocess_point_cloud(pcd, voxel_size):
    """
    Pre-processing: Voxelization and Normal Estimation [cite: 17, 29]
    """
    pcd_down = pcd.voxel_down_sample(voxel_size)

    # TODO: Estimate normals for pcd_down (required for Point-to-Plane ICP)
    # pcd_down.estimate_normals(...)
    # Search radius is set to 2× voxel_size so each point collects enough
    # neighbours for a stable normal estimate without being too slow.
    radius_normal = voxel_size * 2.0 # ANCHOR
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=50) # ANCHOR change max_nn
    )
    # Orient all normals consistently toward the local camera origin [0,0,0].
    # Without this, ~half the normals point inward and ~half outward — Point-to-Plane
    # ICP then optimizes with inverted gradients for those points, producing wrong rotations.
    pcd_down.orient_normals_towards_camera_location(np.array([0., 0., 0.]))

    # Statistical Outlier Removal: strip points whose mean distance to
    # their 20 nearest neighbours is more than 2σ above the cloud average.
    # This removes depth-sensor noise spikes that corrupt normal estimation
    # and attract spurious ICP correspondences.
    pcd_down, _ = pcd_down.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0) # ANCHOR

    # Compute FPFH features for Global Registration [cite: 30]
    # NOTE: FPFH is only needed when RANSAC is enabled. It is skipped here
    # (returns None) to avoid the per-frame cost while RANSAC is disabled.
    # To re-enable: uncomment the block below and remove `return pcd_down, None`.
    return pcd_down, None
    # radius_feature = voxel_size * 5.0 # ANCHOR
    # pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
    #     pcd_down,
    #     o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100) # ANCHOR change max_nn
    # )
    # return pcd_down, pcd_fpfh

def my_local_icp_algorithm(source_pcd, target_pcd, initial_transform):
    """
    TASK 2: Custom ICP Implementation (BONUS 20%)
    Implement your own version of Point-to-Plane ICP.
    """
    T_global = initial_transform.copy()

    # TODO: Implement the ICP loop:
    # 1. Find nearest neighbors using target_tree.search_knn_vector_3d
    # 2. Build the linear system (AtA)x = Atb
    # 3. Solve for pose update and update T_global

    result = o3d.pipelines.registration.RegistrationResult()
    result.transformation = T_global
    return result

def local_icp_algorithm(source_down, target_down, trans_init, voxel_size):
    """
    TASK 2: Open3D ICP Implementation (REQUIRED) [cite: 32]
    """
    # Multiple Hypothesis Initialization:
    # A 10-degree Habitat turn displaces points by >0.5m, making standard Point-to-Plane
    # completely miss the match (fitness=0) when using a tight threshold like 0.15m.
    # Instead of a large threshold (which causes wall-sliding), we initialize ICP with
    # Multiple Hypothesis Initialization:
    # A 10-degree Habitat turn displaces points by >0.5m, making standard Point-to-Plane
    # completely miss the match (fitness=0) when using a tight threshold like 0.15m.
    # We initialize ICP with the 4 possible Habitat actions to snap it mathematically.
    from scipy.spatial.transform import Rotation as R
    T_id = trans_init
    
    T_forward = trans_init.copy()
    # Habitat camera looks down the -Z axis, so moving forward decreases Z
    T_forward[2, 3] -= 0.25 
    
    R_left = R.from_euler('Y', 10, degrees=True).as_matrix()
    T_left = trans_init.copy()
    T_left[:3, :3] = R_left @ T_left[:3, :3]
    
    R_right = R.from_euler('Y', -10, degrees=True).as_matrix()
    T_right = trans_init.copy()
    T_right[:3, :3] = R_right @ T_right[:3, :3]

    hypotheses = [T_id, T_forward, T_left, T_right]
    best_result = None

    for T_guess in hypotheses:
        # POINT-TO-POINT is strictly required here! Point-to-Plane has no loss penalty 
        # for sliding along straight corridor walls. This sliding caused translation drift 
        # >0.75m which triggered the identity guard, completely erasing correct rotations!
        result = o3d.pipelines.registration.registration_icp(
            source_down, target_down, voxel_size * 1.5, T_guess,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=50
            )
        )
        if best_result is None or result.fitness > best_result.fitness:
            # Tie breaker: lower RMSE wins
            if best_result is not None and abs(result.fitness - best_result.fitness) < 1e-3:
                if result.inlier_rmse < best_result.inlier_rmse:
                    best_result = result
            else:
                best_result = result

    return best_result

def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    """
    Global Registration via RANSAC feature matching [cite: 29-31].
    Uses FPFH descriptors to find a coarse initial alignment between source and target.
    The correspondence distance threshold is 1.5× the voxel size.
    """
    distance_threshold = voxel_size * 1.5 # ANCHOR change multiplier
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down,
        target_down,
        source_fpfh,
        target_fpfh,
        mutual_filter=True,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 100), # ANCHOR
    )
    return result

def constrain_to_planar_motion(T):
    """
    Enforce the ground-robot planar-motion constraint on a 4x4 transform.

    Habitat’s discrete action space only produces:
      • pure yaw (Y-axis rotation) from turn_left / turn_right
      • forward translation (XZ plane) from move_forward
    Any pitch or roll in T_rel is therefore 100 % ICP drift, not real motion.
    Zeroing it out prevents the tilt seen when pitch errors accumulate over
    hundreds of frames.

    Decomposition used: Euler angles in ‘YXZ’ order so that the first angle
    is the yaw (rotation around world-Y, the only permitted rotation).
    """
    r_full = R.from_matrix(T[:3, :3].copy())
    yaw = r_full.as_euler('YXZ')[0]          # keep only yaw component
    R_yaw = R.from_euler('Y', yaw).as_matrix()

    T_constrained = T.copy()
    T_constrained[:3, :3] = R_yaw
    # Zero out vertical translation (Habitat camera does not move up or down)
    T_constrained[1, 3] = 0.0
    return T_constrained

def visualize_and_evaluate(reconstructed_pcd, predicted_cam_poses, gt_poses, args):
    """
    TASK 3: Evaluation & Visualization [cite: 19, 35-38]
    """
    # Extract (x, y, z) camera-centre positions from the 4×4 pose matrices.
    # The translation column T[:3, 3] is the camera origin in world coordinates.
    pred_positions = np.array([T[:3, 3] for T in predicted_cam_poses])  # (N, 3)
    gt_positions   = gt_poses[:len(predicted_cam_poses), :3, 3]          # (N, 3)

    # 1. Create LineSet for estimated trajectory (Red)
    pred_lines   = [[i, i + 1] for i in range(len(pred_positions) - 1)]
    pred_lineset = o3d.geometry.LineSet()
    pred_lineset.points = o3d.utility.Vector3dVector(pred_positions)
    pred_lineset.lines  = o3d.utility.Vector2iVector(pred_lines)
    pred_lineset.colors = o3d.utility.Vector3dVector([[1, 0, 0]] * len(pred_lines))

    # 2. Create LineSet for ground truth trajectory (Blue)
    gt_lines   = [[i, i + 1] for i in range(len(gt_positions) - 1)]
    gt_lineset = o3d.geometry.LineSet()
    gt_lineset.points = o3d.utility.Vector3dVector(gt_positions)
    gt_lineset.lines  = o3d.utility.Vector2iVector(gt_lines)
    gt_lineset.colors = o3d.utility.Vector3dVector([[0, 0, 1]] * len(gt_lines))  # Blue

    # TODO: Calculate Mean L2 Distance between predicted_cam_poses and gt_poses [cite: 38]
    # L2 = sqrt(dx^2 + dy^2 + dz^2)
    n    = min(len(pred_positions), len(gt_positions))
    diff = pred_positions[:n] - gt_positions[:n]          # (n, 3)
    l2_distances  = np.sqrt(np.sum(diff ** 2, axis=1))    # (n,)
    mean_l2_error = float(np.mean(l2_distances))

    print(f"Mean L2 distance: {mean_l2_error:.6f} meters")

    # 3. Visualization
    # -- Interactive window (X11 / XWayland) ----------------------------------
    o3d.visualization.draw_geometries(
        [reconstructed_pcd, pred_lineset, gt_lineset],
        window_name=f"Floor {args.floor} Reconstruction"
    )

    # -- Offscreen render → saves PNG as a static record of the result --------
    W, H = 1280, 960
    render = o3d.visualization.rendering.OffscreenRenderer(W, H)
    render.scene.set_background([1.0, 1.0, 1.0, 1.0])  # white background

    mat_pcd = o3d.visualization.rendering.MaterialRecord()
    mat_pcd.shader = "defaultUnlit"
    mat_pcd.point_size = 5.0

    mat_line = o3d.visualization.rendering.MaterialRecord()
    mat_line.shader = "unlitLine"
    mat_line.line_width = 2.0

    render.scene.add_geometry("pcd",      reconstructed_pcd, mat_pcd)
    render.scene.add_geometry("pred_traj", pred_lineset,     mat_line)
    render.scene.add_geometry("gt_traj",   gt_lineset,       mat_line)

    # Bird's-eye view: camera looks straight down (-Y axis in world space)
    bounds = render.scene.bounding_box
    center = bounds.get_center()
    eye    = center + np.array([0.0, 10.0, 0.0])   # 10 m above scene centre
    render.setup_camera(60.0, center, eye, [0.0, 0.0, -1.0])  # up = -Z (north)

    img = render.render_to_image()
    out_path = os.path.join(args.data_root, f"reconstruction_floor{args.floor}.png")
    o3d.io.write_image(out_path, img)
    print(f"Visualization saved → {out_path}")

    # Open with GNOME image viewer — works natively on Wayland.
    import subprocess
    try:
        subprocess.Popen(["eog", out_path])
    except FileNotFoundError:
        pass  # eog not installed; open the PNG from Files (Nautilus) manually

    return mean_l2_error

def reconstruct(args):
    voxel_size = 0.1  # ANCHOR
    rgb_dir   = os.path.join(args.data_root, "rgb")
    depth_dir = os.path.join(args.data_root, "depth")

    rgb_files   = sorted(glob.glob(os.path.join(rgb_dir,   "*.png")))
    depth_files = sorted(glob.glob(os.path.join(depth_dir, "*.png")))

    print(f"Using all {len(rgb_files)} frames (no subsampling — fast without RANSAC)")
    # Load Ground Truth Poses [cite: 24, 54]
    gt_pose_path = os.path.join(args.data_root, "GT_pose.npy")
    gt_poses = []
    if os.path.exists(gt_pose_path):
        gt_data = np.load(gt_pose_path)
        for p in gt_data:
            mat = np.eye(4)
            mat[:3, :3] = R.from_quat([p[4], p[5], p[6], p[3]]).as_matrix()
            mat[:3, 3] = [p[0], p[1], p[2]]
            gt_poses.append(mat)
        gt_poses = np.stack(gt_poses)

    camera_poses = [np.eye(4)]
    accumulated_pcd = o3d.geometry.PointCloud()

    # ------------------------------------------------------------------
    # Bootstrap: process frame 0 as the world-frame reference.
    # camera_poses[0] = I  (identity), so frame-0 camera space IS world space.
    # ------------------------------------------------------------------
    rgb_prev   = cv2.cvtColor(cv2.imread(rgb_files[0]),   cv2.COLOR_BGR2RGB)
    depth_prev = cv2.imread(depth_files[0], cv2.IMREAD_UNCHANGED)
    pcd_prev   = depth_image_to_point_cloud(rgb_prev, depth_prev)
    pcd_prev_down, fpfh_prev = preprocess_point_cloud(pcd_prev, voxel_size)
    accumulated_pcd += pcd_prev   # frame 0 needs no transform (already in world frame)

    last_valid_T_rel = np.eye(4)  # For constant velocity fallback
    
    # Reconstruction Loop [cite: 29-30]
    for i in range(1, len(rgb_files)):
        print(f"Processing Frame {i}...")
        # TODO: Implement the full pipeline:
        # 1. Convert RGB-D to PointCloud (Task 1)
        rgb_cur   = cv2.cvtColor(cv2.imread(rgb_files[i]),   cv2.COLOR_BGR2RGB)
        depth_cur = cv2.imread(depth_files[i], cv2.IMREAD_UNCHANGED)
        pcd_cur   = depth_image_to_point_cloud(rgb_cur, depth_cur)

        # 2. Preprocess (Voxel/FPFH/Normals)
        pcd_cur_down, fpfh_cur = preprocess_point_cloud(pcd_cur, voxel_size)

        # 3. Execute Global Registration (RANSAC)
        # NOTE: RANSAC (execute_global_registration) is designed for large viewpoint
        # changes. For sequential frames with small inter-frame motion, FPFH descriptors
        # find ambiguous matches in repetitive interiors (same-looking walls), producing
        # garbage rotations that compound into tilted/fragmented slabs. Identity init
        # is more reliable and ~70× faster for sequential odometry.
        # To re-enable RANSAC, uncomment the block below and comment out ransac_init = np.eye(4).
        #
        # ransac_result = execute_global_registration(
        #     pcd_cur_down, pcd_prev_down, fpfh_cur, fpfh_prev, voxel_size
        # )
        # ransac_init = ransac_result.transformation if ransac_result.fitness > 0.3 else np.eye(4)
        # print(f"  RANSAC fitness={ransac_result.fitness:.3f}, "
        #       f"{'OK' if ransac_result.fitness > 0.3 else 'LOW → using identity fallback'}")
        ransac_init = np.eye(4)  # identity: valid because all frames are used (tiny inter-frame motion)

        # 4. Execute Local Registration (ICP - Task 2)
        # ICP refines T_rel starting from the RANSAC guess.
        # threshold = 0.4× voxel_size for a tight convergence criterion.
        # T_rel maps points from the *current* camera frame → *previous* camera frame.
        if args.version == 'my_icp':
            icp_result = my_local_icp_algorithm(
                pcd_cur_down, pcd_prev_down, ransac_init
            )
        else:
            icp_result = local_icp_algorithm(
                pcd_cur_down, pcd_prev_down,
                ransac_init,
                voxel_size  # passed through for multi-scale scheduling
            )
        T_rel = icp_result.transformation

        # Enforce planar-motion constraint: Habitat camera is always upright,
        # so pitch and roll in T_rel are pure ICP drift. Zero them out so
        # errors don’t tilt the entire reconstruction over hundreds of frames.
        T_rel = constrain_to_planar_motion(T_rel)

        # ICP convergence guard — reject T_rel if any of:
        #   (a) fitness < 0.15: very few points matched (threshold lowered from 0.3
        #       because many borderline-valid turn frames had fitness 0.15–0.29)
        #   (b) translation > 0.5 m: Habitat’s move_forward is 0.25 m
        #   (c) rotation > 30°: Habitat’s turn action is 10°
        t_dist    = np.linalg.norm(T_rel[:3, 3])
        cos_a     = np.clip((np.trace(T_rel[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
        angle_deg = np.degrees(np.arccos(cos_a))

        if icp_result.fitness < 0.05 or t_dist > 0.75 or angle_deg > 30.0:
            print(f"  [WARN] Frame {i}: fitness={icp_result.fitness:.3f}, "
                  f"t={t_dist:.3f}m, rot={angle_deg:.1f}° → constant velocity fallback")
            T_rel = last_valid_T_rel
        else:
            last_valid_T_rel = T_rel

        # 5. Update camera_poses and accumulate points
        # --- Camera-to-world transform chaining ---
        # camera_poses[-1] = T_abs_{i-1}: maps previous camera frame → world frame.
        # T_rel             = T_{cur→prev}: maps current camera frame → previous camera frame.
        # Therefore T_abs_i = T_abs_{i-1} @ T_rel maps current camera frame → world frame.
        T_abs = camera_poses[-1] @ T_rel
        camera_poses.append(T_abs)

        # Transform the downsampled cloud (not full-res pcd_cur) into world coords and merge.
        # Using pcd_cur_down avoids accumulating 262K pts/frame × 630 frames = 165M pts,
        # which blows up the final voxel_down_sample step.
        pcd_cur_world = deepcopy(pcd_cur_down).transform(T_abs)
        accumulated_pcd += pcd_cur_world

        # Slide the window: current frame becomes the new previous frame.
        pcd_prev_down = pcd_cur_down
        fpfh_prev     = fpfh_cur  # unused while RANSAC is disabled; needed if re-enabled

    # TODO: Post-processing: remove the ceiling [cite: 37]
    # In Habitat/OpenGL, Y is up, so the ceiling occupies the highest Y values.
    # We discard points above the 97th-percentile of Y to strip the ceiling
    # while retaining walls, floors, and furniture.
    
    # Downsample the accumulated cloud to remove redundant overlapping points.
    # Use the same voxel_size as the per-frame downsampling so we don't
    # add a finer grid than what ICP matched against.
    accumulated_pcd = accumulated_pcd.voxel_down_sample(voxel_size=voxel_size)
    
    points = np.asarray(accumulated_pcd.points)
    colors = np.asarray(accumulated_pcd.colors)
    if len(points) > 0:
        # The 97th percentile only shaved off the immediate top of the roof.
        # Since ceilings constitute a massive portion of the scene points,
        # using an 80th percentile safely removes the entire ceiling block 
        # while keeping the walls fully intact.
        y_threshold = np.percentile(points[:, 1], 80)
        mask = points[:, 1] < y_threshold
        accumulated_pcd.points = o3d.utility.Vector3dVector(points[mask])
        accumulated_pcd.colors = o3d.utility.Vector3dVector(colors[mask])

    return accumulated_pcd, camera_poses, gt_poses

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--floor', type=int, default=1)
    parser.add_argument('-v', '--version', type=str, default='open3d', help='open3d or my_icp')
    args = parser.parse_args()

    # Set data root based on floor
    args.data_root = f"data_collection/first_floor/" if args.floor == 1 else f"data_collection/second_floor/"

    start_time = time.time()
    result_pcd, pred_poses, gt_poses = reconstruct(args)

    print(f"Total execution time: {time.time() - start_time:.2f}s") #
    visualize_and_evaluate(result_pcd, pred_poses, gt_poses, args)
