import os
os.environ.setdefault('DISPLAY', ':0')
import re
import glob
import cv2
import numpy as np
import open3d as o3d
import open3d.core as o3c
import argparse
from copy import deepcopy
from scipy.spatial.transform import Rotation as R
import time

# ---------- Camera Intrinsics (Resolution 512x512, FOV 90) ----------
# These parameters are derived from the Habitat pinhole camera model [cite: 26-27].
IMG_W, IMG_H = 512, 512
FOV = np.deg2rad(90.0)
FX = (IMG_W / 2.0) / np.tan(FOV / 2.0)
FY = (IMG_H / 2.0) / np.tan(FOV / 2.0)
CX, CY = IMG_W / 2.0, IMG_H / 2.0

def depth_image_to_point_cloud(rgb_image, depth_image):
    """
	SECTION PIPEINE PART A: Unprojecting Depth Images
    TASK 1: Geometric Unprojection [cite: 12, 25-27]
    Convert depth pixels (u, v, d) into 3D world points (x, y, z).

	The goal of this function is to convert a flat 2D depth image into a 3D Point Cloud.
	A normal color image gives us a 2D grid of RGB pixels, but no sense of distance.
	A depth image is a companion 2D grid where each pixel stores physical distance (in meters).
	
	By combining the 2D pixel location (u, v) with its physical depth distance (d), and
	running it through the Camera Intrinsics mathematical formulas (Focal Length and Center),
	we can "unproject" the flat pixel back out into the real world to find its exact
	3D Cartesian coordinates (X, Y, Z). 
	
	We perform this mapping across all 262,000 pixels simultaneously using NumPy grids
	to instantly generate the 3D shell of the room, and then paint each 3D point using 
	the exact color from the original RGB image.
    """

	# 1. Convert RBG and depth images to numpy arrays
    rgb   = np.array(rgb_image, dtype=np.float64) # Just there to paint color on the point cloud at the end of this function.
    depth = np.array(depth_image, dtype=np.float64)
	
    # 2. Convert depth to meters (Habitat depth is often scaled or normalized)
    depth_m = depth / 255.0 * 10.0

    # 3. Create a coordinate grid for (u, v) pixels (for that particular image)
    u_coords = np.arange(IMG_W) # Generates 0, 1, 2... 512
    v_coords = np.arange(IMG_H) # Generates 0, 1, 2... 512
    uu, vv   = np.meshgrid(u_coords, v_coords) # Sitches them into a 2D coordinate depth grid

    # Filter out pixels with zero / invalid depth before indexing
    valid = depth_m > 0
	# NOTE:  Why is this necessary?
	# 	 	 Without filtering 0 depth pixels, the code will accidentally plot "fake" points.
	#        These "fake" points are located at infinity, and will mess up the point cloud.

    # TODO: Implement unprojection logic here
    # x = (u - CX) * z / FX
    # y = (v - CY) * z / FY
    # z = -depth (assuming camera looks towards -Z)
	
	# 4. Unproject the 2D pixels into 3D Cartesian space
    # Calculate Z-axis: The physical distance pointing straight away from the camera.
    # We use negative depth because Habitat uses an OpenGL coordinate system 
    # where the camera physically looks down the negative Z-axis (-Z).
    z_cam =  -depth_m[valid]

    # Calculate X-axis (Horizontal): How far left or right the point is.
    # We subtract CX to find how many pixels the point is from the optical center of the lens, 
    # then multiply by scaled distance and divide by focal length (FX) to convert to meters.
    x_cam =  (uu[valid] - CX) * depth_m[valid] / FX

    # Calculate Y-axis (Vertical): How far up or down the point is.
    # The negative sign flips the image pixel rows (which grow downwards sequentially) so that
    # the 3D Y-axis correctly points "UP" towards the ceiling in the real world.
    y_cam = -(vv[valid] - CY) * depth_m[valid] / FY

	# 5. Stack the 3D coordinates and paint the RGB colors
    # Combines the separate X, Y, and Z_cam into a single massive list of [X, Y, Z] points.
    points_3d  = np.stack([x_cam, y_cam, z_cam], axis=1)
    
    # Normalizes the raw color values from 0-255 down to 0.0-1.0 (required by Open3D)
    colors_norm = rgb[valid] / 255.0 

    # 6. Store the raw NumPy results inside an Open3D PointCloud container
    # Open3D's system requires points to be stored in their own Vector3dVector format. 
    # So we just dump our raw NumPy calculations into the Open3D bucket here.
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d) # Store the 3D coordinates
    pcd.colors = o3d.utility.Vector3dVector(colors_norm) # Assign the normalized colors to the points

    return pcd

def preprocess_point_cloud(pcd, voxel_size, device):
    """
	SECTION PIPELINE PART B: Voxelization (Downsampling) and Normals Estimation
    TASK 2: Pre-processing: Voxelization and Normals Estimation [cite: 17, 29]

	The depth_image_to_point_cloud function produces a raw 3D point cloud that 
	contains around 262,000 points. Aligning one 3D point cloud (~262,000 points) 
	with another 3D point cloud (another ~262,000 points) would make my laptop 
	crash and burn (figuritively). This is where voxelization comes in.

	Normals estimation is where we calculate the direction of the surface at each point. 
	Normals are invisible 3D arrows pointing straight out of every dot, 
	pointing away from the surface, which helps the computer understand how to orient 
	the point cloud against the previous point cloud.
    """

	# Applies a 3D grid (0.1m x 0.1m x 0.1m) over the raw 3D point cloud 
	# and keeps only one point per grid cell.
    pcd_down = pcd.voxel_down_sample(voxel_size) 

    # TODO: Estimate normals for pcd_down (required for Point-to-Plane ICP)
    radius_normal = voxel_size * 2.0 # ANCHOR: parameter (original: voxel_size * 5.0)

	# Defines a search radius (0.5m for original voxel_size) and the number of neighbors (30) 
	# to use for estimating the normal at each point. 
	# If there are thousands of points in the search radius, it will use the 30 closest points 
	# to balance speed and accuracy.
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)) 

	# Orients the normals so they point outwards from the camera.	
    pcd_down.orient_normals_towards_camera_location(np.array([0., 0., 0.])) 

	# Converts the point cloud to a tensor for GPU acceleration.
    tpcd_down = o3d.t.geometry.PointCloud.from_legacy(pcd_down, device=device) 

	# # Compute FPFH features for Global Registration [cite: 30]
	# radius_feature = voxel_size * 5.0
	# pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
	# 	pcd_down,
	# 	o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
	# )
	# NOTE: Gave up on FPFH since it was causing too much problems.
	#       FPFH works by looking at the new point cloud, finds a cluster of points 
	# 		that resembles a corner, and searches all the previous point clouds that 
	# 		resembles a corner (or some other feature of a room) to help piece the 
	# 		new point cloud in reference to the previous ones (kinda like a lego). 
	# 		However, there's an issue: it assumes that the robot teleported around to 
	# 		find that corner and has no clue where the robot could be. It's even worse 
	# 		when the robot walks through long, featureless hallways. It started to think that 
	# 		the walls at the end were the same walls it saw at the beginning. This led to 
	# 		random jumps in the point cloud, which caused a the reconstruction to 
	# 		look jumbled af. I was this close to crashing out.
	# 		
	# 		My friend suggested to use the local tracking instead. Local tracking works by 
	# 		looking at the current point cloud and the previous point cloud and finding the 
	# 		transformation that best aligns them. It's like trying to fit two puzzle pieces 
	# 		together. It's much better than FPFH because it doesn't assume that the robot 
	# 		has teleported around to find that corner, removing the random jumps in the point cloud.
	#
	#		FPFH is more preferrable when we are taking random disjoined photos/point clouds of 
	# 		the room and have no idea where the agent would be in the room at a particular point in time.
 
    return pcd_down, tpcd_down # Returns the downsampled point cloud and the tensor version of it.

def constrain_to_planar_motion(T):
    """ 
	SECTION HELPER FUNCTION: Enforce the ground-robot planar-motion constraint
	
	Why did we have to include this?
	Standard ICP is designed for free flying objects like drones or handheld cameras. 
	It calculates movement across all 6 Degrees of Freedom (X, Y, Z translation, and Pitch, Yaw, Roll).
	
	However, our apartment data was collected by a ground-based robot agent that moves on a flat surface. 
	The robot does not jump into the air (Y-translation) or tilt sideways (Pitch/Roll). 
	
	So if we let standard ICP run, tiny mathematical estimating mistakes 
	(like guessing that there's a 0.05-degree tilt) accumulate. Over many frames, the 0.05-degree tilt 
	compounds into 30+ degrees. This caused the reconstruction to warp (at that point I was starting 
	to give up and question my life choices) where the floor started to slowly move up (and affecting 
	the rest of the reconstruction).
	
	It works by getting a rough estimate of the transformation from ICP (which is a wild 6-DoF guess), 
	and removing the vertical movement (Y-translation) and the tilt (Pitch/Roll). 
	This ensures the  reconstructed house stays flat over arbitrary distances.

	The transformation matrix (T) looks like this:
	[ R11, R12, R13,   Tx ]
	[ R21, R22, R23,   Ty ]
	[ R31, R32, R33,   Tz ]
	[   0,   0,   0,    1 ]
	"""
    # 1. Isolate the 3x3 Rotation Submatrix (the R11-R33) from the 4x4 Transformation Matrix (T).
    # Use scipy's Rotation (R) module to mathematically parse the 3D rotation.
    r_full = R.from_matrix(T[:3, :3].copy())
    
    # 2. Extract the true "Yaw" (Left/Right turning) value.
    # Convert the full 3D rotation into separated YXZ Euler angles (Y=12deg). 
    # Because 'Y' is the vertical axis in our world, rotating around 'Y' means horizontal turning (yaw).
    # We extract [0] (the Y angle), throwing away the Pitch(X) and Roll(Z) angles.
    yaw = r_full.as_euler('YXZ')[0]
    
    # 3. Create a new flat rotation matrix utilizing ONLY the Yaw value.
    R_yaw = R.from_euler('Y', yaw).as_matrix()
	# Fill the rest with 0 so it looks like this:
	# [  cos(Yaw),   0,   sin(Yaw) ]
	# [         0,   1,          0 ]
	# [ -sin(Yaw),   0,   cos(Yaw) ]

    # 4. Splice the constrained flat rotation back into a copy of the Transformation Matrix.
    T_constrained = T.copy()
    T_constrained[:3, :3] = R_yaw
	# [  cos(Yaw),         0,  sin(Yaw),      Tx ]
	# [         0,         1,         0, *Wild_Ty* ]  <-- The Tilt is fixed, but it is STILL FLYING!
	# [ -sin(Yaw),         0,  cos(Yaw),      Tz ]
	# [         0,         0,         0,       1 ]
    
    # 5. Remove vertical flying (Altitude/Y Translation).
    # The [1, 3] slot in a 4x4 transformation matrix controls sliding Up and Down. 
    # Forcing it to 0.0 vertically pins the robot perfectly to the floor.
    T_constrained[1, 3] = 0.0
	# [  cos(Yaw),         0,  sin(Yaw),      Tx ]
	# [         0,         1,         0,     *0.0* ]  <-- The Tilt is fixed, and it is NO LONGER FLYING!
	# [ -sin(Yaw),         0,  cos(Yaw),      Tz ]
	# [         0,         0,         0,       1 ]

    return T_constrained # Return the constrained transformation matrix.

# give up on the my_local_icp_algorithm orz (i just deleted the function)

def local_icp_algorithm_tensor(source_tpcd, target_tpcd, trans_init, voxel_size):
    """
	SECTION PIPELINE PART C: Local ICP 
	TASK 3: Open3D ICP Implementation (REQUIRED) [cite: 32]

	This is where we connect the lego bricks (connect two consecutive point clouds together).
	Standard Open3D uses a system called "TransformationEstimationPointToPlane()", which 
	tends to slide uncontrollably along featureless corridors because it has no mathematical 
	penalty for sliding a flat plate against an infinitely long flat wall.
	
	I managed to solve this by doing...
	1. Bounding the translation limits later in the code (so the slide is mathematically capped).
	2. Using the `o3d.t` Tensor API. 

	Instead of running the heavy mathematics on the CPU, 
	the Tensor API runs the entire ICP tracking algorithm directly on the GPU. 
	This makes the alignment more stable, allowing 
	a larger search radius (voxel_size * 5.0) to catch wild turns without lagging.
    """

	# TODO: Use o3d.pipelines.registration.registration_icp
	# Estimation method should be TransformationEstimationPointToPlane()

    # 1. Convert the starting guess (trans_init) into a GPU Tensor format. 
    # ICP needs a rough mathematical starting point (like "the robot moved forward 0.25m") 
    # to begin the snapping process, otherwise it searches randomly.
    init_tensor = o3c.Tensor(trans_init, dtype=o3c.float64)
	# trans_init is the initial guess of the transformation matrix. Looks like this when it's testing T_fwd:
	# [[ 1.0,  0.0,  0.0,   0.0  ],
	#  [ 0.0,  1.0,  0.0,   0.0  ],
	#  [ 0.0,  0.0,  1.0,  *-0.25* ],  <-- -0.25m step on the Z-axis
	#  [ 0.0,  0.0,  0.0,   1.0  ]]
	#
	# init_tensor is a GPU Tensor that represents the initial guess of the transformation matrix.
	# It looks the same as trans_init, just that the data is stored on the GPU instead of the CPU.

    # 2. Run the actual Tensor-based ICP algorithm on the GPU
    result = o3d.t.pipelines.registration.icp(
        source_tpcd, # The new puzzle piece (Current Point Cloud)
        target_tpcd, # The puzzle board (Previous Point Cloud)
        
        # Search Radius (voxel_size * 5.0). This is how far the tracking algorithm will "reach out" 
        # to find matching walls. Because we use the Tensor GPU API, we can use large 
        # 0.5m radius to catch wild 10-degree turns without lagging the computer!
        voxel_size * 5.0, # ANCHOR: parameter
        
        init_tensor, # The initial physical guess we prepared in step 1
        
        # Point-to-Plane metric: Uses the arrows (normals) we calculated earlier.
        # Instead of treating both point clouds as Lego grids that must snap perfectly 
        # peg-to-peg (which causes jagged twisting), it uses the normals to turn the 
        # previous point cloud into a smooth surface. The new points can now 
        # effortlessly slide along the flat surface until the room corners perfectly align!
        o3d.t.pipelines.registration.TransformationEstimationPointToPlane(),
        
        # Convergence Criteria: Stop the sliding loop if the points stop shifting 
        # (difference of 1e-6), or if the loop simply spins out and hits 100 tries.
		# Without this, the legos will never snap together.
        o3d.t.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=100) # ANCHOR: parameter max_iteration original=100
    )
    
    # Returns the physical transformation matrix that mathematically locks the two point clouds
    return result

def visualize_and_evaluate(reconstructed_pcd, predicted_cam_poses, gt_poses, args):
    """
    SECTION PIPELINE PART D: Evaluation & Visualization

    Now that the main loop has successfully finished tracking the robot, this function 
    grades our accuracy by comparing our tracked path against the absolute "Ground Truth" 
    path saved by the Simulator backend. 

    It mathematically compares the (X, Y, Z) coordinates of our path to the true path, 
    and prints out the final L2 Error. After that, it sets up a virtual camera floating 
    above the reconstructed 3D house and takes a BEV pic.
    """
	
    # 1. Extract the physical (X,Y,Z) positions from the 4x4 transformation matrices 
	# (the constrained one from constrain_to_planar_motion function).
    # The [:, 3] column inside a matrix always holds the pure translation (location).
    pred_positions = np.array([T[:3, 3] for T in predicted_cam_poses]) 
    gt_positions   = gt_poses[:len(predicted_cam_poses), :3, 3]

    # 2. Draw the "Predicted Path" (The path our ICP algorithm tracked)
    # We connect dot 0 to dot 1, dot 1 to dot 2, etc, to form a continuous solid line.
    pred_lines   = [[i, i + 1] for i in range(len(pred_positions) - 1)]
    pred_lineset = o3d.geometry.LineSet()
    pred_lineset.points = o3d.utility.Vector3dVector(pred_positions)
    pred_lineset.lines  = o3d.utility.Vector2iVector(pred_lines)
    pred_lineset.colors = o3d.utility.Vector3dVector([[1, 0, 0]] * len(pred_lines)) # Paint the predicted line RED

    # 3. Draw the "Ground Truth Path" (The absolute perfect path from the simulator)
    gt_lines   = [[i, i + 1] for i in range(len(gt_positions) - 1)]
    gt_lineset = o3d.geometry.LineSet()
    gt_lineset.points = o3d.utility.Vector3dVector(gt_positions)
    gt_lineset.lines  = o3d.utility.Vector2iVector(gt_lines)
    gt_lineset.colors = o3d.utility.Vector3dVector([[0, 0, 0]] * len(gt_lines)) # Paint the true perfect line BLACK

    # Ensure both arrays are the same length (in case tracking broke early)
    n    = min(len(pred_positions), len(gt_positions))
    diff = pred_positions[:n] - gt_positions[:n]

    # Calculate L2 Distance
    l2_distances  = np.sqrt(np.sum(diff ** 2, axis=1))
    mean_l2_error = float(np.mean(l2_distances))

    print(f"Mean L2 distance: {mean_l2_error:.6f} meters")

    o3d.visualization.draw_geometries(
        [reconstructed_pcd, pred_lineset, gt_lineset],
        window_name=f"Floor {args.floor} Reconstruction (Optimized)"
    )

    # 5. Set up the Photo Studio (Offscreen Renderer)
    # The Offscreen Renderer generates a high-res image behind the scenes without opening a window.
    W, H = 1280, 960 # Set the final image resolution
    render = o3d.visualization.rendering.OffscreenRenderer(W, H)
    render.scene.set_background([1.0, 1.0, 1.0, 1.0]) # Set the empty void background to completely white

    # Define how the Point Cloud (the house walls/floor) should look. 
    # point_size is increased to 5.0 to merge the dots together into a solid-looking floor.
    mat_pcd = o3d.visualization.rendering.MaterialRecord()
    mat_pcd.shader = "defaultUnlit"
    mat_pcd.point_size = 10.0 # ANCHOR play around with this? cosmetic

    # Define how the Trajectory lines (The Red and Black paths) should look.
    mat_line = o3d.visualization.rendering.MaterialRecord()
    mat_line.shader = "unlitLine"
    mat_line.line_width = 3.0 # ANCHOR play around with this? cosmetic

    # Drop the Point Cloud and the colored Lines into the invisible Photo Studio scene
    render.scene.add_geometry("pcd",      reconstructed_pcd, mat_pcd)
    render.scene.add_geometry("pred_traj", pred_lineset,     mat_line)
    render.scene.add_geometry("gt_traj",   gt_lineset,       mat_line)

    # Render Settings: Position a virtual camera high above the very center of the house
	# ANCHOR Angle the camera for the saved image. Pic kinda trash (doesn't cover the whole house), so imma work on this after submission.
    bounds = render.scene.bounding_box
    center = bounds.get_center()
    eye    = center + np.array([0.0, 20.0, 0.0]) # Raise camera 20 meters artificially straight up into the air
    render.setup_camera(60.0, center, eye, [0.0, 0.0, -1.0]) # FOV=60, Look straight down at the floor

    # 7. Snap the picture and save it to the hard drive in your data folder
    img = render.render_to_image()
    out_path = os.path.join(args.data_root, f"reconstruction_optimized_floor{args.floor}.png")
    o3d.io.write_image(out_path, img)
    print(f"Visualization saved -> {out_path}")

    return mean_l2_error

def reconstruct(args):
    """
    SECTION PIPELINE PART E: Main Reconstruction Loop

    This function acts as the conductor of the entire pipeline. It loads the images, 
    triggers the unprojection (Part A), runs voxelization (Part B), and sequentially 
    tracks the camera's path across all frames using Local ICP (Part C).

    The core optimization here is the "Multi-Hypothesis Tracker" paired with a 
    "Constant Velocity Fallback". Because we chose to skip Global Registration, 
    we must manually give ICP a hint on where to start. We explicitly test 5 actions 
    per frame: (Stay Still, Move Forward, Turn Left, Turn Right, or Repeat Last Move). 
    Whichever physical guess generates the tightest interlocking point cloud is chosen!

    If tracking completely fails (e.g. staring at a blank featureless wall), the system 
    triggers the fail-safe and assumes the robot simply kept moving at a "Constant Velocity".
    """
    voxel_size = 0.1 # 0.1 (original) 0.05 (smaller, more accurate but slower; most important parameter); in meters ANCHOR: parameter
    rgb_dir   = os.path.join(args.data_root, "rgb")
    depth_dir = os.path.join(args.data_root, "depth")

    rgb_files   = sorted(glob.glob(os.path.join(rgb_dir,   "*.png")), key=lambda x: int(os.path.basename(x).split(os.sep)[-1].split('.')[0]))
    depth_files = sorted(glob.glob(os.path.join(depth_dir, "*.png")), key=lambda x: int(os.path.basename(x).split(os.sep)[-1].split('.')[0]))

    print(f"Collected {len(rgb_files)} frames for reconstruction.")

    # Optimized Tensor Device Discovery
    if o3c.cuda.is_available():
        device = o3c.Device("cuda:0")
        print("=> Using GPU (CUDA) for Tensor ICP")
    else:
        device = o3c.Device("cpu:0")
        print("=> Using CPU for Tensor ICP")

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

    # Initialize the global trajectory (starts at origin 0,0,0)
    camera_poses = [np.eye(4)]
    # Initialize the global point cloud (starts as empty 3D room)
    accumulated_pcd = o3d.geometry.PointCloud()

    # Bootstrap (Anchor) Frame 0 as the starting point. 
	# This is the first lego piece before we enter the for-loop to add more lego pieces.
    rgb_prev   = cv2.cvtColor(cv2.imread(rgb_files[0]),   cv2.COLOR_BGR2RGB) # Load rgb image
    depth_prev = cv2.imread(depth_files[0], cv2.IMREAD_UNCHANGED) # Load depth image
    pcd_prev   = depth_image_to_point_cloud(rgb_prev, depth_prev) # Convert depth image to 3D point cloud
    pcd_prev_down, tpcd_prev = preprocess_point_cloud(pcd_prev, voxel_size, device) # Downsample and estimate normals
    accumulated_pcd += pcd_prev_down # Add the first point cloud to the global point cloud

    last_valid_T_rel = np.eye(4)

    # Pre-calculate the physical action guesses (Multi-Hypothesis Transforms)
    # The Habitat Simulator steps are exactly 0.25m and turns are exactly 10 degrees.
    T_fwd = np.eye(4); T_fwd[2, 3] = -0.25 # Guess: Robot stepped forward 0.25m down the -Z axis
    T_L   = np.eye(4); T_L[:3, :3] = R.from_euler('Y',  np.deg2rad(10)).as_matrix() # Guess: Robot turned Left 10 degrees
    T_R   = np.eye(4); T_R[:3, :3] = R.from_euler('Y', -np.deg2rad(10)).as_matrix() # Guess: Robot turned Right 10 degrees

    # Reconstruction Loop [cite: 29-30]
    for i in range(1, len(rgb_files)):
        t0 = time.time()

        rgb_cur   = cv2.cvtColor(cv2.imread(rgb_files[i]),   cv2.COLOR_BGR2RGB)
        depth_cur = cv2.imread(depth_files[i], cv2.IMREAD_UNCHANGED)
        pcd_cur   = depth_image_to_point_cloud(rgb_cur, depth_cur)
        pcd_cur_down, tpcd_cur = preprocess_point_cloud(pcd_cur, voxel_size, device)

        # Iterate multiple hypotheses explicitly providing discrete action priors
        hypotheses = [last_valid_T_rel, T_fwd, T_L, T_R, np.eye(4)]
        
        best_icp = None
        best_fit = -1.0
        # Iterate through all 5 hypotheses to see which one mathematically snaps best
        for h_init in hypotheses:
            # Run the Tensor ICP on the GPU using the current guess (h_init)
            h_res = local_icp_algorithm_tensor(tpcd_cur, tpcd_prev, h_init, voxel_size)
            
            # Extract the raw translation distance (h_t) and rotation angle (h_rot) that ICP suggested
            h_T = h_res.transformation.cpu().numpy() # hypothesis transformation matrix
            h_t = np.linalg.norm(h_T[:3, 3]) # hypothesis translation distance
            h_cos = np.clip((np.trace(h_T[:3, :3]) - 1.0) / 2.0, -1.0, 1.0) # hypothesis rotation matrix
            h_rot = np.degrees(np.arccos(h_cos)) # hypothesis rotation angle

            # STRICT KINEMATIC BOUNDING: The absolute secret weapon of this script!
			# Maintain strict error thresholds to prevent drift 
			# (like the same hall being replicated a bunch of times just because the agent was turning
			# ...nearly drove me insane)
            # Since the robot physically can't move more than 0.25m or 10-degrees per frame,
            # we just reject ANY tracked suggestion from ICP that exceeds 0.32m or 12 degrees.
            if h_res.fitness > best_fit and h_t <= 0.32 and h_rot <= 12.0:
                best_icp = h_res
                best_fit = h_res.fitness
        # Ensure the chosen alignment is actually reasonable (fitness >= 10% overlap)
        if best_icp is not None and best_fit >= 0.10: # Gonna be overwritten anyway. But I don't wanna delete it cuz the code works...lol
            T_rel = best_icp.transformation.cpu().numpy()
        else:
            # CONSTANT VELOCITY FALLBACK: If ICP completely failed to find walls, 
            # assume the robot just kept doing whatever physical movement it did in the previous frame.
			# I notice that this option for fallback occurs when there are multiple images of the same wall
			# and the robot is just turning in place...learned that the hard way during data collection,
			# which I will explain further in the report :/
            T_rel = last_valid_T_rel

        # Apply the helper function to mathematically pin the robot back down to the surface.
        T_rel = constrain_to_planar_motion(T_rel)
        
        t_dist    = np.linalg.norm(T_rel[:3, 3])
        cos_a     = np.clip((np.trace(T_rel[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
        angle_deg = np.degrees(np.arccos(cos_a))

        if best_fit < 0.2 or t_dist > 0.60 or angle_deg > 25.0: #0.10 is original value for points overlap, 25.0 is the original angle value
            T_rel = last_valid_T_rel
			# We know that the robot can't move more than 0.25m or 10-degrees per frame,
			# so if ICP suggests something else, it's wrong, and we set T_rel to whatever action we did in the last valid frame.
        else:
            last_valid_T_rel = T_rel
        # Global Pose Chaining
        # Mathematically multiply the old global position by the new relative movement 
        # to calculate exactly where the robot is globally standing now
        T_abs = camera_poses[-1] @ T_rel
        camera_poses.append(T_abs)

        # Physically move the new 3D points to their global coordinates and merge them into the master map
        try:
            accumulated_pcd += deepcopy(pcd_cur_down).transform(T_abs)
        except Exception as e:
            print(f"  [WARN] Failed to accumulate frame {i}: {e}")

        tpcd_prev = tpcd_cur
        dt = time.time() - t0
        print(f"  Frame {i:03d} done | fitness: {best_fit:.3f} | time: {dt:.3f}s")

    # TODO: Post-processing: remove the ceiling [cite: 37]
    accumulated_pcd = accumulated_pcd.voxel_down_sample(voxel_size=voxel_size)
    points = np.asarray(accumulated_pcd.points)
    colors = np.asarray(accumulated_pcd.colors)
    if len(points) > 0:
        # Calculate the 80th percentile height. Because the camera mostly looks straight forward, 
        # the highest 20% of dots in the Y-axis are mathematically guaranteed to be the ceiling.
        y_threshold = np.percentile(points[:, 1], 80)
        
        # Create a boolean mask keepings ONLY points BELOW that height (the walls and floor)
        mask = points[:, 1] < y_threshold

        # Permanently overwrite the master point cloud with the trimmed mesh 
        accumulated_pcd.points = o3d.utility.Vector3dVector(points[mask])
        accumulated_pcd.colors = o3d.utility.Vector3dVector(colors[mask])
    return accumulated_pcd, camera_poses, gt_poses

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--floor', type=int, default=1)
    parser.add_argument('-v', '--version', type=str, default='open3d', help='Maintained for compatibility')
    args = parser.parse_args()

	# Set data root based on floor
    args.data_root = f"data_collection/first_floor_191_frames/" if args.floor == 1 else f"data_collection/second_floor_191_frames/"

    start_time = time.time()
    result_pcd, pred_poses, gt_poses = reconstruct(args)

    print(f"\n======================================")
    print(f"Total execution time: {time.time() - start_time:.2f}s")
    visualize_and_evaluate(result_pcd, pred_poses, gt_poses, args)
