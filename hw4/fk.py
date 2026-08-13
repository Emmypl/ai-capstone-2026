import os, argparse, json
import numpy as np
import traceback

from scipy.spatial.transform import Rotation as R

def get_matrix_from_pose(pose) -> np.ndarray:
    """Convert a 6D/7D pose vector into a 4x4 homogeneous transform.

    Parameters
    ----------
    pose : list | tuple | np.ndarray
        Pose representation.
        - 6 elements: ``[x, y, z, rx, ry, rz]`` where ``r*`` is a rotation vector.
        - 7 elements: ``[x, y, z, qx, qy, qz, qw]`` quaternion.

    Returns
    -------
    np.ndarray
        A 4x4 homogeneous transform matrix in float64.

    Notes
    -----
    This helper is commonly used in robotics pipelines to switch between
    vector pose representation and matrix-based transform composition.
    """
    assert len(pose) in (6, 7), f'pose must contain 6 or 7 elements, but got {len(pose)}'
    pos_m = np.asarray(pose[:3], dtype=np.float64)

    if len(pose) == 6:
        rot_m = R.from_rotvec(np.asarray(pose[3:], dtype=np.float64)).as_matrix()
    else:
        rot_m = R.from_quat(np.asarray(pose[3:], dtype=np.float64)).as_matrix()

    ret_m = np.identity(4, dtype=np.float64)
    ret_m[:3, :3] = rot_m
    ret_m[:3, 3] = pos_m
    return ret_m

def get_pose_from_matrix(matrix, pose_size : int = 7) -> np.ndarray:
    """Convert a 4x4 homogeneous transform into a 6D or 7D pose vector.

    Parameters
    ----------
    matrix : list | tuple | np.ndarray
        4x4 transform matrix.
    pose_size : int, default=7
        Output pose format.
        - 6: ``[x, y, z, rx, ry, rz]`` (rotation vector)
        - 7: ``[x, y, z, qx, qy, qz, qw]`` (quaternion)

    Returns
    -------
    np.ndarray
        Pose vector in float64.

    Notes
    -----
    If the rotation block is numerically unstable (e.g., det <= 0), this
    function projects it to the nearest orthonormal matrix via SVD first.
    """
    mat = np.asarray(matrix, dtype=np.float64)
    assert pose_size in (6, 7), f'pose_size should be 6 or 7, but got {pose_size}'
    assert mat.shape == (4, 4), f'pose must contain 4 x 4 elements, but got {mat.shape}'

    pos = mat[:3, 3]

    # --- 安全生成旋轉 ---
    rot_mat = mat[:3, :3]
    try:
        if pose_size == 6:
            rot = R.from_matrix(rot_mat).as_rotvec()
        else:
            rot = R.from_matrix(rot_mat).as_quat()
    except ValueError:
        # 如果 det<=0，修正成最近正交矩陣
        U, _, Vt = np.linalg.svd(rot_mat)
        rot_mat_fixed = U @ Vt
        if pose_size == 6:
            rot = R.from_matrix(rot_mat_fixed).as_rotvec()
        else:
            rot = R.from_matrix(rot_mat_fixed).as_quat()

    return np.asarray(list(pos) + list(rot), dtype=np.float64)


def _acquire_isaac_debug_draw_interface():
    """Acquire the Isaac Sim persistent debug draw interface.

    Returns
    -------
    object | None
        Debug draw interface object if available; otherwise ``None``.

    Notes
    -----
    This function enables the extension at runtime and gracefully degrades if
    the extension is unavailable.
    """
    try:
        from isaacsim.core.utils.extensions import enable_extension
        enable_extension("isaacsim.util.debug_draw")
        from isaacsim.util.debug_draw import _debug_draw
        return _debug_draw.acquire_debug_draw_interface()
    except Exception:
        return None


def _draw_pose_axes_isaac(debug_draw, pose7d, axis_len=0.02, width=2.0,
                          color_x=(1.0, 0.0, 0.0, 1.0),
                          color_y=(0.0, 1.0, 0.0, 1.0),
                          color_z=(0.0, 0.0, 1.0, 1.0)):
    """Draw XYZ axes for a 7D pose using Isaac persistent debug lines.

    Parameters
    ----------
    debug_draw : object
        Interface returned by ``_acquire_isaac_debug_draw_interface``.
    pose7d : list | tuple | np.ndarray
        Pose vector ``[x, y, z, qx, qy, qz, qw]``.
    axis_len : float, default=0.02
        Axis line length in meters.
    width : float, default=2.0
        Rendered line width.
    color_x, color_y, color_z : tuple[float, float, float, float]
        RGBA colors for x/y/z axes.

    Returns
    -------
    None

    Notes
    -----
    Axis visualization helps students compare estimated and ground-truth end-
    effector orientation directly in the simulator.
    """
    if debug_draw is None:
        return
    T = get_matrix_from_pose(np.asarray(pose7d, dtype=np.float64))
    o = T[:3, 3]
    x_end = o + axis_len * T[:3, 0]
    y_end = o + axis_len * T[:3, 1]
    z_end = o + axis_len * T[:3, 2]

    debug_draw.draw_lines([tuple(o)], [tuple(x_end)], [color_x], [float(width)])
    debug_draw.draw_lines([tuple(o)], [tuple(y_end)], [color_y], [float(width)])
    debug_draw.draw_lines([tuple(o)], [tuple(z_end)], [color_z], [float(width)])

SIM_TIMESTEP = 1.0 / 240.0
JACOBIAN_SCORE_MAX = 10.0
JACOBIAN_ERROR_THRESH = 0.05
FK_SCORE_MAX = 10.0
FK_ERROR_THRESH = 0.005
TASK1_SCORE_MAX = JACOBIAN_SCORE_MAX + FK_SCORE_MAX

def cross(a : np.ndarray, b : np.ndarray) -> np.ndarray :
    """Compute the 3D vector cross product.

    Parameters
    ----------
    a : np.ndarray
        First 3D vector.
    b : np.ndarray
        Second 3D vector.

    Returns
    -------
    np.ndarray
        Cross product ``a x b``.
    """
    return np.cross(a, b)

def get_ur5_DH_params():
    """Return homework-specific UR5 classic DH parameters.

    Returns
    -------
    list[dict]
        Six DH dictionaries containing ``a``, ``d``, and ``alpha``.

    Notes
    -----
    The parameters are tailored to this assignment setup and may differ from
    official UR documentation.
    """

    # TODO: this is the DH parameters (following classic DH convention) of the robot in this assignment,
    # It will be a little bit different from the official spec 
    # You need to use these parameters to compute the forward kinematics and Jacobian matrix
    # details : 
    # see "pybullet_robot_envs/envs/ur5_envs/robot_data/ur5/ur5.urdf" in this project folder
    # official spec : https://www.universal-robots.com/articles/ur/application-installation/dh-parameters-for-calculations-of-kinematics-and-dynamics/
    
    dh_params = [
        {'a':  0,      'd': 0.0892,  'alpha':  np.pi/2,  },    # joint1
        {'a':  -0.425, 'd': 0,       'alpha':  0         },    # joint2
        {'a':  -0.392, 'd': 0,       'alpha':  0         },    # joint3
        {'a':  0.,     'd': 0.1093,  'alpha':  np.pi/2   },    # joint4
        {'a':  0.,     'd': 0.09475, 'alpha': -np.pi/2   },    # joint5
        {'a':  0,      'd': 0.2023,  'alpha': 0          },    # joint6
    ]

    return dh_params

def your_fk(DH_params : dict, q, base_pos) -> np.ndarray:
    """Compute FK pose and geometric Jacobian from DH parameters.

    Parameters
    ----------
    DH_params : dict
        Per-joint DH parameter dictionaries for 6 joints.
    q : list | tuple | np.ndarray
        Joint angles (radians), length 6.
    base_pos : list | tuple | np.ndarray
        Robot base translation ``[x, y, z]`` in world frame.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        - pose_7d: ``[x, y, z, qx, qy, qz, qw]``
        - jacobian: 6x6 geometric Jacobian in base frame

    Homework Hints
    --------------
    What to implement:
    1. Multiply classic DH transforms along the kinematic chain.
    2. Collect joint axis/origin in base frame.
    3. Build Jacobian columns with:
       ``Jv_i = z_i x (p_end - p_i)`` and ``Jw_i = z_i``.

    Example
    -------
    ```python
    dh = get_ur5_DH_params()
    q = np.zeros(6)
    pose, jac = your_fk(dh, q, np.array([-0.2, 0.13, 0.6]))
    ```

    Notes
    -----
    This implementation intentionally avoids simulator APIs so students can
    focus on pure kinematics math.
    """
    # robot initial position
    base_pose = list(base_pos) + [0, 0, 0]  

    assert len(DH_params) == 6 and len(q) == 6, f'Both DH_params and q should contain 6 values,\n' \
                                                f'but get len(DH_params) = {DH_params}, len(q) = {len(q)}'

    A = get_matrix_from_pose(base_pose) # a 4x4 matrix, type should be np.ndarray
    jacobian = np.zeros((6, 6)) # a 6x6 matrix, type should be np.ndarray

    # -------------------------------------------------------------------------------- #
    # --- TODO: Read the task description                                          --- #
    # --- Task 1 : Compute Forward-Kinematic and Jacobain of the robot by yourself --- #
    # ---          Try to implement `your_fk` function without using any pybullet  --- #
    # ---          API. (20% for accuracy)                                         --- #
    # -------------------------------------------------------------------------------- #
    
    #### your code ####
    

    # A = ? # may be more than one line
    # jacobian = ? # may be more than one line

    raise NotImplementedError
    # hint : 
    # https://automaticaddison.com/the-ultimate-guide-to-jacobian-matrices-for-robotics/
    
    ###############################################

    # adjustment don't touch
    adjustment = np.asarray([[ 0, -1,  0],
                             [ 0,  0,  0],
                             [ 0,  0, -1]])
    A[:3, :3] = A[:3,:3]@adjustment
    pose_7d = np.asarray(get_pose_from_matrix(A,7))

    return pose_7d, jacobian


def score_fk(student_fk_function, headless=False, visualize_pose=False):
    """Run official FK/Jacobian scoring for a student function.

    Parameters
    ----------
    student_fk_function : Callable
        Student FK function with signature compatible with
        ``student_fk_function(DH_params, q, base_pos)`` returning
        ``(pose_7d, jacobian_6x6)``.
    headless : bool, default=False
        Whether Isaac Sim runs without GUI.
    visualize_pose : bool, default=False
        Whether to draw predicted/ground-truth axes and trajectories.

    Returns
    -------
    dict
        Score summary including per-file and total scores.

    Notes
    -----
    The simulator and debug-draw behavior is intentionally kept the same as
    the original main-loop logic.
    """
    try:
        from isaacsim import SimulationApp
    except ImportError as exc:
        raise ImportError("Isaac Sim python modules are not available in current environment.") from exc

    sim_app = SimulationApp({"headless": bool(headless), "width": 1280, "height": 720})

    try:
        from isaacsim.core.api import World

        world = World(stage_units_in_meters=1.0)
        world.scene.add_default_ground_plane()
        world.reset()  # Ensure physics and scene are initialized.

        debug_draw = _acquire_isaac_debug_draw_interface()
        if debug_draw is not None:
            debug_draw.clear_lines()
            debug_draw.clear_points()
        elif visualize_pose:
            print("[Warning] isaacsim.util.debug_draw is unavailable; skipping pose visualization.")

        # Warm-up stepping to stabilize world state.
        for _ in range(10):
            world.step(render=not headless)

        testcase_files = [
            'test_case/fk_test_case_easy.json',
            'test_case/fk_test_case_medium.json',
            'test_case/fk_test_case_hard.json',
        ]

        dh_params = get_ur5_DH_params()
        base_pos = np.asarray([-0.2, 0.13, 0.6], dtype=np.float64)

        testcase_file_num = len(testcase_files)
        fk_score = [FK_SCORE_MAX / testcase_file_num for _ in range(testcase_file_num)]
        fk_error_cnt = [0 for _ in range(testcase_file_num)]
        jacobian_score = [JACOBIAN_SCORE_MAX / testcase_file_num for _ in range(testcase_file_num)]
        jacobian_error_cnt = [0 for _ in range(testcase_file_num)]

        print("============================ Task 1 : Forward Kinematic ============================\n")
        for file_id, testcase_file in enumerate(testcase_files):

            with open(testcase_file, 'r') as f_in:
                fk_dict = json.load(f_in)

            test_case_name = os.path.split(testcase_file)[-1]

            joint_poses = fk_dict['joint_poses']
            poses = fk_dict['poses']
            jacobians = fk_dict['jacobian']

            cases_num = len(fk_dict['joint_poses'])
            penalty = (TASK1_SCORE_MAX / testcase_file_num) / (0.3 * cases_num)

            pred_traj = []
            gt_traj = []

            for i in range(cases_num):
                your_pose, your_jacobian = student_fk_function(dh_params, joint_poses[i], base_pos)
                gt_pose = poses[i]

                if visualize_pose and debug_draw is not None:
                    pred_traj.append(tuple(np.asarray(your_pose[:3], dtype=np.float64)))
                    gt_traj.append(tuple(np.asarray(gt_pose[:3], dtype=np.float64)))

                    _draw_pose_axes_isaac(
                        debug_draw,
                        your_pose,
                        axis_len=0.02,
                        width=2.0,
                        color_x=(1.0, 0.0, 0.0, 1.0),
                        color_y=(0.0, 1.0, 0.0, 1.0),
                        color_z=(0.0, 0.0, 1.0, 1.0),
                    )
                    _draw_pose_axes_isaac(
                        debug_draw,
                        gt_pose,
                        axis_len=0.02,
                        width=1.0,
                        color_x=(1.0, 0.5, 0.5, 1.0),
                        color_y=(0.5, 1.0, 0.5, 1.0),
                        color_z=(0.5, 0.5, 1.0, 1.0),
                    )

                    if len(pred_traj) >= 2:
                        debug_draw.draw_lines([pred_traj[-2]], [pred_traj[-1]], [(1.0, 1.0, 0.0, 1.0)], [2.0])
                    if len(gt_traj) >= 2:
                        debug_draw.draw_lines([gt_traj[-2]], [gt_traj[-1]], [(0.0, 1.0, 1.0, 1.0)], [1.5])

                fk_error = np.linalg.norm(your_pose - np.asarray(gt_pose), ord=2)
                if fk_error > FK_ERROR_THRESH:
                    fk_score[file_id] -= penalty
                    fk_error_cnt[file_id] += 1

                jacobian_error = np.linalg.norm(your_jacobian - np.asarray(jacobians[i]), ord=2)
                if jacobian_error > JACOBIAN_ERROR_THRESH:
                    jacobian_score[file_id] -= penalty
                    jacobian_error_cnt[file_id] += 1

                world.step(render=not headless)

            fk_score[file_id] = 0.0 if fk_score[file_id] < 0.0 else fk_score[file_id]
            jacobian_score[file_id] = 0.0 if jacobian_score[file_id] < 0.0 else jacobian_score[file_id]

            score_msg = "- Testcase file : {}\n".format(test_case_name) + \
                        "- Your Score Of Forward Kinematic : {:00.03f} / {:00.03f}, Error Count : {:4d} / {:4d}\n".format(
                                fk_score[file_id], FK_SCORE_MAX / testcase_file_num, fk_error_cnt[file_id], cases_num) + \
                        "- Your Score Of Jacobian Matrix   : {:00.03f} / {:00.03f}, Error Count : {:4d} / {:4d}\n".format(
                                jacobian_score[file_id], JACOBIAN_SCORE_MAX / testcase_file_num, jacobian_error_cnt[file_id], cases_num)

            print(score_msg)

        total_fk_score = 0.0
        total_jacobian_score = 0.0
        for file_id in range(testcase_file_num):
            total_fk_score += fk_score[file_id]
            total_jacobian_score += jacobian_score[file_id]

        print("====================================================================================")
        print("- Your Total Score : {:00.03f} / {:00.03f}".format(
            total_fk_score + total_jacobian_score, FK_SCORE_MAX + JACOBIAN_SCORE_MAX))
        print("====================================================================================")

        return {
            "fk_score": fk_score,
            "fk_error_count": fk_error_cnt,
            "jacobian_score": jacobian_score,
            "jacobian_error_count": jacobian_error_cnt,
            "total_score": total_fk_score + total_jacobian_score,
        }

    except Exception:
        print("\n[Error] following are the error messages:")
        traceback.print_exc()
        print("--------------------------------------------------\n")
        raise

    finally:
        sim_app.close()

def main(args):
    """CLI entry point for FK homework evaluation.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line options. ``headless`` and ``visualize_pose`` are forwarded
        to the scorer.

    Returns
    -------
    dict
        Score summary from ``score_fk``.
    """
    return score_fk(
        your_fk,
        headless=bool(args.headless),
        visualize_pose=bool(args.visualize_pose),
    )


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--headless', action='store_true', default=False,
                        help='run Isaac Sim without rendering window')
    parser.add_argument('--gui', '-g', action='store_true', default=False, help='gui : whether show the window')
    parser.add_argument('--visualize-pose', '-vp', action='store_true', default=True, help='whether show the poses of end effector')
    args = parser.parse_args()
    main(args)
