import argparse, os, json
import numpy as np
import traceback
from scipy.spatial.transform import Rotation as R
from scipy.linalg import pinv

# you may use your forward kinematic algorithm to compute 
from fk import your_fk, get_ur5_DH_params

SIM_TIMESTEP = 1.0 / 240.0
TASK2_SCORE_MAX = 40
IK_ERROR_THRESH = 0.02

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



def _get_initial_q(q_init=None):
    """Validate and normalize an initial 6-DoF joint vector.

    Parameters
    ----------
    q_init : list | tuple | np.ndarray | None
        Initial joint values. Must contain at least 6 numbers.

    Returns
    -------
    np.ndarray
        Joint vector of shape ``(6,)`` in float64.

    Raises
    ------
    ValueError
        If ``q_init`` is missing or has fewer than 6 elements.
    """
    if q_init is not None:
        q_init = np.asarray(q_init, dtype=np.float64).reshape(-1)
        if q_init.size < 6:
            raise ValueError(f"q_init should have at least 6 values, got {q_init.size}")
        return q_init[:6].copy()
    raise ValueError(
        "Cannot infer initial joints. Provide q_init or pass a sequence/articulation object."
    )


def your_ik(new_pose : list or tuple or np.ndarray, 
                base_pos, max_iters : int=1000, stop_thresh : float=.001, q_init=None):
    """Solve inverse kinematics using iterative Jacobian pseudo-inverse updates.

    Parameters
    ----------
    new_pose : list | tuple | np.ndarray
        Target end-effector pose in 7D format ``[x, y, z, qx, qy, qz, qw]``.
    base_pos : list | tuple | np.ndarray
        Robot base translation in world frame.
    max_iters : int, default=1000
        Maximum optimization iterations.
    stop_thresh : float, default=0.001
        Stopping threshold on the 6D pose error norm.
    q_init : list | tuple | np.ndarray | None
        Initial joint guess (length >= 6).

    Returns
    -------
    list
        Estimated 6 joint values in radians.

    Homework Hints
    --------------
    Input:
    - Target pose ``new_pose`` and a valid initial guess ``q_init``.
    Output:
    - Joint angles that minimize pose error.

    Suggested implementation logic:
    1. Evaluate current pose and Jacobian via ``your_fk``.
    2. Build 6D error ``[position_error, orientation_error]``.
    3. Compute ``delta_q = pinv(J) @ error``.
    4. Apply step size and clip by joint limits.
    5. Stop when error norm is below threshold.

    Example
    -------
    ```python
    target = [0.4, 0.0, 0.8, 0.0, 0.7071, 0.0, 0.7071]
    q_sol = your_ik(target, base_pos=[-0.2, 0.13, 0.6], q_init=np.zeros(6))
    ```

    Notes
    -----
    Orientation error is computed from relative rotation
    ``R_target @ R_current.T`` converted to axis-angle form.
    """



    joint_limits = np.asarray([
            [-3*np.pi/2, -np.pi/2], # joint1
            [-2.3562, -1],           # joint2
            [-17, 17],              # joint3
            [-17, 17],              # joint4
            [-17, 17],              # joint5
            [-17, 17],              # joint6
        ])

    tmp_q = _get_initial_q(q_init=q_init)
    base_pos = np.asarray(base_pos if base_pos is not None else [0.0, 0.0, 0.0], dtype=np.float64)
        
    # -------------------------------------------------------------------------------- #
    # --- TODO: Read the task description                                          --- #
    # --- Task 2 : Compute Inverse-Kinematic Solver of the robot by yourself.      --- #
    # ---          Try to implement `your_ik` without simulator IK APIs           --- #
    # ---          API. (40% for accuracy)                                         --- #
    # --- Note : please modify the code in `your_ik` function.                     --- #
    # -------------------------------------------------------------------------------- #
    
    #### your code ####

    # TODO: update tmp_q using an iterative optimization loop.
    # tmp_q = ? # may be more than one line
    
    # hint : 
    # 1. You may use `your_fk` function and jacobian matrix to do this
    # 2. Be careful when computing the delta x
    # 3. You may use some hyper parameters (i.e., step rate) in optimization loops

    ###################
    

    return list(tmp_q) # 6 DoF


def score_ik(student_ik_function, headless=False):
    """Run official IK scoring for a student IK function.

    Parameters
    ----------
    student_ik_function : Callable
        Student IK function compatible with
        ``student_ik_function(new_pose, base_pos, q_init=...)``.
    headless : bool, default=False
        Whether Isaac Sim runs without GUI.

    Returns
    -------
    dict
        Score summary including per-file scores and total score.

    Notes
    -----
    The simulator setup and articulation control flow are preserved from the
    original main-loop implementation.
    """
    try:
        from isaacsim import SimulationApp
    except ImportError as exc:
        raise ImportError("Isaac Sim python modules are not available in current environment.") from exc

    sim_app = SimulationApp({"headless": bool(headless), "width": 1280, "height": 720})

    try:

        from isaacsim.core.utils.stage import add_reference_to_stage
        from isaacsim.storage.native import get_assets_root_path
        from isaacsim.core.api.controllers.articulation_controller import ArticulationController
        from isaacsim.core.prims import Articulation
        from isaacsim.core.utils.types import ArticulationAction
        from isaacsim.core.api.world import World
        world = World(stage_units_in_meters=1.0)
        world.scene.add_default_ground_plane()


        # Load UR5 into Isaac world using the requested API set.
        assets_root = get_assets_root_path()
        if assets_root is None:
            raise RuntimeError("Isaac assets root path is None")
        usd_path = assets_root + "/Isaac/Robots/UniversalRobots/ur5/ur5.usd"
        prim_path = "/World/envs/env_0/ur5"

        add_reference_to_stage(usd_path, prim_path)

        robot_view = Articulation(prim_paths_expr=prim_path, name="ur5_view")
        articulation_controller = ArticulationController()

        # Reset after stage edits, then initialize articulation controller.
        world.reset()
        articulation_controller.initialize(robot_view)


        # Match Isaac initial pose to the reference initial joint states.
        reference_init_states = np.asarray([
            -3.141592642791131,
            -1.5707963240621052,
            1.5707963521600738,
            -1.5707963267948966,
            -1.5707963267948966,
            1.06243199169874e-08,
        ], dtype=np.float64)

        current_positions = np.asarray(robot_view.get_joint_positions(), dtype=np.float64).reshape(-1)
        target_positions = current_positions.copy()
        n_apply = min(target_positions.size, reference_init_states.size)
        target_positions[:n_apply] = reference_init_states[:n_apply]

        # Drive articulation to target initial joint state.
        articulation_controller.apply_action(ArticulationAction(joint_positions=target_positions))
        for _ in range(40):
            world.step(render=not headless)

        current_positions = np.asarray(robot_view.get_joint_positions(), dtype=np.float64).reshape(-1)
        for _ in range(10):
            world.step(render=not headless)

        testcase_files = [
            'test_case/ik_test_case_easy.json',
            'test_case/ik_test_case_medium.json',
            'test_case/ik_test_case_hard.json',
        ]

        dh_params = get_ur5_DH_params()
        # Keep base frame consistent with the verified Isaac FK test configuration.
        base_pos = np.asarray([-0.2, 0.13, 0.6], dtype=np.float64)
        current_positions = np.asarray(robot_view.get_joint_positions(), dtype=np.float64).reshape(-1)
        if current_positions.size < 6:
            raise RuntimeError(f"UR5 articulation has invalid dof size: {current_positions.size}")
        q_curr = current_positions[:6].copy()
        testcase_file_num = len(testcase_files)
        ik_score = [TASK2_SCORE_MAX / testcase_file_num for _ in range(testcase_file_num)]
        ik_error_cnt = [0 for _ in range(testcase_file_num)]

        print("============================ Task 2 : Inverse Kinematic ============================\n")

        for file_id, testcase_file in enumerate(testcase_files):
            try:
                with open(testcase_file, 'r') as f_in:
                    ik_dict = json.load(f_in)
            except Exception:
                traceback.print_exc()
                continue

            test_case_name = os.path.split(testcase_file)[-1]
            poses = ik_dict['next_poses']
            cases_num = len(poses)
            penalty = (TASK2_SCORE_MAX / testcase_file_num) / (0.3 * cases_num)

            ik_errors = []
            for case_id, target_pose in enumerate(poses):
                try:
                    q_sol = student_ik_function(
                        new_pose=target_pose,
                        base_pos=base_pos,
                        q_init=q_curr,
                    )
                    q_curr = np.asarray(q_sol, dtype=np.float64)

                    # Apply IK solution to UR5 articulation in Isaac Sim.
                    current_positions = np.asarray(robot_view.get_joint_positions(), dtype=np.float64).reshape(-1)
                    target_positions = current_positions.copy()
                    target_positions[:6] = q_curr
                    action = ArticulationAction(joint_positions=target_positions)
                    articulation_controller.apply_action(action)

                    # Let articulation move before evaluating end-effector pose.
                    for _ in range(int(1 / SIM_TIMESTEP * 0.1)):
                        world.step(render=not headless)

                    solved_pose, _ = your_fk(dh_params, q_curr, base_pos)
                    ik_error = np.linalg.norm(np.asarray(solved_pose) - np.asarray(target_pose), ord=2)
                    ik_errors.append(ik_error)
                    if ik_error > IK_ERROR_THRESH:
                        ik_score[file_id] -= penalty
                        ik_error_cnt[file_id] += 1
                except Exception:
                    traceback.print_exc()
                    world.step(render=not headless)
                    continue


            ik_score[file_id] = 0.0 if ik_score[file_id] < 0.0 else ik_score[file_id]
            ik_errors = np.asarray(ik_errors)
            mean_file_err = float(np.mean(ik_errors)) if ik_errors.size > 0 else float('nan')

            score_msg = "- Testcase file : {}\n".format(test_case_name) + \
                        "- Mean Error : {:0.06f}\n".format(mean_file_err) + \
                        "- Error Count : {:3d} / {:3d}\n".format(ik_error_cnt[file_id], cases_num) + \
                        "- Your Score Of Inverse Kinematic : {:00.03f} / {:00.03f}\n".format(
                                ik_score[file_id], TASK2_SCORE_MAX / testcase_file_num)
            print(score_msg)

        total_ik_score = 0.0
        for file_id in range(testcase_file_num):
            total_ik_score += ik_score[file_id]

        print("====================================================================================")
        print("- Your Total Score : {:00.03f} / {:00.03f}".format(total_ik_score , TASK2_SCORE_MAX))
        print("====================================================================================")

        return {
            "ik_score": ik_score,
            "ik_error_count": ik_error_cnt,
            "total_score": total_ik_score,
        }
    except Exception:
        traceback.print_exc()
        raise
    finally:

        sim_app.close()


def main(args):
    """CLI entry point for IK homework evaluation.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line options. ``headless`` is forwarded to the scorer.

    Returns
    -------
    dict
        Score summary from ``score_ik``.
    """
    return score_ik(your_ik, headless=bool(args.headless))
    


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--headless', action='store_true', default=False,
                        help='run Isaac Sim without rendering window')
    args = parser.parse_args()
    main(args)