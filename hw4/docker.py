import os
import subprocess
import sys
import argparse

SERVICE_NAME = "isaac-sim-hw4"
VENV_PYTHON = ".venv/bin/python"


def log(msg: str):
    """Print a CLI log line."""
    print(msg)


def _build_env(width: int = 1280, height: int = 720):
    """Build shared environment variables for docker compose execution."""
    env_vars = os.environ.copy()
    env_vars.update({
        "OMNI_KIT_ACCEPT_EULA": "Y",
        "PRIVACY_CONSENT": "Y",
        "DISPLAY": os.getenv("DISPLAY", ":1"),
        "NVIDIA_VISIBLE_DEVICES": "all",
        "NVIDIA_DRIVER_CAPABILITIES": "all,graphics,display,utility,compute",
        "ROS_LOCALHOST_ONLY": "0",
        "ROS_DOMAIN_ID": "0",
        "ROS_DISTRO": "humble",
        "RMW_IMPLEMENTATION": "rmw_fastrtps_cpp",
        "WINDOW_WIDTH": str(width),
        "WINDOW_HEIGHT": str(height),
    })
    return env_vars


def _run_in_isaac_container(container_command: str, env_vars: dict):
    """Build image and execute a command in the Isaac Sim service container."""
    log(f"[CLI] Building Docker image ({SERVICE_NAME})...")
    build_cmd = ["docker", "compose", "build", SERVICE_NAME]
    subprocess.run(build_cmd, env=env_vars, check=True)

    log(f"[CLI] Running in container: {container_command}")
    run_cmd = [
        "docker", "compose", "run", "--rm",
        SERVICE_NAME,
        "/bin/bash", "-lc",
        container_command,
    ]
    subprocess.run(run_cmd, env=env_vars, check=True)

def launch_simulator(task, session_dir, episode, width, height):
    """Launch Isaac Sim with ROS2 bridge enabled"""
    try:
        env_vars = _build_env(width=width, height=height)
        env_vars["TASK_NAME"] = task

        log(f"[CLI] Launching Isaac Sim + ROS2: task={task}, resolution={width}x{height}")
        container_command = (
            f"{VENV_PYTHON} scripts/generate_data.py "
            f"--task {task} --session_dir {session_dir} --episode {episode}"
        )
        _run_in_isaac_container(container_command, env_vars)

    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Docker execution failed: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] {str(e)}", file=sys.stderr)
        sys.exit(1)


def run_fk(headless, visualize_pose, width, height):
    """Run fk.py inside the Isaac Sim Docker environment."""
    try:
        env_vars = _build_env(width=width, height=height)
        args = []
        if headless:
            args.append("--headless")
        if visualize_pose:
            args.append("--visualize-pose")
        arg_str = " ".join(args)
        container_command = f"{VENV_PYTHON} fk.py {arg_str}".strip()
        _run_in_isaac_container(container_command, env_vars)

    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Docker execution failed: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] {str(e)}", file=sys.stderr)
        sys.exit(1)


def run_ik(headless, width, height):
    """Run ik.py inside the Isaac Sim Docker environment."""
    try:
        env_vars = _build_env(width=width, height=height)
        args = "--headless" if headless else ""
        container_command = f"{VENV_PYTHON} ik.py {args}".strip()
        _run_in_isaac_container(container_command, env_vars)

    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Docker execution failed: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] {str(e)}", file=sys.stderr)
        sys.exit(1)


def build_parser():
    """Create argparse parser for docker utility commands."""
    parser = argparse.ArgumentParser(description="Run hw4 Isaac Sim tasks in Docker")
    subparsers = parser.add_subparsers(dest="command", required=True)

    launch_parser = subparsers.add_parser("launch-simulator", help="Launch Isaac Sim + ROS2 data generation")
    launch_parser.add_argument("--task", choices=["kitchen", "dining-room", "living-room"], required=True)
    launch_parser.add_argument("--session_dir", type=str, required=True)
    launch_parser.add_argument("--episode", type=int, default=0)
    launch_parser.add_argument("--width", type=int, default=1280)
    launch_parser.add_argument("--height", type=int, default=720)

    fk_parser = subparsers.add_parser("run-fk", help="Run fk.py in Isaac Sim container")
    fk_parser.add_argument("--headless", action="store_true")
    fk_parser.add_argument("--visualize-pose", action="store_true")
    fk_parser.add_argument("--width", type=int, default=1280)
    fk_parser.add_argument("--height", type=int, default=720)

    ik_parser = subparsers.add_parser("run-ik", help="Run ik.py in Isaac Sim container")
    ik_parser.add_argument("--headless", action="store_true")
    ik_parser.add_argument("--width", type=int, default=1280)
    ik_parser.add_argument("--height", type=int, default=720)

    return parser


def main():
    """Dispatch subcommands to command handlers."""
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "launch-simulator":
        launch_simulator(
            task=args.task,
            session_dir=args.session_dir,
            episode=args.episode,
            width=args.width,
            height=args.height,
        )
        return

    if args.command == "run-fk":
        run_fk(
            headless=args.headless,
            visualize_pose=args.visualize_pose,
            width=args.width,
            height=args.height,
        )
        return

    if args.command == "run-ik":
        run_ik(
            headless=args.headless,
            width=args.width,
            height=args.height,
        )
        return

if __name__ == "__main__":
    main()