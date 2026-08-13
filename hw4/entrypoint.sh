#!/bin/bash
set -e

# ———————————— Activate hw4 venv ————————————
if [ -d "/workspace/hw4/.venv" ]; then
  echo "[entrypoint] Activating hw4 venv"
  source /workspace/hw4/.venv/bin/activate
else
  echo "[entrypoint][WARN] .venv not found, using system python"
fi

# ———————————— Set default Isaac Sim args ————————————
# 忽略 Vulkan 驅動版本檢查
export OMNI_KIT_ARGS="--/rtx/verifyDriverVersion/enabled=false"

# Omniverse 也會讀這個（關鍵）
export OMNI_KIT_DISABLE_DRIVER_VERSION_CHECK=1
# ———————————— Execute command ————————————
exec "$@"
