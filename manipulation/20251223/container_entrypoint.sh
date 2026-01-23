#!/bin/bash
# 容器启动时自动执行的脚本
# 确保GPU环境配置在新Pod中生效

echo "🚀 容器启动 - 自动配置GPU环境..."

# 激活conda环境
source /root/HW/RL_qizhi-main/PPO-Humanoid/ENTER/etc/profile.d/conda.sh
conda activate BHW

# 设置关键环境变量
export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

# XLA优化
XLA_FLAGS=${XLA_FLAGS:-""}
if [[ "$XLA_FLAGS" != *"--xla_gpu_triton_gemm_any=True"* ]]; then
    export XLA_FLAGS="$XLA_FLAGS --xla_gpu_triton_gemm_any=True"
fi

# 设置EGL配置路径
export EGL_VENDOR_LIBRARY_FILENAMES="/root/HW/RL_qizhi-main/PPO-Humanoid/ENTER/envs/BHW/etc/glvnd/egl_vendor.d/10_nvidia.json"

echo "✅ GPU环境变量设置完成"
echo "  MUJOCO_GL: $MUJOCO_GL"
echo "  XLA_PYTHON_CLIENT_ALLOCATOR: $XLA_PYTHON_CLIENT_ALLOCATOR"
echo "  EGL_VENDOR_LIBRARY_FILENAMES: $EGL_VENDOR_LIBRARY_FILENAMES"

# 如果是Jupyter启动，显示额外信息
if [[ "$1" == *"jupyter"* ]] || [[ "$1" == *"notebook"* ]]; then
    echo "📓 Jupyter环境已配置"
    echo "GPU配置将在notebook中自动加载"
fi

# 执行原始命令
exec "$@"