#!/bin/bash
# NVIDIA EGL 环境配置修复脚本
# 专为 MuJoCo/JAX GPU 渲染环境设计

echo "🔧 开始修复 NVIDIA EGL 配置..."
echo "================================================"

# 检查是否有sudo权限
if ! sudo -n true 2>/dev/null; then
    echo "⚠️  需要sudo权限来安装系统包"
    echo "请运行: sudo bash fix_nvidia_egl.sh"
    exit 1
fi

# 1. 更新包列表
echo "📦 更新包列表..."
sudo apt update

# 2. 安装必要的OpenGL和EGL库
echo "🎯 安装OpenGL和EGL库..."
sudo apt install -y \
    libgl1-mesa-glx \
    libgl1-mesa-dev \
    libegl1-mesa \
    libegl1-mesa-dev \
    libglvnd0 \
    libglvnd-dev \
    libgles2-mesa \
    libgles2-mesa-dev

# 3. 安装NVIDIA驱动支持（如果没有完整驱动）
echo "🚀 安装NVIDIA驱动支持..."
sudo apt install -y \
    nvidia-driver-535 \
    nvidia-cuda-toolkit \
    libnvidia-eglcore

# 4. 创建EGL vendor目录（如果不存在）
echo "📁 创建EGL vendor目录..."
sudo mkdir -p /usr/share/glvnd/egl_vendor.d/
sudo mkdir -p /usr/share/vulkan/icd.d/

# 5. 创建NVIDIA ICD配置文件
echo "⚙️  创建NVIDIA ICD配置文件..."
sudo tee /usr/share/glvnd/egl_vendor.d/10_nvidia.json > /dev/null << 'EOF'
{
    "file_format_version" : "1.0.0",
    "ICD" : {
        "library_path" : "libEGL_nvidia.so.0"
    }
}
EOF

# 6. 创建Vulkan ICD配置（用于更好的GPU支持）
sudo tee /usr/share/vulkan/icd.d/nvidia_icd.json > /dev/null << 'EOF'
{
    "file_format_version": "1.0.0",
    "ICD": {
        "library_path": "libvulkan.so.1",
        "api_version": "1.3.204"
    }
}
EOF

# 7. 设置正确的权限
echo "🔐 设置文件权限..."
sudo chmod 644 /usr/share/glvnd/egl_vendor.d/10_nvidia.json
sudo chmod 644 /usr/share/vulkan/icd.d/nvidia_icd.json

# 8. 更新动态链接库缓存
echo "🔄 更新动态链接库缓存..."
sudo ldconfig

# 9. 验证安装
echo "✅ 验证安装..."
echo "检查EGL库："
find /usr -name "libEGL*" 2>/dev/null | head -5

echo ""
echo "检查NVIDIA库："
find /usr -name "*nvidia*" -name "*.so*" 2>/dev/null | head -5

echo ""
echo "检查ICD配置文件："
ls -la /usr/share/glvnd/egl_vendor.d/

# 10. 设置环境变量
echo ""
echo "🌍 建议设置以下环境变量："
echo "export MUJOCO_GL=egl"
echo "export XLA_PYTHON_CLIENT_ALLOCATOR=platform"
echo ""

# 11. 测试GPU
echo "🧪 测试GPU连接..."
nvidia-smi

echo ""
echo "✅ NVIDIA EGL 配置修复完成！"
echo "现在可以重新运行Jupyter notebook了。"
echo "================================================"