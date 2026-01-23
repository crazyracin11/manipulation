"""
MuJoCo GPU 环境配置模块
专为 JAX/Jupyter 环境设计，避免 FileNotFound 错误
"""

import os
import subprocess
import sys
from pathlib import Path

def setup_mujoco_gpu_environment():
    """
    配置 MuJoCo GPU 渲染环境，具有错误处理和多种备选方案
    """
    print("🔧 配置 MuJoCo GPU 渲染环境...")

    # 检查GPU可用性
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ NVIDIA GPU 检测成功")
            print(result.stdout.split('\n')[7])  # GPU信息行
        else:
            print("⚠️  NVIDIA GPU 未检测到")
            return False
    except FileNotFoundError:
        print("❌ nvidia-smi 命令未找到，请安装NVIDIA驱动")
        return False

    # 设置MuJoCo渲染后端
    print("🎨 设置MuJoCo渲染后端为EGL...")
    os.environ['MUJOCO_GL'] = 'egl'

    # 设置JAX内存分配器
    print("🧠 设置JAX内存分配器为platform...")
    os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'

    # 配置XLA flags以优化GPU性能
    xla_flags = os.environ.get('XLA_FLAGS', '')
    if '--xla_gpu_triton_gemm_any=True' not in xla_flags:
        xla_flags += ' --xla_gpu_triton_gemm_any=True'
    os.environ['XLA_FLAGS'] = xla_flags

    # 处理NVIDIA ICD配置 - 安全创建
    success = setup_nvidia_icd_config()

    if success:
        print("✅ MuJoCo GPU 环境配置完成")
        return True
    else:
        print("⚠️  ICD配置失败，但GPU渲染仍可能工作")
        return True  # 即使ICD配置失败，也继续尝试

def setup_nvidia_icd_config():
    """
    安全地创建NVIDIA ICD配置文件
    具有完整的错误处理和目录创建
    """
    nvidia_icd_paths = [
        '/usr/share/glvnd/egl_vendor.d/10_nvidia.json',
        '/etc/glvnd/egl_vendor.d/10_nvidia.json',
        '/usr/local/share/glvnd/egl_vendor.d/10_nvidia.json'
    ]

    nvidia_config = """{
    "file_format_version" : "1.0.0",
    "ICD" : {
        "library_path" : "libEGL_nvidia.so.0"
    }
}"""

    for config_path in nvidia_icd_paths:
        try:
            # 确保目录存在
            config_dir = Path(config_path).parent
            config_dir.mkdir(parents=True, exist_ok=True)

            # 检查是否已有配置
            if Path(config_path).exists():
                print(f"✅ NVIDIA ICD配置已存在: {config_path}")
                return True

            # 尝试写入配置文件
            with open(config_path, 'w') as f:
                f.write(nvidia_config)

            print(f"✅ 成功创建NVIDIA ICD配置: {config_path}")
            return True

        except PermissionError:
            print(f"⚠️  权限不足，无法写入: {config_path}")
            continue
        except Exception as e:
            print(f"⚠️  创建配置失败 {config_path}: {e}")
            continue

    # 如果所有路径都失败，尝试创建临时配置
    try:
        temp_config = Path.home() / '.config' / 'glvnd' / 'egl_vendor.d' / '10_nvidia.json'
        temp_config.parent.mkdir(parents=True, exist_ok=True)

        with open(temp_config, 'w') as f:
            f.write(nvidia_config)

        # 设置环境变量指向临时配置
        os.environ['__EGL_VENDOR_LIBRARY_FILENAMES'] = str(temp_config)
        print(f"✅ 创建临时NVIDIA ICD配置: {temp_config}")
        return True

    except Exception as e:
        print(f"❌ 无法创建任何ICD配置: {e}")
        return False

def test_mujoco_installation():
    """
    测试MuJoCo安装是否成功
    """
    try:
        import mujoco
        # 测试基本MuJoCo功能
        model = mujoco.MjModel.from_xml_string('<mujoco/>')
        print("✅ MuJoCo 测试成功")
        return True
    except Exception as e:
        print(f"❌ MuJoCo 测试失败: {e}")
        return False

def setup_jax_environment():
    """
    配置JAX环境
    """
    try:
        import jax
        print(f"✅ JAX版本: {jax.__version__}")

        # 配置JAX使用CUDA
        jax.config.update('jax_platform_name', 'cuda')

        # 显示设备信息
        devices = jax.devices()
        print(f"🎯 JAX设备: {[str(d) for d in devices]}")

        if len(devices) > 0 and 'gpu' in str(devices[0]).lower():
            print("✅ JAX GPU 配置成功")
            return True
        else:
            print("⚠️  JAX 使用CPU，建议安装CUDA版本的jaxlib")
            return False

    except ImportError:
        print("❌ JAX 未安装")
        return False
    except Exception as e:
        print(f"❌ JAX 配置失败: {e}")
        return False

# 主函数
def configure_environment():
    """
    完整的环境配置流程
    """
    print("🚀 开始配置RL训练环境...")
    print("=" * 50)

    success_steps = 0
    total_steps = 4

    # 步骤1: 配置MuJoCo GPU
    if setup_mujoco_gpu_environment():
        success_steps += 1

    # 步骤2: 测试MuJoCo
    if test_mujoco_installation():
        success_steps += 1

    # 步骤3: 配置JAX
    if setup_jax_environment():
        success_steps += 1

    # 步骤4: 环境变量检查
    print(f"\n🌍 环境变量配置:")
    print(f"  MUJOCO_GL: {os.environ.get('MUJOCO_GL', 'Not set')}")
    print(f"  XLA_PYTHON_CLIENT_ALLOCATOR: {os.environ.get('XLA_PYTHON_CLIENT_ALLOCATOR', 'Not set')}")
    success_steps += 1

    print("\n" + "=" * 50)
    print(f"配置完成: {success_steps}/{total_steps} 步骤成功")

    if success_steps >= 3:
        print("🎉 环境配置基本成功，可以开始训练!")
        return True
    else:
        print("⚠️  环境配置存在问题，可能影响训练效果")
        return False

if __name__ == "__main__":
    configure_environment()