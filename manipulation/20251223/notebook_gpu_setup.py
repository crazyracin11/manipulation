"""
Jupyter Notebook GPU 环境配置代码
替代原有有问题的代码片段，避免 FileNotFoundError
"""

# 在Jupyter notebook中使用以下代码替代原来的配置代码

def setup_notebook_gpu_environment():
    """
    专门为Jupyter notebook设计的安全GPU环境配置
    """
    import os
    import subprocess
    from pathlib import Path

    print("🔧 配置Jupyter GPU环境...")

    # 1. 检查GPU
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ GPU检测成功")
        else:
            print("⚠️  GPU未检测到")
    except:
        print("❌ 无法检查GPU状态")

    # 2. 设置关键环境变量
    os.environ['MUJOCO_GL'] = 'egl'
    os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'  # 内存优化

    # 3. 配置XLA flags
    xla_flags = os.environ.get('XLA_FLAGS', '')
    if '--xla_gpu_triton_gemm_any=True' not in xla_flags:
        xla_flags += ' --xla_gpu_triton_gemm_any=True'
    os.environ['XLA_FLAGS'] = xla_flags

    # 4. 安全创建NVIDIA ICD配置（避免原代码的错误）
    create_safe_icd_config()

    print("✅ Jupyter GPU环境配置完成")

def create_safe_icd_config():
    """
    安全创建ICD配置，避免FileNotFoundError
    """
    import os
    from pathlib import Path

    # 尝试多个可能的路径
    icd_paths = [
        '/usr/share/glvnd/egl_vendor.d/10_nvidia.json',
        '/etc/glvnd/egl_vendor.d/10_nvidia.json',
        str(Path.home() / '.config' / 'glvnd' / 'egl_vendor.d' / '10_nvidia.json')
    ]

    config_content = """{
    "file_format_version" : "1.0.0",
    "ICD" : {
        "library_path" : "libEGL_nvidia.so.0"
    }
}"""

    for config_path in icd_paths:
        try:
            config_dir = Path(config_path).parent

            # 检查目录是否可写
            if config_dir.exists() and os.access(config_dir, os.W_OK):
                with open(config_path, 'w') as f:
                    f.write(config_content)
                print(f"✅ 创建ICD配置: {config_path}")
                return

        except (PermissionError, FileNotFoundError) as e:
            continue
        except Exception as e:
            continue

    # 如果无法创建系统级配置，创建用户级配置
    try:
        user_config = Path.home() / '.config' / 'glvnd' / 'egl_vendor.d' / '10_nvidia.json'
        user_config.parent.mkdir(parents=True, exist_ok=True)

        with open(user_config, 'w') as f:
            f.write(config_content)

        # 设置环境变量
        os.environ['__EGL_VENDOR_LIBRARY_FILENAMES'] = str(user_config)
        print(f"✅ 创建用户ICD配置: {user_config}")

    except Exception as e:
        print(f"⚠️  无法创建ICD配置: {e}")
        print("继续使用默认EGL配置...")

# 使用示例：
# 在notebook开头运行:
# setup_notebook_gpu_environment()

# 然后导入相关库:
# import mujoco
# import jax
# 等等...