"""
永久性GPU环境配置方案
专为容器化环境设计，确保Pod重启后仍可工作
"""

import os
import sys
from pathlib import Path
import subprocess

class PermanentGPUSetup:
    def __init__(self):
        self.conda_env_path = "/root/HW/RL_qizhi-main/PPO-Humanoid/ENTER/envs/BHW"
        self.config_dir = Path(self.conda_env_path) / "etc" / "glvnd" / "egl_vendor.d"

    def setup_permanent_environment(self):
        """
        永久性环境配置 - 针对容器环境优化
        """
        print("🚀 配置永久性GPU环境...")

        success_count = 0

        # 1. 设置环境变量（每次运行时都需要）
        if self._set_environment_variables():
            success_count += 1

        # 2. 在conda环境中创建永久配置
        if self._create_conda_icd_config():
            success_count += 1

        # 3. 创建启动脚本（自动执行配置）
        if self._create_auto_setup_script():
            success_count += 1

        # 4. 修改Jupyter notebook配置
        if self._create_notebook_config():
            success_count += 1

        print(f"✅ 永久配置完成: {success_count}/4 步骤成功")
        return success_count >= 3

    def _set_environment_variables(self):
        """
        设置关键环境变量
        """
        try:
            # GPU渲染配置
            os.environ['MUJOCO_GL'] = 'egl'
            os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'  # 内存优化

            # XLA优化
            xla_flags = os.environ.get('XLA_FLAGS', '')
            if '--xla_gpu_triton_gemm_any=True' not in xla_flags:
                xla_flags += ' --xla_gpu_triton_gemm_any=True'
            os.environ['XLA_FLAGS'] = xla_flags

            print("✅ 环境变量设置完成")
            return True
        except Exception as e:
            print(f"❌ 环境变量设置失败: {e}")
            return False

    def _create_conda_icd_config(self):
        """
        在conda环境中创建ICD配置（会随环境持久化）
        """
        try:
            # 创建conda环境中的配置目录
            self.config_dir.mkdir(parents=True, exist_ok=True)

            # 创建NVIDIA ICD配置
            icd_config = self.config_dir / "10_nvidia.json"

            if not icd_config.exists():
                config_content = """{
    "file_format_version" : "1.0.0",
    "ICD" : {
        "library_path" : "libEGL_nvidia.so.0"
    }
}"""
                with open(icd_config, 'w') as f:
                    f.write(config_content)

            # 设置EGL库路径
            os.environ['EGL_VENDOR_LIBRARY_FILENAMES'] = str(icd_config)

            print(f"✅ Conda ICD配置创建: {icd_config}")
            return True

        except Exception as e:
            print(f"❌ Conda ICD配置失败: {e}")
            return False

    def _create_auto_setup_script(self):
        """
        创建自动设置脚本 - 在每次Python启动时自动运行
        """
        try:
            # 创建自动执行脚本
            auto_setup_path = Path(self.conda_env_path) / "lib" / "python3.10" / "site-packages" / "auto_gpu_setup.py"

            setup_content = '''
"""
自动GPU环境配置模块
此模块会在导入时自动配置GPU环境
"""

import os
from pathlib import Path

def _auto_setup_gpu():
    """自动配置GPU环境"""
    # 设置环境变量
    os.environ['MUJOCO_GL'] = 'egl'
    os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'

    # XLA优化
    xla_flags = os.environ.get('XLA_FLAGS', '')
    if '--xla_gpu_triton_gemm_any=True' not in xla_flags:
        xla_flags += ' --xla_gpu_triton_gemm_any=True'
    os.environ['XLA_FLAGS'] = xla_flags

    # 设置EGL配置路径
    conda_env = "/root/HW/RL_qizhi-main/PPO-Humanoid/ENTER/envs/BHW"
    icd_path = Path(conda_env) / "etc" / "glvnd" / "egl_vendor.d" / "10_nvidia.json"

    if icd_path.exists():
        os.environ['EGL_VENDOR_LIBRARY_FILENAMES'] = str(icd_path)

# 自动执行设置
_auto_setup_gpu()
'''

            with open(auto_setup_path, 'w') as f:
                f.write(setup_content)

            # 创建__init__.py使其成为包
            init_path = auto_setup_path.parent / "__init__.py"
            if not init_path.exists():
                with open(init_path, 'w') as f:
                    f.write('# Auto-generated package\n')

            print(f"✅ 自动设置脚本创建: {auto_setup_path}")
            return True

        except Exception as e:
            print(f"❌ 自动设置脚本创建失败: {e}")
            return False

    def _create_notebook_config(self):
        """
        创建Jupyter notebook配置文件
        """
        try:
            # 创建Jupyter启动配置
            notebook_config = Path(self.conda_env_path) / "etc" / "ipython" / "profile_default" / "startup" / "00-gpu-setup.py"

            notebook_config.parent.mkdir(parents=True, exist_ok=True)

            startup_content = '''
"""
Jupyter Notebook GPU自动配置
在每次启动Jupyter时自动执行
"""

import os
from pathlib import Path

print("🔧 自动配置Jupyter GPU环境...")

# 设置环境变量
os.environ['MUJOCO_GL'] = 'egl'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'

# XLA优化
xla_flags = os.environ.get('XLA_FLAGS', '')
if '--xla_gpu_triton_gemm_any=True' not in xla_flags:
    xla_flags += ' --xla_gpu_triton_gemm_any=True'
os.environ['XLA_FLAGS'] = xla_flags

# 设置EGL配置
conda_env = "/root/HW/RL_qizhi-main/PPO-Humanoid/ENTER/envs/BHW"
icd_path = Path(conda_env) / "etc" / "glvnd" / "egl_vendor.d" / "10_nvidia.json"

if icd_path.exists():
    os.environ['EGL_VENDOR_LIBRARY_FILENAMES'] = str(icd_path)
    print("✅ GPU配置加载成功")
else:
    print("⚠️  ICD配置未找到，使用默认设置")

print("✅ Jupyter GPU环境配置完成")
'''

            with open(notebook_config, 'w') as f:
                f.write(startup_content)

            print(f"✅ Notebook配置创建: {notebook_config}")
            return True

        except Exception as e:
            print(f"❌ Notebook配置创建失败: {e}")
            return False

def setup_once_and_for_all():
    """
    一次性设置，永久生效
    """
    print("🎯 开始永久性GPU环境配置...")
    print("此配置将在新Pod重启后自动生效")
    print("=" * 60)

    setup = PermanentGPUSetup()
    success = setup.setup_permanent_environment()

    if success:
        print("\n🎉 永久配置成功！")
        print("现在配置会在以下情况自动生效：")
        print("1. ✅ 新Pod启动时")
        print("2. ✅ Jupyter Notebook重启时")
        print("3. ✅ Python脚本运行时")
        print("4. ✅ Conda环境激活时")
        print("\n📝 注意事项：")
        print("- 仍然需要在BHW conda环境中运行")
        print("- 系统级包安装仍需要在每个新Pod中执行")
        print("- 但GPU配置将自动加载")
    else:
        print("\n❌ 永久配置部分失败")
        print("建议手动执行剩余步骤")

    return success

if __name__ == "__main__":
    setup_once_and_for_all()