#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
构建 whl 包的辅助脚本
"""
import subprocess
import sys
import os
from pathlib import Path

def build_wheel():
    """构建 whl 包"""
    # 获取当前脚本所在目录
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    print("=" * 60)
    print("开始构建 paddleocr-vl-ov whl 包...")
    print("=" * 60)
    
    # 检查必要的文件
    if not (script_dir / "pyproject.toml").exists():
        print("错误: 未找到 pyproject.toml 文件")
        sys.exit(1)
    
    # 清理之前的构建文件
    print("\n清理之前的构建文件...")
    for dir_name in ["build", "dist", "*.egg-info"]:
        import glob
        for path in glob.glob(str(script_dir / dir_name)):
            import shutil
            if os.path.isdir(path):
                shutil.rmtree(path)
                print(f"  已删除: {path}")
            elif os.path.isfile(path):
                os.remove(path)
                print(f"  已删除: {path}")
    
    # 安装构建依赖
    print("\n安装构建依赖...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "--upgrade", 
            "setuptools", "wheel", "build"
        ])
    except subprocess.CalledProcessError as e:
        print(f"错误: 安装构建依赖失败: {e}")
        sys.exit(1)
    
    # 构建 whl 包
    print("\n开始构建 whl 包...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "build", "--wheel"
        ])
        print("\n" + "=" * 60)
        print("构建成功！")
        print("=" * 60)
        print(f"\nwhl 包位置: {script_dir / 'dist'}")
        print("\n可以使用以下命令安装:")
        print(f"  pip install {script_dir / 'dist' / 'paddleocr_vl_ov-*.whl'}")
    except subprocess.CalledProcessError as e:
        print(f"\n错误: 构建失败: {e}")
        sys.exit(1)
    except FileNotFoundError:
        # 如果没有 build 模块，使用 setuptools
        print("\n使用 setuptools 构建...")
        try:
            subprocess.check_call([
                sys.executable, "setup.py", "bdist_wheel"
            ])
            print("\n" + "=" * 60)
            print("构建成功！")
            print("=" * 60)
            print(f"\nwhl 包位置: {script_dir / 'dist'}")
        except subprocess.CalledProcessError as e:
            print(f"\n错误: 构建失败: {e}")
            sys.exit(1)

if __name__ == "__main__":
    build_wheel()

