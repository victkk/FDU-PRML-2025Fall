#!/usr/bin/env python3
"""
验证 notebook 中所有代码单元的运行
"""
import subprocess
import sys

print("=" * 70)
print("验证 Exercise2 的 code.ipynb 实现")
print("=" * 70)

# 使用 jupyter nbconvert 运行 notebook
result = subprocess.run(
    ["jupyter", "nbconvert", "--to", "notebook", "--execute",
     "--inplace", "code.ipynb"],
    cwd="/home/zhangzicheng/workspace/FDU-PRML-2025Fall/Exercise2",
    capture_output=True,
    text=True
)

if result.returncode == 0:
    print("\n✅ Notebook 执行成功！")
    print("\n现在可以使用以下命令将 notebook 转换为 PDF:")
    print("  cd /home/zhangzicheng/workspace/FDU-PRML-2025Fall/Exercise2")
    print("  jupyter nbconvert --to pdf code.ipynb --output 学号_名字_Exercise2.pdf")
else:
    print("\n❌ Notebook 执行失败！")
    print("\n错误信息:")
    print(result.stderr)
    sys.exit(1)
