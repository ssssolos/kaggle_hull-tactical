import sys
import subprocess

print("当前解释器路径:", sys.executable)
print("scipy 是否可导入:", end=" ")
try:
    import scipy
    print("✅ 成功")
    print("scipy 版本:", scipy.__version__)
except Exception as e:
    print("❌ 失败")
    print("错误详情:", e)

# 检查 pip 安装位置
print("\n已安装的 scipy 路径:")
result = subprocess.run([sys.executable, "-m", "pip", "show", "scipy"], 
                         capture_output=True, text=True)
print(result.stdout or "⚠️ 未找到 scipy")