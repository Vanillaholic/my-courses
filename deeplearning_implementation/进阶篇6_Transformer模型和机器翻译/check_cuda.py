import sys
print("Python版本:", sys.version)
print("\n" + "="*50)

try:
    import numpy as np
    print("NumPy版本:", np.__version__)
    print("NumPy安装路径:", np.__file__)
except Exception as e:
    print("NumPy导入失败:", str(e))

print("\n" + "="*50)

try:
    import torch
    print("PyTorch版本:", torch.__version__)
    print("PyTorch安装路径:", torch.__file__)
    print("\nCUDA是否可用:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA版本:", torch.version.cuda)
        print("GPU数量:", torch.cuda.device_count())
        print("当前GPU:", torch.cuda.current_device())
        print("GPU名称:", torch.cuda.get_device_name(0))
    else:
        print("CUDA不可用，可能的原因:")
        print("  1. 没有安装CUDA版本的PyTorch")
        print("  2. 没有NVIDIA GPU")
        print("  3. CUDA驱动未安装或版本不匹配")
except Exception as e:
    print("PyTorch检查失败:", str(e))

print("\n" + "="*50)
print("设备信息:")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("当前使用设备:", device)
