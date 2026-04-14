import os
import sys
import subprocess

print("===== PYTHON ENV TEST =====")
print("Python executable:", sys.executable)
print("Python version:", sys.version)
print("Current working directory:", os.getcwd())
print("PATH:", os.environ.get("PATH", ""))
print("PYTHONPATH:", os.environ.get("PYTHONPATH", ""))

print("\n===== PACKAGE TESTS =====")

try:
    import torch
    print("torch version:", torch.__version__)
    print("torch.cuda.is_available():", torch.cuda.is_available())
    print("torch.cuda.device_count():", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("GPU name:", torch.cuda.get_device_name(0))
except Exception as e:
    print("torch import/use failed:", repr(e))

try:
    import timm
    print("timm version:", timm.__version__)
except Exception as e:
    print("timm import failed:", repr(e))

print("\n===== NVIDIA-SMI =====")
try:
    result = subprocess.run(
        ["nvidia-smi"],
        capture_output=True,
        text=True,
        check=False
    )
    print("Return code:", result.returncode)
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
except Exception as e:
    print("nvidia-smi failed:", repr(e))

print("\n===== DONE =====")