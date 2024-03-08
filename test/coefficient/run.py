import subprocess

dir_names = ["corr_coeff", "mi_gief", "mi_kde", "mi_model", "mi_quant", "mic"]

for name in dir_names:
    print("\n" + "-" * 20)
    print(f"testing {name}...")
    result = subprocess.run(f"python {name}/test.py", shell=True, capture_output=True)
    print(result.stdout.decode("utf-8"))
    
    if result.returncode == 0:
        print(f"{name}单元测试通过")
    else:
        print(f"{name}单元测试未通过")