import subprocess
import time
import os

def get_free_vram():
    """获取所有 GPU 的剩余显存总和 (MiB)"""
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            encoding='utf-8'
        )
        # 解析每行输出并求和
        free_memories = [int(x.strip()) for x in output.strip().split('\n') if x.strip()]
        return sum(free_memories)
    except Exception as e:
        print(f"检查显存时出错: {e}")
        return 0

def main():
    # 设定阈值为 50GB (1GB = 1024MB)
    threshold_mib = 50 * 1024  
    check_interval = 60  # 轮询间隔（秒）

    print(f"VRAM 监控已启动...")
    print(f"目标：剩余显存总和 >= 50GB ({threshold_mib} MiB)")
    print(f"检查频率：每 {check_interval} 秒一次")

    while True:
        current_free = get_free_vram()
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        
        print(f"[{current_time}] 当前总剩余显存: {current_free} MiB")

        if current_free >= threshold_mib:
            print(f"条件触发！检测到显存大于 50GB。正在启动 run.bat...")
            try:
                # 使用 shell=True 并在控制台中运行 bat 文件
                subprocess.run(["run.bat"], shell=True, check=True)
                print("run.bat 任务执行完成。")
                # 如果只想运行一次就退出，可以在这里 break。如果想循环运行，则去掉 break。
                # 考虑到典型需求是“等到有位置就跑实验”，通常运行一次即可退出。
                break
            except subprocess.CalledProcessError as e:
                print(f"执行 run.bat 时返回错误码: {e.returncode}")
                # 视情况决定是否继续轮询或退出
                break
            except Exception as e:
                print(f"执行时发生异常: {e}")
                break
        
        time.sleep(check_interval)

if __name__ == "__main__":
    main()
