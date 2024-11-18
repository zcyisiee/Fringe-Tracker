import subprocess
import sys
import os
import signal
import time

def run_servers():
    # 获取当前目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    frontend_dir = os.path.join(current_dir, 'fringe-tracker')
    
    try:
        # 启动后端服务
        backend = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "api:app", "--host", "127.0.0.1", "--port", "8000"],
            cwd=current_dir
        )
        
        # 等待后端服务启动
        time.sleep(2)
        
        # 启动前端服务
        frontend = subprocess.Popen(
            ["npm", "run", "dev"],
            cwd=frontend_dir
        )
        
        print("\n干涉条纹追踪器已启动！")
        print("请在浏览器中访问: http://localhost:3000")
        print("按Ctrl+C停止服务\n")
        
        # 等待用户中断
        frontend.wait()
        
    except KeyboardInterrupt:
        print("\n正在停止服务...")
    finally:
        # 确保清理所有进程
        try:
            backend.terminate()
            frontend.terminate()
        except:
            pass
        
        print("服务已停止")

if __name__ == "__main__":
    run_servers()
