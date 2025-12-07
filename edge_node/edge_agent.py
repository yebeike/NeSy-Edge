import time
import requests
import json
import os
import docker
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# ================= 配置区域 =================
# 1. Cloud (Mac) 的地址，请替换为你刚才获取的 Mac IP
SERVER_URL = "http://192.168.0.101:8000"  

# 2. 我们要监控的日志文件路径
# 注意：这是 Docker 映射出来的路径，Docker 会自动创建 logs 文件夹
LOG_FILE_PATH = "./target_system/logs/nginx/error.log"

# 3. 目标容器名称 (和 docker-compose.yml 里的一致)
TARGET_CONTAINER = "web_server_target"

# ===========================================

class LogFileHandler(FileSystemEventHandler):
    """
    负责监控日志文件变化的处理器
    """
    def __init__(self, filepath):
        self.filepath = filepath
        self._f = None
        self._connect_file()

    def _connect_file(self):
        # 尝试打开文件并定位到末尾（只读新产生的日志）
        try:
            if os.path.exists(self.filepath):
                self._f = open(self.filepath, 'r')
                self._f.seek(0, 2) # 跳到文件末尾
            else:
                print(f"[Warn] Log file not found yet: {self.filepath}")
        except Exception as e:
            print(f"[Error] Failed to open log file: {e}")

    def on_modified(self, event):
        # 当文件被修改时触发
        if not event.is_directory and event.src_path.endswith("error.log"):
            if self._f is None:
                self._connect_file()
                return

            # 读取新增的内容
            new_lines = self._f.read()
            if new_lines:
                print(f"[Log] Detected new log: {new_lines.strip()}")
                self.send_to_cloud(new_lines)

    def send_to_cloud(self, log_content):
        # 将日志发送给 Brain (Mac)
        payload = {"log": log_content, "source": "edge_node_1"}
        try:
            requests.post(f"{SERVER_URL}/analyze_log", json=payload, timeout=2)
            print("[Network] Log sent to Cloud successfully.")
        except Exception as e:
            print(f"[Network] Failed to send log to Cloud: {e}")

def execute_command(command):
    """
    执行来自 Cloud 的修复指令
    """
    print(f"[Action] Received command: {command}")
    
    if command == "RESTART_NGINX":
        try:
            client = docker.from_env()
            container = client.containers.get(TARGET_CONTAINER)
            print(f"[Docker] Restarting container: {TARGET_CONTAINER}...")
            container.restart()
            print(f"[Docker] Container restarted successfully!")
        except Exception as e:
            print(f"[Docker] Error restarting container: {e}")
    else:
        print(f"[Action] Unknown command, ignoring.")

def poll_server():
    """
    轮询 Cloud 是否有新指令
    """
    try:
        # 这里的 endpoint /get_command 是我们稍后要在 Mac 上写的
        response = requests.get(f"{SERVER_URL}/get_command", timeout=2)
        if response.status_code == 200:
            data = response.json()
            cmd = data.get("command")
            if cmd:
                execute_command(cmd)
    except Exception:
        # 忽略连接错误（因为 Mac 服务可能还没起）
        pass

if __name__ == "__main__":
    print(">>> NuSy-Edge Agent Started...")
    print(f">>> Monitoring: {LOG_FILE_PATH}")
    print(f">>> Cloud Server: {SERVER_URL}")

    # 1. 启动日志监控
    # 确保目录存在，否则 watchdog 会报错
    log_dir = os.path.dirname(LOG_FILE_PATH)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
        # 创建一个空的 error.log 防止报错
        with open(LOG_FILE_PATH, 'w') as f:
            f.write("")

    event_handler = LogFileHandler(LOG_FILE_PATH)
    observer = Observer()
    observer.schedule(event_handler, path=log_dir, recursive=False)
    observer.start()

    # 2. 主循环：轮询指令
    try:
        while True:
            poll_server()
            time.sleep(2) # 每 2 秒问一次 Cloud
    except KeyboardInterrupt:
        observer.stop()
    
    observer.join()