import time
import requests
import os
import docker
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from config import CLOUD_SERVER_URL, LOG_FILE_PATH, TARGET_CONTAINER, logger

class LogHandler(FileSystemEventHandler):
    def __init__(self, filepath):
        self.filepath = filepath
        self._f = None
        self._connect_file()

    def _connect_file(self):
        try:
            if os.path.exists(self.filepath):
                self._f = open(self.filepath, 'r')
                self._f.seek(0, 2)
            else:
                logger.warning(f"Log file not ready: {self.filepath}")
        except Exception as e:
            logger.error(f"File open error: {e}")

    def on_modified(self, event):
        if not event.is_directory and event.src_path.endswith("error.log"):
            if self._f is None:
                self._connect_file()
                return
            
            new_lines = self._f.read()
            if new_lines:
                line = new_lines.strip()
                logger.info(f"👁️  New Log: {line[:50]}...")
                self.send_to_cloud(line)

    def send_to_cloud(self, log_content):
        try:
            requests.post(f"{CLOUD_SERVER_URL}/analyze_log", 
                          json={"log": log_content, "source": "edge_node_1"}, 
                          timeout=5)
        except Exception as e:
            logger.error(f"Network Error: {e}")

def execute_command(cmd):
    if cmd == "RESTART_NGINX":
        try:
            client = docker.from_env()
            container = client.containers.get(TARGET_CONTAINER)
            logger.info(f"🔧 Executing: Restarting {TARGET_CONTAINER}...")
            container.restart()
            logger.info(f"✅ Restart Complete.")
        except Exception as e:
            logger.error(f"Docker Error: {e}")

def poll_server():
    try:
        # 使用 Session 可以复用 TCP 连接，稍微快一点
        with requests.Session() as s:
            resp = s.get(f"{CLOUD_SERVER_URL}/get_command", timeout=2)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("command"):
                    execute_command(data["command"])
    except Exception:
        pass # 轮询时不报错，保持安静

if __name__ == "__main__":
    logger.info(">>> NuSy-Edge Agent v0.2 Starting...")
    logger.info(f"Target: {CLOUD_SERVER_URL}")

    # 确保目录存在
    os.makedirs(os.path.dirname(LOG_FILE_PATH), exist_ok=True)
    if not os.path.exists(LOG_FILE_PATH):
        with open(LOG_FILE_PATH, 'w') as f: f.write("")

    # 启动监控
    handler = LogHandler(LOG_FILE_PATH)
    observer = Observer()
    observer.schedule(handler, path=os.path.dirname(LOG_FILE_PATH), recursive=False)
    observer.start()

    try:
        while True:
            poll_server()
            time.sleep(2)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()