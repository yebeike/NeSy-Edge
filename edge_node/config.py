# edge_node/config.py
import logging

# ================= 配置中心 =================
# !!! 请务必确认 Mac 的 IP !!!
CLOUD_SERVER_URL = "http://192.168.0.102:8000"

# 监控路径
LOG_FILE_PATH = "./target_system/logs/nginx/error.log"
TARGET_CONTAINER = "web_server_target"

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S" # Edge 端只显示时间，省空间
)
logger = logging.getLogger("NuSy-Agent")