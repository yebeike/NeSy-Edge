import logging

# ================= 配置中心 =================
# 核心修改：使用 1.7b 的轻量版
LLM_MODEL = "qwen3:1.7b"  

HOST = "0.0.0.0"     
PORT = 8000          

LLM_PARAMS = {
    "temperature": 0.1,     
    "num_predict": 1024,    
    "repeat_penalty": 1.1,
}

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("NuSy-Brain")
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)