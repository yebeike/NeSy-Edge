import os
from pathlib import Path
from dotenv import load_dotenv

# 加载 .env 文件中的环境变量 (如 OPENAI_API_KEY)
load_dotenv()

# --- 路径配置 ---
# 获取项目的根目录 (假设 config.py 在 src/ 下)
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = ROOT_DIR / "results"

# OpenStack 数据集路径
OPENSTACK_LOG_PATH = RAW_DATA_DIR / "OpenStack" / "OpenStack_2k.log"
OPENSTACK_GROUND_TRUTH_PATH = RAW_DATA_DIR / "OpenStack" / "OpenStack_2k.log_structured.csv"

# --- 参数配置 ---
# RQ1: Parsing
ENTROPY_THRESHOLD = 0.4  # 熵阈值 (需要后续实验微调)
SIMILARITY_THRESHOLD = 0.8 # 缓存匹配相似度

# LLM 配置
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_MODEL = "gpt-3.5-turbo"  # 或 "gpt-4" 用于更高精度

# --- 创建必要的目录 ---
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
(RESULTS_DIR / "figures").mkdir(exist_ok=True)
(RESULTS_DIR / "logs").mkdir(exist_ok=True)

print(f"✅ Configuration loaded. Root dir: {ROOT_DIR}")