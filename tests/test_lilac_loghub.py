import sys
import os
import time

# ================= 路径配置核心 (Path Setup) =================
# 1. 获取当前脚本的绝对路径 (.../NuSy-Edge/tests)
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. 获取项目根目录 (.../NuSy-Edge)
project_root = os.path.dirname(current_dir)

# 3. 定义 cloud_node 路径
cloud_node_path = os.path.join(project_root, "cloud_node")

# 4. 将 cloud_node 加入 Python 搜索路径，这样才能 import neuro_parser
sys.path.append(cloud_node_path)

# 5. 定义 Loghub 数据路径 (假设 loghub 与 NuSy-Edge 同级)
LOGHUB_PATH = os.path.abspath(os.path.join(project_root, "..", "loghub", "Linux", "Linux_2k.log"))

# ==========================================================

def run_test():
    print(f"📂 Project Root: {project_root}")
    print(f"📂 Target Log File: {LOGHUB_PATH}")

    if not os.path.exists(LOGHUB_PATH):
        print("❌ Error: Log file not found. Please check your directory structure.")
        return

    # 关键步骤：切换工作目录到 cloud_node
    # 这样 ChromaDB 才会读取 cloud_node/chroma_db 里的数据，而不是在 tests/ 下新建一个空的
    os.chdir(cloud_node_path)
    print(f"📂 Working Directory switched to: {os.getcwd()}")

    try:
        from neuro_parser import NeuroParser
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        return

    # 初始化 LILAC 解析器
    print("🚀 Initializing LILAC Parser...")
    parser = NeuroParser()

    print("\n=== Start Streaming Logs from Loghub (First 20 lines) ===\n")

    # 读取日志
    with open(LOGHUB_PATH, 'r', encoding='utf-8', errors='ignore') as f:
        logs = [next(f) for _ in range(20)]

    total_time_cache = 0
    hits = 0
    misses = 0

    for i, log in enumerate(logs):
        log = log.strip()
        if not log: continue

        # 打印原始日志（截断以防太长）
        print(f"[{i+1}] Raw: {log[:80]}...") 
        
        # 执行解析
        result = parser.parse(log)
        
        # 打印结果
        if result['source'] == 'cache':
            # 命中缓存：显示毫秒
            latency_ms = result['time'] * 1000
            print(f"   ⚡ HIT  | Time: {latency_ms:.2f}ms | Template: {result['template']}")
            hits += 1
            total_time_cache += result['time']
        else:
            # 未命中：显示秒
            print(f"   🐢 MISS | Time: {result['time']:.2f}s   | Template: {result['template']}")
            misses += 1

        print("-" * 50)

    # 统计信息
    print(f"\n=== LILAC Performance Summary ===")
    print(f"Total Logs:   {len(logs)}")
    print(f"Cache Hits:   {hits}")
    print(f"Cache Misses: {misses}")
    if hits > 0:
        avg_time = (total_time_cache / hits) * 1000
        print(f"Avg Hit Time: {avg_time:.2f} ms")
        if avg_time < 10:
            print("✅ Status: Excellent (Real-time performance achieved)")
        else:
            print("⚠️ Status: Good, but could be faster")

if __name__ == "__main__":
    run_test()