import uvicorn
import pandas as pd
import asyncio
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
# 引入我们的三大核心组件
from neuro_parser import NeuroParser
from aggregator import LogAggregator
from causal_engine import CausalEngine
from config import HOST, PORT, logger

app = FastAPI()
command_queue = []

# 全局实例
parser = None
aggregator = None
brain = None

class LogPayload(BaseModel):
    log: str
    source: str

@app.on_event("startup")
async def startup_event():
    global parser, aggregator, brain
    logger.info("🚀 System Startup: Initializing NuSy-Edge Brain...")
    
    # 1. 启动感知层 (LILAC)
    parser = NeuroParser()
    
    # 2. 启动记忆层 (Aggregator) - 10秒一个窗口
    aggregator = LogAggregator(window_size_seconds=10)
    
    # 3. 启动推理层 (CausalNex)
    brain = CausalEngine()
    
    logger.info("✅ All Systems Ready (Neuro-Symbolic Loop Active).")

async def run_root_cause_analysis():
    """
    [核心逻辑] 后台运行的根因分析任务
    """
    logger.info("🧠 Triggering Root Cause Analysis (Background)...")
    
    # 1. 获取时序数据 (最近 60 个窗口)
    df = aggregator.get_dataframe(limit_windows=60)
    
    # 如果数据太少，没法算因果，直接跳过
    # (至少需要 10 行数据才能跑出像样的结果，MVP演示时我们可以放宽要求)
    if len(df) < 5:
        logger.info("⏳ Not enough data for causal inference yet. Waiting...")
        return

    # 2. 运行 DYNOTEARS 学习因果图
    # 在真实场景中，这一步不需要每次都跑，可以每分钟跑一次。
    # MVP 为了演示效果，我们每次触发都跑。
    sm = brain.learn_structure(df, use_llm_constraints=True)
    
    if not sm:
        logger.error("❌ Causal Learning failed.")
        return

    # 3. 根因定位
    # 我们假设当前检测到了 'ERR_NGINX_500' (这是我们在 Aggregator 里映射的变量名)
    # 我们问系统：谁导致了 Nginx 报错？
    target_symptom = "ERR_DB_CONN" # 这里我们假设 Nginx 报的是连接错误
    
    # 注意：DYNOTEARS 输出的节点通常带 lag 后缀 (如 ERR_DB_CONN_lag1)
    # 我们在 find_root_cause 里已经处理了逻辑
    causes = brain.find_root_cause(target_symptom + "_lag0", None)
    
    if not causes:
        # 尝试找不带后缀的
        causes = brain.find_root_cause(target_symptom, None)

    logger.info(f"🧐 Diagnosis Report: Detected Symptom [{target_symptom}]")
    logger.info(f"🔍 Potential Root Causes: {causes}")

    # 4. 生成决策 (自愈)
    # 如果发现某个原因的权重很高，就生成修复指令
    if causes:
        top_cause, weight = causes[0]
        
        # 决策逻辑：根据根因下药
        if "ERR_DB_CONN" in top_cause or "ERR_DB_FAIL" in top_cause:
            # 如果根因是 DB 问题 (不管是 lag1 还是 lag0)
            # 注意：这里我们做了一个假设，如果 DB 报错本身就是根因（自回归），或者 DB 导致了其他，都重启 DB
            logger.warning("🚨 ROOT CAUSE IDENTIFIED: DATABASE FAILURE")
            if "RESTART_DB" not in command_queue:
                command_queue.append("RESTART_DB")
                logger.info("💊 Prescription: RESTART_DB command queued.")
        
        elif "ERR_AUTH_FAIL" in top_cause:
            logger.warning("🚨 ROOT CAUSE: AUTH ISSUE (Maybe Config?)")
            # 可以在这里加 RESTART_NGINX
            
    else:
        logger.info("🤷 No clear root cause found yet. Need more data.")


@app.post("/analyze_log")
async def analyze_log(payload: LogPayload, background_tasks: BackgroundTasks):
    """
    接收日志 -> 解析 -> 存入 -> (可选) 触发推理
    """
    log_content = payload.log.strip()
    if not log_content: return {"status": "ignored"}

    # 1. LILAC 解析
    parse_result = parser.parse(log_content)
    if not parse_result: return {"status": "skipped"}
    
    template = parse_result["template"]
    source = parse_result["source"]
    
    logger.info(f"📥 [{source.upper()}] Template: {template}")

    # 2. 存入 Aggregator
    aggregator.add_log(template)

    # 3. 触发机制
    # 如果是严重错误，触发后台分析
    # 为了防止太频繁，我们在 aggregator 里其实可以做个限流，这里简单处理
    if "connection refused" in template.lower() or "failed" in template.lower():
        # 使用 BackgroundTasks 确保不阻塞 Edge 的 HTTP 请求
        background_tasks.add_task(run_root_cause_analysis)

    return {"status": "ok", "template": template}

@app.get("/get_command")
async def get_command():
    if command_queue:
        cmd = command_queue.pop(0)
        logger.info(f"📤 Dispatching Command: {cmd}")
        return {"command": cmd}
    return {"command": None}

if __name__ == "__main__":
    # 记得激活 nusy 环境运行
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")