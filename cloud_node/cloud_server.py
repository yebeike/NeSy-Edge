import time
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
# 引入我们刚才写好的 LILAC 解析器
from neuro_parser import NeuroParser
from config import HOST, PORT, logger

app = FastAPI()
command_queue = []

# 全局解析器实例
parser = None

class LogPayload(BaseModel):
    log: str
    source: str

@app.on_event("startup")
async def startup_event():
    global parser
    logger.info("🚀 System Startup: Initializing LILAC NeuroParser...")
    # 这里会自动加载 ChromaDB 和 Qwen3
    parser = NeuroParser()
    logger.info("✅ LILAC Parser Ready.")

@app.post("/analyze_log")
async def analyze_log(payload: LogPayload):
    log_content = payload.log.strip()
    if not log_content:
        return {"status": "ignored"}

    logger.info(f"📥 Received: {log_content[:50]}...") 

    try:
        # ==========================================
        # 核心升级：使用 LILAC 解析，而不是直接问 LLM
        # ==========================================
        # 1. 解析日志 (System 1 or System 2)
        parse_result = parser.parse(log_content)
        
        if not parse_result:
            return {"status": "skipped"}
            
        template = parse_result["template"]
        source = parse_result["source"] # 'cache' or 'llm'
        duration = parse_result["time"]
        
        logger.info(f"🔍 Analyzed via [{source.upper()}] | Time: {duration:.4f}s")
        logger.info(f"📝 Template: {template}")

        # 2. 基于模板的规则决策 (Symbolic Reasoning 雏形)
        # 相比于问 LLM "要不要重启"，直接匹配关键词更可控、更快
        decision = "IGNORE"
        
        # 定义一些“危险”的关键词
        critical_keywords = ["connection refused", "connection failed", "authentication failure", "fatal error"]
        
        # 检查模板中是否包含危险词 (不区分大小写)
        if any(k in template.lower() for k in critical_keywords):
            decision = "RESTART_NGINX"
            command_queue.append("RESTART_NGINX")
            logger.warning(f"🚨 CRITICAL FAULT MATCHED -> Command Queued.")
        
        return {
            "status": "ok", 
            "decision": decision, 
            "template": template,
            "source": source
        }

    except Exception as e:
        logger.error(f"Analysis Error: {e}")
        return {"status": "error"}

@app.get("/get_command")
async def get_command():
    if command_queue:
        cmd = command_queue.pop(0)
        logger.info(f"📤 Dispatching Command: {cmd}")
        return {"command": cmd}
    return {"command": None}

if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")