import time
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_ollama import ChatOllama
# 导入新的配置参数
from config import LLM_MODEL, LLM_PARAMS, HOST, PORT, logger 

app = FastAPI()
command_queue = []
llm = None

class LogPayload(BaseModel):
    log: str
    source: str

def warmup_model():
    global llm
    logger.info(f"🔥 System warming up... Loading model [{LLM_MODEL}]")
    start = time.time()
    
    # 关键修改：在这里传入优化参数！
    llm = ChatOllama(
        model=LLM_MODEL,
        temperature=LLM_PARAMS["temperature"],
        num_predict=LLM_PARAMS["num_predict"],
        repeat_penalty=LLM_PARAMS["repeat_penalty"]
    )
    
    try:
        # 发送简单的 hi 触发加载
        llm.invoke("hi")
        duration = time.time() - start
        logger.info(f"✅ Model Loaded! Cold start time: {duration:.2f}s")
    except Exception as e:
        logger.error(f"❌ Model load failed: {e}")

@app.on_event("startup")
async def startup_event():
    warmup_model()

@app.post("/analyze_log")
async def analyze_log(payload: LogPayload):
    # 只处理看起来像报错的日志，过滤掉空行
    log_content = payload.log.strip()
    if not log_content:
        return {"status": "ignored"}

    logger.info(f"📥 Log: {log_content[:60]}...") 

    # 极简 Prompt
    prompt = (
        f"Log: '{log_content}'\n\n"
        "Instructions:\n"
        "1. Analyze if this log indicates a database connection failure.\n"
        "2. CRITICAL RULE: Direct output only. NO thinking, NO explanation, NO tags like <think>.\n" # 强力禁止思考标签
        "3. Output 'RESTART_NGINX' if critical, otherwise 'IGNORE'.\n"
    )

    try:
        inference_start = time.time()
        
        response = llm.invoke(prompt)
        
        # 清洗数据：万一它还是输出了 <think>，我们手动删掉
        raw_content = response.content.strip()
        # 简单过滤：如果包含 RESTART_NGINX 就认为中了，不管它有没有废话
        if "RESTART_NGINX" in raw_content.upper():
            decision = "RESTART_NGINX"
        else:
            decision = "IGNORE"
        
        inference_time = time.time() - inference_start
        
        logger.info(f"🤖 Decision: [{decision}] (Raw len: {len(raw_content)}) | Time: {inference_time:.4f}s")

        if decision == "RESTART_NGINX":
            command_queue.append("RESTART_NGINX")
            logger.warning("🚨 CRITICAL FAULT -> Command Queued.")
        
        return {"status": "ok", "decision": decision}

    except Exception as e:
        logger.error(f"AI Error: {e}")
        return {"status": "error"}

@app.get("/get_command")
async def get_command():
    if command_queue:
        cmd = command_queue.pop(0)
        logger.info(f"📤 Sending Command: {cmd}")
        return {"command": cmd}
    return {"command": None}

if __name__ == "__main__":
    # log_level="warning" 配合上面的 config，彻底根治刷屏
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")