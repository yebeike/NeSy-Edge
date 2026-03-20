# RQ3 NuSy-Agent: Gemini or DeepSeek API + tool-calling loop (no LangChain)
# Run from project root. .env: gemini_api_key, deepseek_api_key (read-only).

import os
import sys
import json
import time
import urllib.request
import urllib.error

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from experiments.rq3.tools import get_tool_definitions_for_api, run_tool


def _get_deepseek_api_key():
    """Read DeepSeek API key from env or .env (read-only)."""
    v = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("deepseek_api_key")
    if v and isinstance(v, str):
        v = v.strip().strip('"').strip("'").strip("\u201c\u201d\u2018\u2019")
        if len(v) > 10:
            return v
    env_path = os.path.join(_PROJECT_ROOT, ".env")
    if os.path.exists(env_path):
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    for sep in ("=", ":"):
                        if sep in line and "deepseek" in line.lower() and "key" in line.lower():
                            parts = line.split(sep, 1)
                            if len(parts) == 2:
                                v = parts[1].strip().strip('"').strip("'").strip("\u201c\u201d\u2018\u2019")
                                if len(v) > 10:
                                    return v
                            break
        except Exception:
            pass
    return None


def _deepseek_tools():
    """OpenAI-compatible tool list for DeepSeek."""
    ours = get_tool_definitions_for_api()
    out = []
    for t in ours:
        out.append({
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("parameters", {}),
            },
        })
    return out


def _get_gemini_api_keys():
    """Read API key(s) from env or .env (read-only). Returns list [key1, key2, ...] for retry."""
    keys = []
    for env_name in ("GEMINI_API_KEY", "gemini_api_key", "gemini_api_key_2", "gemini api key"):
        v = os.environ.get(env_name)
        if v and isinstance(v, str):
            v = v.strip().strip('"').strip("'").strip("\u201c\u201d\u2018\u2019")
            if len(v) > 10:
                keys.append(v)
    env_path = os.path.join(_PROJECT_ROOT, ".env")
    if os.path.exists(env_path):
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    # key=value or key: value
                    for sep in ("=", ":"):
                        if sep in line and "gemini" in line.lower() and "key" in line.lower():
                            parts = line.split(sep, 1)
                            if len(parts) == 2:
                                v = parts[1].strip().strip('"').strip("'").strip("\u201c\u201d\u2018\u2019")
                                if len(v) > 10 and v not in keys:
                                    keys.append(v)
                            break
        except Exception:
            pass
    return keys if keys else [None]


def _gemini_tool_declarations():
    """Convert our tool defs to Gemini functionDeclarations format."""
    ours = get_tool_definitions_for_api()
    out = []
    for t in ours:
        params = t.get("parameters", {})
        # Gemini uses camelCase for top-level in parameters
        decl = {
            "name": t["name"],
            "description": t.get("description", ""),
            "parameters": {
                "type": "object",
                "properties": params.get("properties", {}),
                "required": params.get("required", []),
            },
        }
        out.append(decl)
    return out


def _call_gemini(contents: list, api_key: str, model: str = "gemini-2.0-flash") -> dict:
    """POST to Gemini generateContent. contents = list of {role, parts}."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    payload = {
        "contents": contents,
        "tools": [{"functionDeclarations": _gemini_tool_declarations()}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 2048},
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode() if e.fp else ""
        if e.code == 429:
            raise RuntimeError(
                f"Gemini API rate limit (429). Try again later or use another model (e.g. gemini-1.5-flash). Body: {body[:300]}"
            )
        raise RuntimeError(f"Gemini API error {e.code}: {body}")
    except Exception as e:
        raise RuntimeError(f"Gemini request failed: {e}")


def _extract_parts_and_function_calls(response: dict):
    """Return (list of parts from candidate, list of (name, args) for function calls)."""
    parts = []
    calls = []
    try:
        cands = response.get("candidates", [])
        if not cands:
            return [], []
        content = cands[0].get("content", {})
        for part in content.get("parts", []):
            parts.append(part)
            if "functionCall" in part:
                fc = part["functionCall"]
                calls.append((fc.get("name", ""), fc.get("args") or {}))
    except Exception:
        pass
    return parts, calls


def _deepseek_http(messages: list, api_key: str, model: str, max_tokens: int, use_tools: bool = True) -> dict:
    """Call DeepSeek API with urllib (no openai package). use_tools=False for Vanilla LLM baseline."""
    url = "https://api.deepseek.com/v1/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.2,
    }
    if use_tools:
        payload["tools"] = [{"type": "function", "function": {"name": t["name"], "description": t.get("description", ""), "parameters": t.get("parameters", {})}} for t in get_tool_definitions_for_api()]
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        if e.code == 429:
            raise RuntimeError("429 rate limit")
        raise RuntimeError(f"DeepSeek API {e.code}: {e.read().decode()[:200]}")


def _usage_from_resp(resp) -> dict:
    """Extract prompt_tokens, completion_tokens, total_tokens from OpenAI or raw API response."""
    u = getattr(resp, "usage", None) if hasattr(resp, "usage") else resp.get("usage") if isinstance(resp, dict) else None
    if not u:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    return {
        "prompt_tokens": getattr(u, "prompt_tokens", None) or u.get("prompt_tokens", 0),
        "completion_tokens": getattr(u, "completion_tokens", None) or u.get("completion_tokens", 0),
        "total_tokens": getattr(u, "total_tokens", None) or u.get("total_tokens", 0),
    }


def run_agent_deepseek(
    user_message: str,
    api_key: str = None,
    model: str = "deepseek-chat",
    max_tokens: int = 1024,
    max_rounds: int = 10,
) -> tuple:
    """
    Run agent via DeepSeek (OpenAI-compatible). Prefer deepseek-chat + low max_tokens to save cost.
    Returns (final_text, tool_calls_log, usage_dict). Uses openai package if available, else urllib.
    """
    api_key = api_key or _get_deepseek_api_key()
    if not api_key:
        return "(error: no DeepSeek API key. Set deepseek_api_key in .env)", [], {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    use_openai = True
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    except ImportError:
        use_openai = False
        client = None
    messages = [{"role": "user", "content": user_message}]
    tool_calls_log = []
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for _ in range(max_rounds):
        if use_openai:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=_deepseek_tools(),
                max_tokens=max_tokens,
                temperature=0.2,
            )
            u = _usage_from_resp(resp)
            total_usage["prompt_tokens"] += u["prompt_tokens"]
            total_usage["completion_tokens"] += u["completion_tokens"]
            total_usage["total_tokens"] += u["total_tokens"]
            msg = resp.choices[0].message if resp.choices else None
            if not msg:
                break
            msg_content = getattr(msg, "content", None) or (msg.get("content") if isinstance(msg, dict) else None)
            tool_calls = getattr(msg, "tool_calls", None) or (msg.get("tool_calls") if isinstance(msg, dict) else [])
            tc_list = []
            for tc in tool_calls or []:
                tid = getattr(tc, "id", None) or (tc.get("id") if isinstance(tc, dict) else "")
                fn = getattr(tc, "function", None) or (tc.get("function") if isinstance(tc, dict) else {})
                fname = getattr(fn, "name", None) or (fn.get("name") if isinstance(fn, dict) else "")
                fargs = getattr(fn, "arguments", None) or (fn.get("arguments", "{}") if isinstance(fn, dict) else "{}")
                tc_list.append((tid, fname, fargs))
        else:
            resp = _deepseek_http(messages, api_key, model, max_tokens)
            err = resp.get("error")
            if err:
                raise RuntimeError(err.get("message", str(err)))
            u = _usage_from_resp(resp)
            total_usage["prompt_tokens"] += u["prompt_tokens"]
            total_usage["completion_tokens"] += u["completion_tokens"]
            total_usage["total_tokens"] += u["total_tokens"]
            choice = (resp.get("choices") or [None])[0]
            msg = choice.get("message", {}) if choice else {}
            msg_content = msg.get("content") or ""
            tc_list = []
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function", {})
                tc_list.append((tc.get("id", ""), fn.get("name", ""), fn.get("arguments", "{}")))
        messages.append({
            "role": "assistant",
            "content": msg_content or "",
            "tool_calls": [{"id": tid, "type": "function", "function": {"name": n, "arguments": a}} for tid, n, a in tc_list],
        })
        if not tc_list:
            return (msg_content or "").strip() or "(no text)", tool_calls_log, total_usage
        for tid, name, fargs in tc_list:
            try:
                args = json.loads(fargs) if isinstance(fargs, str) else (fargs or {})
            except json.JSONDecodeError:
                args = {}
            result = run_tool(name, args)
            tool_calls_log.append({"name": name, "args": args, "result": result})
            messages.append({"role": "tool", "tool_call_id": tid, "content": result})
    return "(max rounds reached)", tool_calls_log, total_usage


def run_chat_only_deepseek(
    user_message: str,
    api_key: str = None,
    model: str = "deepseek-chat",
    max_tokens: int = 1024,
) -> tuple:
    """Single-turn chat, no tools. For Vanilla LLM / RAG baselines. Returns (response_text, usage_dict)."""
    api_key = api_key or _get_deepseek_api_key()
    empty_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    if not api_key:
        return "(error: no DeepSeek API key)", empty_usage
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": user_message}],
            max_tokens=max_tokens,
            temperature=0.2,
        )
        msg = resp.choices[0].message if resp.choices else None
        usage = _usage_from_resp(resp)
        return ((msg.content or "").strip() if msg else "", usage)
    except ImportError:
        resp = _deepseek_http(
            [{"role": "user", "content": user_message}],
            api_key, model, max_tokens, use_tools=False,
        )
        if resp.get("error"):
            err = resp.get("error")
            raise RuntimeError(err.get("message", str(err)) if isinstance(err, dict) else str(err))
        choice = (resp.get("choices") or [None])[0]
        msg = choice.get("message", {}) if choice else {}
        usage = _usage_from_resp(resp)
        return ((msg.get("content") or "").strip(), usage)


def run_agent(user_message: str, api_key: str = None, model: str = "gemini-2.0-flash", max_rounds: int = 10) -> tuple:
    """
    Run the diagnostic agent: user_message (e.g. raw log + instruction) -> (final_text, tool_calls_log).
    Returns (final_answer_text, list of {name, args, result} for each tool call).
    """
    key_list = _get_gemini_api_keys()
    if api_key:
        key_list = [api_key]
    if not key_list or key_list[0] is None:
        return "(error: no Gemini API key. Set GEMINI_API_KEY or gemini_api_key in .env)", []

    contents = [{"role": "user", "parts": [{"text": user_message}]}]
    tool_calls_log = []
    key_index = 0
    api_key = key_list[0]
    retry_delay = 20  # seconds on 429
    for _ in range(max_rounds):
        last_error = None
        for try_key in key_list[key_index:]:
            try:
                response = _call_gemini(contents, try_key, model=model)
                last_error = None
                api_key = try_key
                break
            except RuntimeError as e:
                last_error = e
                if "429" in str(e) and key_index + 1 < len(key_list):
                    key_index += 1
                    continue
                if "429" in str(e):
                    time.sleep(20)
                    continue
                raise
        if last_error:
            raise last_error
        model_parts, function_calls = _extract_parts_and_function_calls(response)
        if not model_parts:
            break
        contents.append({"role": "model", "parts": model_parts})
        if not function_calls:
            # Done: last part(s) should be text
            texts = []
            for p in model_parts:
                if "text" in p:
                    texts.append(p["text"])
            return ("\n".join(texts)).strip() or "(no text)", tool_calls_log

        # Execute all tools and append one user message with multiple function responses
        response_parts = []
        for name, args in function_calls:
            result = run_tool(name, args)
            tool_calls_log.append({"name": name, "args": args, "result": result})
            response_parts.append({"functionResponse": {"name": name, "response": {"result": result}}})
        contents.append({"role": "user", "parts": response_parts})
    return "(max rounds reached)", tool_calls_log


SYSTEM_INSTRUCTION = """You are an ops diagnostic assistant. For each log:
1. Call log_parser(raw_log, dataset) to get the log template.
2. Call causal_navigator(template_str, domain) to get root cause(s).
3. Call knowledge_retriever(query, dataset) with the root cause.
4. Write a short report: template, root cause(s), and one recommended action. Use only tool-returned facts."""

# Shorter for DeepSeek to save input tokens (same content, fewer words)
SYSTEM_INSTRUCTION_SHORT = """Ops diagnostic agent. Steps: 1) log_parser(raw_log, dataset) 2) causal_navigator(template, domain) 3) knowledge_retriever(root_cause, dataset) 4) Short report (template, root cause, action). Only use tool results."""


def run_agent_with_instruction(
    user_message: str,
    api_key: str = None,
    model: str = "gemini-2.0-flash",
    max_rounds: int = 10,
    backend: str = "gemini",
    max_tokens: int = 1024,
) -> tuple:
    """Run agent (Gemini or DeepSeek). Returns (text, tool_calls_log, usage_dict). usage_dict is empty for Gemini."""
    instruction = SYSTEM_INSTRUCTION_SHORT if backend == "deepseek" else SYSTEM_INSTRUCTION
    full_message = instruction + "\n\n---\n\n" + user_message
    if backend == "deepseek":
        rounds = min(max_rounds, 6)
        return run_agent_deepseek(full_message, api_key=api_key, model=model or "deepseek-chat", max_tokens=max_tokens, max_rounds=rounds)
    text, log = run_agent(full_message, api_key=api_key, model=model, max_rounds=max_rounds)
    return text, log, {}


if __name__ == "__main__":
    keys = _get_gemini_api_keys()
    key = keys[0] if keys else None
    if not key:
        print("No Gemini API key found.")
        sys.exit(1)
    # One test case
    test_msg = """Analyze the following log and give root cause and recommendation.

Raw log:
081110 211541 18 INFO dfs.DataNode: 10.250.15.198:50010 Starting thread to transfer block blk_4292382298896622412 to 10.250.15.240:50010

Dataset: HDFS"""
    print("Running agent (one test)...")
    answer, log = run_agent_with_instruction(test_msg, api_key=key)
    print("Tool calls:", len(log))
    for t in log:
        print(" ", t["name"], "->", (t["result"][:80] + "..." if len(t["result"]) > 80 else t["result"]))
    print("Final answer:", answer[:500] if len(answer) > 500 else answer)
