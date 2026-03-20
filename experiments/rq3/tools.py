# RQ3 Agent tools: log_parser, causal_navigator, knowledge_retriever
# Run from project root: PYTHONPATH=. python experiments/rq3/tools.py

import os
import re
import sys
import json

# Project root (experiments/rq3/tools.py -> two levels up)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Paths
DATA_PROCESSED = os.path.join(_PROJECT_ROOT, "data", "processed")
CAUSAL_KNOWLEDGE_PATH = os.path.join(DATA_PROCESSED, "causal_knowledge.json")


def get_causal_knowledge_path(override: str = None) -> str:
    """Resolve causal graph path: override > env CAUSAL_KNOWLEDGE_PATH > default."""
    if override and override.strip():
        return override.strip()
    env_path = (os.environ.get("CAUSAL_KNOWLEDGE_PATH") or "").strip()
    if env_path:
        return env_path
    return CAUSAL_KNOWLEDGE_PATH


# Lazy singletons; causal cache keyed by path for Data Efficiency (multiple graphs)
_edge_node = None
_kb = None
_causal_kb_cache = {}  # path -> list of facts


def _norm(t: str) -> str:
    """Normalize template for comparison: strip, collapse spaces, unify placeholders."""
    if not t or not isinstance(t, str):
        return ""
    t = t.strip().lower()
    # Unify placeholders so RQ1 <*> and RQ2 [*] match
    for placeholder in ["<*>", "[*]", "<* >"]:
        t = t.replace(placeholder, " @ ")
    while "  " in t:
        t = t.replace("  ", " ")
    return t


def log_parser(raw_log: str, dataset: str) -> str:
    """
    Parse raw log to template (RQ1). dataset in ["HDFS", "OpenStack"].
    Returns template string or empty on error.
    """
    global _edge_node
    if _edge_node is None:
        from src.system.edge_node import NuSyEdgeNode
        _edge_node = NuSyEdgeNode()
    try:
        template, *_ = _edge_node.parse_log_stream(raw_log, dataset)
        return (template or "").strip()
    except Exception as e:
        return f"(parse_error: {e})"


def causal_navigator(template_str: str, domain: str, causal_path: str = None) -> str:
    """
    Look up root cause(s) for a given phenomenon template in causal graph.
    domain in ["hdfs", "openstack"] (lowercase as in causal_knowledge).
    Returns JSON string: list of {source_template, relation, weight} or [] if none.
    """
    path = causal_path or get_causal_knowledge_path()
    if not os.path.exists(path):
        return json.dumps({"error": "causal_knowledge.json not found", "path": path})

    global _causal_kb_cache
    if path not in _causal_kb_cache:
        with open(path, "r", encoding="utf-8") as f:
            _causal_kb_cache[path] = json.load(f)
    _causal_kb = _causal_kb_cache[path]

    domain_lower = domain.lower() if domain else "hdfs"
    template_norm = _norm(template_str)
    results = []
    for fact in _causal_kb:
        if fact.get("domain") != domain_lower:
            continue
        target = _norm(fact.get("target_template", ""))
        if not target:
            continue
        # Match: exact, or significant token overlap (RQ1 uses <*>, RQ2 uses [*])
        entry = {
            "source_template": fact.get("source_template", ""),
            "relation": fact.get("relation", ""),
            "weight": fact.get("weight", 0),
        }
        if template_norm == target:
            results.append(entry)
            continue
        # Token overlap: split by space and @
        t_tokens = set(re.findall(r"[a-z0-9]+|@", template_norm))
        g_tokens = set(re.findall(r"[a-z0-9]+|@", target))
        if t_tokens & g_tokens and len(t_tokens & g_tokens) >= min(3, len(g_tokens) * 0.4):
            results.append(entry)
    # Sort by weight descending
    results.sort(key=lambda x: abs(x.get("weight", 0)), reverse=True)
    return json.dumps(results[:10] if results else [])


def knowledge_retriever(query: str, dataset: str, top_k: int = 3) -> str:
    """
    Retrieve reference documents by root cause description (or template text).
    dataset in ["HDFS", "OpenStack"].
    Returns concatenated reference text for the LLM.
    """
    global _kb
    if _kb is None:
        try:
            from src.system.knowledge_base import KnowledgeBase
            _kb = KnowledgeBase()
        except Exception as e:
            return f"(knowledge_base_error: {e})"
    try:
        items = _kb.search(query, dataset, top_k=top_k)
        if not items:
            return "(no references found)"
        parts = []
        for i, it in enumerate(items, 1):
            parts.append(f"[{i}] log: {it.get('raw_log', '')}\n    template: {it.get('template', '')}")
        return "\n".join(parts)
    except Exception as e:
        return f"(retrieve_error: {e})"


def get_tool_definitions_for_api():
    """Return tool declarations in OpenAI-style format (for Gemini REST / generic API)."""
    return [
        {
            "name": "log_parser",
            "description": "Parse a raw log line into a structured log template (event type). Use this first on the raw log.",
            "parameters": {
                "type": "object",
                "properties": {
                    "raw_log": {"type": "string", "description": "The raw log line to parse."},
                    "dataset": {"type": "string", "enum": ["HDFS", "OpenStack"], "description": "Dataset source."},
                },
                "required": ["raw_log", "dataset"],
            },
        },
        {
            "name": "causal_navigator",
            "description": "Look up root cause(s) for a given log template in the causal graph. Call this after log_parser with the template and domain.",
            "parameters": {
                "type": "object",
                "properties": {
                    "template_str": {"type": "string", "description": "The log template string (output of log_parser)."},
                    "domain": {"type": "string", "enum": ["hdfs", "openstack"], "description": "Domain (lowercase): hdfs or openstack."},
                },
                "required": ["template_str", "domain"],
            },
        },
        {
            "name": "knowledge_retriever",
            "description": "Retrieve reference documents (historical logs and templates) by a query, e.g. root cause template. Use after causal_navigator to get solution references.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Query text, e.g. root cause template or description."},
                    "dataset": {"type": "string", "enum": ["HDFS", "OpenStack"], "description": "Dataset filter."},
                    "top_k": {"type": "integer", "description": "Number of references to retrieve (default 3).", "default": 3},
                },
                "required": ["query", "dataset"],
            },
        },
    ]


def run_tool(name: str, args: dict) -> str:
    """Execute one tool by name with given args. Returns string result."""
    args = dict(args or {})
    if name == "log_parser":
        return log_parser(args.get("raw_log", ""), args.get("dataset", "HDFS"))
    if name == "causal_navigator":
        return causal_navigator(
            args.get("template_str", ""),
            args.get("domain", "hdfs"),
            args.get("causal_path") or get_causal_knowledge_path(),
        )
    if name == "knowledge_retriever":
        return knowledge_retriever(
            args.get("query", ""),
            args.get("dataset", "HDFS"),
            int(args.get("top_k", 3)),
        )
    return json.dumps({"error": f"unknown tool: {name}"})


if __name__ == "__main__":
    # Quick sanity check: run tools on first test case
    test_set_path = os.path.join(DATA_PROCESSED, "rq3_test_set.json")
    if not os.path.exists(test_set_path):
        print("rq3_test_set.json not found in data/processed")
        sys.exit(1)
    with open(test_set_path, "r", encoding="utf-8") as f:
        cases = json.load(f)
    c = cases[0]
    raw, dataset = c["raw_log"], c["dataset"]
    print("=== log_parser ===")
    tpl = log_parser(raw, dataset)
    print(tpl)
    print("\n=== causal_navigator ===")
    domain = "hdfs" if dataset == "HDFS" else "openstack"
    root = causal_navigator(tpl, domain)
    print(root)
    print("\n=== knowledge_retriever (query = root cause or template) ===")
    refs = knowledge_retriever(tpl or raw[:100], dataset, top_k=2)
    print(refs[:500] + "..." if len(refs) > 500 else refs)
