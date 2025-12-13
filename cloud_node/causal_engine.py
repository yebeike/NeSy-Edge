import logging
import pandas as pd
import warnings
from causalnex.structure.dynotears import from_pandas_dynamic
from causalnex.structure import StructureModel
from langchain_ollama import ChatOllama
from config import LLM_MODEL

logger = logging.getLogger("NuSy-Causal")
warnings.filterwarnings("ignore") # 忽略 CausalNex 的一些废话警告

class CausalEngine:
    def __init__(self):
        # 初始化 LLM (用于生成约束)
        # 注意：这里我们用稍微高一点的 temperature，让它发挥一点常识推理能力
        self.llm = ChatOllama(model=LLM_MODEL, temperature=0.1)
        self.structure_model = None

    def generate_constraints(self, columns):
        """
        [ILS-CSL 核心]
        利用 LLM 的领域知识，生成 Tabu Edges (禁止的边)。
        防止算法算出 "Fire causes Smoke" 的反向因果。
        """
        logger.info("🧠 Asking LLM for causal constraints (Priors)...")
        tabu_edges = []
        
        # 简单的 Prompt 策略：让 LLM 判断两两关系
        # 在真实大规模场景中，这里会优化为只问高置信度的边
        # MVP 简化：只构建一个通用的 Prompt 询问逻辑
        
        prompt_template = (
            "Analyze the causal relationship between these two system events:\n"
            "Event A: '{a}'\n"
            "Event B: '{b}'\n"
            "Question: Is it IMPOSSIBLE for A to cause B? (e.g., 'Server Crash' causing 'High Latency' is possible, but 'High Latency' causing 'Server Crash' is less likely directly).\n"
            "Answer 'YES' if it is impossible/highly unlikely. Answer 'NO' otherwise.\n"
            "Output ONLY 'YES' or 'NO'."
        )

        # 这里的 columns 就是 ['ERR_DB_CONN', 'ERR_AUTH_FAIL'...]
        # 只有变量不多的情况下才做全排列，否则太慢。MVP 我们假设变量 < 5 个
        import itertools
        for col_a, col_b in itertools.permutations(columns, 2):
            # 简单过滤：自己不能导致自己
            if col_a == col_b: continue
            
            # 构造 Prompt
            prompt = prompt_template.format(a=col_a, b=col_b)
            
            try:
                # 这一步会有点慢，但在初始化阶段可以接受
                resp = self.llm.invoke(prompt).content.strip().upper()
                if "YES" in resp:
                    logger.info(f"🚫 Constraint Added: {col_a} -> {col_b} is FORBIDDEN.")
                    tabu_edges.append((col_a, col_b))
            except Exception as e:
                logger.error(f"LLM Constraint Error: {e}")

        return tabu_edges

    def learn_structure(self, df: pd.DataFrame, use_llm_constraints=True):
        logger.info(f"running DYNOTEARS on {len(df)} rows...")
        
        # === 修复 1: 强制转换为 float 类型 ===
        df = df.astype(float)
        
        # === 修复 2 (新): 重置索引为整数 ===
        # DYNOTEARS 要求索引必须是整数 (0, 1, 2...)，不能是时间戳
        df_for_causal = df.reset_index(drop=True)
        # ==================================

        # 1. 获取约束 (注意：这里我们用原 df 的列名，没问题)
        tabu_edges = []
        if use_llm_constraints:
            if len(df.columns) <= 5:
                tabu_edges = self.generate_constraints(df.columns)
            else:
                logger.warning("Too many columns for LLM constraints, skipping.")

        # 2. 运行 DYNOTEARS
        try:
            sm = from_pandas_dynamic(
                df_for_causal, # 使用重置索引后的数据
                p=1, 
                lambda_w=0.1, 
                lambda_a=0.1, 
                w_threshold=0.3,
                tabu_edges=tabu_edges
            )
            self.structure_model = sm
            return sm
        except Exception as e:
            import traceback
            logger.error(f"Structure Learning Failed: {e}")
            logger.error(traceback.format_exc())
            return None
        
        
    def find_root_cause(self, target_node, observed_data):
        """
        基于学习到的图进行简单的根因推断
        (MVP 简化版：找 target_node 的父节点中权重最大的)
        """
        if not self.structure_model:
            return None
            
        # 获取指向 target_node 的所有边
        # 这里的图包含 intra (同层) 和 inter (跨层)
        # 我们主要关心谁指向了 target_node
        try:
            # networkx 的图结构
            graph = self.structure_model
            
            # 找到所有前驱节点 (Predecessors)
            potential_causes = []
            for node in graph.predecessors(target_node):
                weight = graph.get_edge_data(node, target_node)['weight']
                potential_causes.append((node, weight))
            
            # 按权重排序，权重越大越可能是根因
            potential_causes.sort(key=lambda x: x[1], reverse=True)
            
            return potential_causes
        except Exception:
            return []