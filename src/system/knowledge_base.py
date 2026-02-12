import chromadb
from chromadb.utils import embedding_functions
import os
import uuid

class KnowledgeBase:
    def __init__(self, collection_name="nusy_edge_kb", persist_path="data/chroma_db"):
        """
        初始化 RAG 引擎 (基于 ChromaDB)
        """
        # 自动定位到项目根目录下的 data/chroma_db
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.persist_path = os.path.join(project_root, persist_path)
        
        # 初始化客户端
        self.client = chromadb.PersistentClient(path=self.persist_path)
        
        # 使用轻量级 Embedding 模型 (all-MiniLM-L6-v2 速度快，效果好)
        self.ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        # 获取或创建集合
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.ef,
            metadata={"hnsw:space": "cosine"} # 使用余弦相似度
        )
        print(f"📘 [KnowledgeBase] Initialized collection '{collection_name}' at {self.persist_path}")

    def add_knowledge(self, raw_logs, templates, dataset_type):
        """
        将由于是 Demo，我们直接把 (Log, Template) 对存进去。
        raw_logs: 标准日志内容
        templates: 对应的 Ground Truth 模板
        dataset_type: "OpenStack" 或 "HDFS" (用于过滤)
        """
        ids = [str(uuid.uuid4()) for _ in range(len(raw_logs))]
        metadatas = [{"template": t, "dataset": dataset_type} for t in templates]
        
        # 分批写入，防止内存爆炸
        batch_size = 500
        total = len(raw_logs)
        print(f"   [RAG] Ingesting {total} logs into Knowledge Base...")
        
        for i in range(0, total, batch_size):
            end = min(i + batch_size, total)
            self.collection.add(
                documents=raw_logs[i:end],
                metadatas=metadatas[i:end],
                ids=ids[i:end]
            )

    def search(self, query_log, dataset_type, top_k=3):
        """
        核心功能：检索。
        给定一条变异日志 (query_log)，找出知识库里最像的 top_k 条标准日志。
        """
        results = self.collection.query(
            query_texts=[query_log],
            n_results=top_k,
            where={"dataset": dataset_type} # 确保只在对应的数据集里搜
        )
        
        # 整理返回结果
        retrieved_items = []
        if results['documents']:
            docs = results['documents'][0]
            metas = results['metadatas'][0]
            for doc, meta in zip(docs, metas):
                retrieved_items.append({
                    "raw_log": doc,
                    "template": meta['template']
                })
                
        return retrieved_items

    def clear(self):
        """清空知识库 (重跑实验用)"""
        self.client.delete_collection(self.collection.name)
        print("🗑️ [KnowledgeBase] Collection cleared.")