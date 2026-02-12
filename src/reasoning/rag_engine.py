import sys
import os
import chromadb
from chromadb.utils import embedding_functions

class RAGEngine:
    def __init__(self, collection_name="openstack_knowledge"):
        # 使用本地持久化存储，模拟真实系统
        self.db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "chroma_db")
        self.client = chromadb.PersistentClient(path=self.db_path)
        
        # 使用本地 Embedding 模型 (会自动下载，约 80MB)
        self.ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        # 获取或创建集合
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.ef
        )

    def index_knowledge(self, file_path):
        """读取文本文件并建立索引"""
        if not os.path.exists(file_path):
            print(f"⚠️ Knowledge file not found: {file_path}")
            return

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 简单切分：按双换行符切分不同的故障案例
        docs = [chunk.strip() for chunk in content.split('\n\n') if chunk.strip()]
        
        # 存入数据库
        # 为了演示，先清空旧数据防止重复
        if self.collection.count() > 0:
            existing_ids = self.collection.get()['ids']
            self.collection.delete(ids=existing_ids)
            
        ids = [f"doc_{i}" for i in range(len(docs))]
        self.collection.add(documents=docs, ids=ids)
        print(f"✅ Indexed {len(docs)} knowledge chunks into ChromaDB.")

    def retrieve(self, query: str, n_results=1) -> list:
        """检索最相关的知识"""
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        return results['documents'][0]