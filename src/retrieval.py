import os
import json
import faiss
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv
import dashscope
from dashscope import TextEmbedding
from pathlib import Path
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# 加载环境变量并初始化DashScope
load_dotenv()
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")
if not dashscope.api_key:
    raise EnvironmentError("❌ 未找到DASHSCOPE_API_KEY，请配置.env文件")

# 全局配置
CONFIG = {
    "EMBEDDING_MODEL": TextEmbedding.Models.text_embedding_v1,
    "VECTOR_STORE_DIR": Path("vector_store"),  # 与ingestion.py向量库目录一致
    "TOP_K_DEFAULT": 5,  # 默认检索数量
    "SIMILARITY_THRESHOLD": 0.7,  # 相似度阈值，低于此值的结果会被过滤
    "MAX_CHUNK_LENGTH": 1000,  # 最大文本块长度（字符）
    "ENABLE_RERANKING": True,  # 是否启用重排序
}

class VectorStoreManager:
    """向量库管理器，负责加载和管理向量索引"""
    
    def __init__(self):
        self.vector_store_dir = CONFIG["VECTOR_STORE_DIR"]
        self.merged_index_path = self.vector_store_dir / "merged_faiss.index"
        self.merged_chunks_path = self.vector_store_dir / "merged_chunks.json"
        self.indices = {}  # 缓存加载的索引
        self.chunks_data = {}  # 缓存文本块数据
        
    def load_merged_index(self) -> Tuple[Optional[faiss.Index], Optional[Dict]]:
        """
        加载合并的向量索引
        :return: (索引对象, 文本块数据) 或 (None, None)
        """
        try:
            if not self.merged_index_path.exists() or not self.merged_chunks_path.exists():
                logger.warning("⚠️  未找到合并的向量索引，尝试加载单个文档索引")
                return None, None
            
            # 加载合并索引
            index = faiss.read_index(str(self.merged_index_path))
            
            # 加载合并的文本块数据
            with open(self.merged_chunks_path, "r", encoding="utf-8") as f:
                chunks_data = json.load(f)
            
            logger.info(f"✅ 已加载合并向量索引，包含 {chunks_data.get('total_chunks', 0)} 个文本块")
            return index, chunks_data
            
        except Exception as e:
            logger.error(f"❌ 加载合并索引失败：{str(e)}")
            return None, None
    
    def load_single_indices(self) -> List[Tuple[str, faiss.Index, Dict]]:
        """
        加载所有单个文档的向量索引
        :return: 列表，每个元素为(文档名, 索引对象, 文本块数据)
        """
        indices_list = []
        
        # 查找所有索引文件
        index_files = list(self.vector_store_dir.glob("*_faiss.index"))
        
        for index_file in index_files:
            try:
                # 解析文档信息
                file_name = index_file.stem
                if "_faiss" in file_name:
                    doc_name = file_name.replace("_faiss", "")
                    doc_type = "pdf" if "pdf" in file_name.lower() else "doc"
                else:
                    doc_name = file_name
                    doc_type = "unknown"
                
                # 查找对应的文本块文件
                chunks_pattern = f"*{doc_name}*chunks.json"
                chunks_files = list(self.vector_store_dir.glob(chunks_pattern))
                
                if not chunks_files:
                    logger.warning(f"⚠️  跳过 {doc_name}：无对应文本块文件")
                    continue
                
                chunks_file = chunks_files[0]
                
                # 加载索引
                index = faiss.read_index(str(index_file))
                
                # 加载文本块数据
                with open(chunks_file, "r", encoding="utf-8") as f:
                    chunks_data = json.load(f)
                
                indices_list.append((doc_name, doc_type, index, chunks_data))
                logger.debug(f"✅ 已加载 {doc_name} ({doc_type}) 的向量索引")
                
            except Exception as e:
                logger.error(f"❌ 加载 {index_file.name} 失败：{str(e)}")
                continue
        
        logger.info(f"✅ 已加载 {len(indices_list)} 个文档的向量索引")
        return indices_list
    
    def get_all_indices(self):
        """
        获取所有可用的向量索引
        :return: 优先返回合并索引，如果没有则返回所有单个索引
        """
        # 先尝试加载合并索引
        merged_index, merged_chunks = self.load_merged_index()
        if merged_index and merged_chunks:
            return [("merged", "merged", merged_index, merged_chunks)]
        
        # 如果没有合并索引，加载所有单个索引
        return self.load_single_indices()

def get_query_embedding(query: str) -> np.ndarray:
    """
    生成查询语句的Embedding（调用DashScope）
    :param query: 用户查询问题
    :return: 归一化后的查询向量
    """
    if not query or query.strip() == "":
        raise ValueError("查询语句不能为空")
    
    try:
        # 调用DashScope生成单句Embedding
        resp = TextEmbedding.call(
            model=CONFIG["EMBEDDING_MODEL"],
            input=[query.strip()]
        )
        
        if resp.status_code != 200:
            logger.error(f"❌ Embedding生成失败：{resp.message}")
            raise ValueError(f"Embedding生成失败：{resp.message}")
        
        # 提取并转换向量
        emb = np.array(resp["output"]["embeddings"][0]["embedding"], dtype=np.float32)
        
        # 归一化（与FAISS内积索引匹配）
        emb = emb.reshape(1, -1)
        faiss.normalize_L2(emb)
        
        logger.debug(f"✅ 查询Embedding生成成功，维度：{emb.shape}")
        return emb
        
    except Exception as e:
        logger.error(f"❌ 生成查询Embedding时出错：{str(e)}")
        raise

def rerank_results(query: str, results: List[Dict], top_k: int = 5) -> List[Dict]:
    """
    对检索结果进行重排序
    :param query: 查询问题
    :param results: 检索结果列表
    :param top_k: 返回数量
    :return: 重排序后的结果
    """
    if not CONFIG["ENABLE_RERANKING"] or len(results) <= 1:
        return results[:top_k]
    
    try:
        # 简单的基于关键词和长度的重排序策略
        # 可以根据需求替换为更复杂的重排序模型
        
        def calculate_score(chunk: str, query: str) -> float:
            """计算文本块的相关性得分"""
            chunk_lower = chunk.lower()
            query_lower = query.lower()
            
            # 1. 关键词匹配得分
            keyword_score = 0
            query_words = set(query_lower.split())
            chunk_words = set(chunk_lower.split())
            
            matched_words = query_words.intersection(chunk_words)
            if matched_words:
                keyword_score = len(matched_words) / len(query_words)
            
            # 2. 长度得分（优先适中长度的文本块）
            chunk_len = len(chunk)
            if chunk_len < 100:  # 太短的文本块得分降低
                length_score = 0.5
            elif chunk_len > 800:  # 太长的文本块得分降低
                length_score = 0.7
            else:
                length_score = 1.0
            
            # 3. 结构得分（包含问题/答案格式的得分更高）
            structure_score = 1.0
            question_markers = ["问题：", "q：", "问：", "题目：", "试题：", "？"]
            answer_markers = ["答案：", "a：", "答：", "解答：", "解析："]
            
            has_question = any(marker in chunk_lower for marker in question_markers)
            has_answer = any(marker in chunk_lower for marker in answer_markers)
            
            if has_question and has_answer:
                structure_score = 1.5  # 同时包含问题和答案的文本块得分更高
            elif has_question or has_answer:
                structure_score = 1.2  # 包含其中一个的得分稍高
            
            # 综合得分
            total_score = (keyword_score * 0.4 + 
                          length_score * 0.2 + 
                          structure_score * 0.4)
            
            return total_score
        
        # 为每个结果计算得分
        scored_results = []
        for result in results:
            score = calculate_score(result["chunk"], query)
            scored_results.append({
                **result,
                "rerank_score": score
            })
        
        # 按得分排序
        scored_results.sort(key=lambda x: x["rerank_score"], reverse=True)
        
        logger.info(f"✅ 重排序完成，返回前 {top_k} 个结果")
        return scored_results[:top_k]
        
    except Exception as e:
        logger.warning(f"⚠️  重排序失败，返回原始结果：{str(e)}")
        return results[:top_k]

def retrieve_similar_chunks(
    query: str, 
    top_k: int = None, 
    doc_filter: Optional[str] = None,
    similarity_threshold: float = None
) -> List[Dict[str, Any]]:
    """
    检索与查询最相似的文本块（增强版）
    :param query: 用户查询问题
    :param top_k: 返回最相似的k个文本块
    :param doc_filter: 文档过滤器（指定检索某个文档）
    :param similarity_threshold: 相似度阈值
    :return: 检索结果列表，包含详细元数据
    """
    if top_k is None:
        top_k = CONFIG["TOP_K_DEFAULT"]
    
    if similarity_threshold is None:
        similarity_threshold = CONFIG["SIMILARITY_THRESHOLD"]
    
    try:
        # 1. 生成查询向量
        query_emb = get_query_embedding(query)
        logger.info(f"🔍 开始检索：'{query}'")
        
        # 2. 初始化向量库管理器
        store_manager = VectorStoreManager()
        
        # 3. 获取所有向量索引
        all_indices = store_manager.get_all_indices()
        if not all_indices:
            logger.error("❌ 无可用向量索引，请先运行 ingestion.py")
            return []
        
        # 4. 执行检索
        all_results = []
        
        for doc_name, doc_type, index, chunks_data in all_indices:
            # 应用文档过滤器
            if doc_filter and doc_name != doc_filter and doc_filter != "all":
                continue
            
            # 获取文本块列表
            text_chunks = chunks_data.get("chunks", [])
            if not text_chunks:
                logger.warning(f"⚠️  {doc_name} 无文本块数据")
                continue
            
            # 设置检索数量（针对单个索引）
            search_k = min(top_k * 2, len(text_chunks))  # 检索稍多一些的结果用于后续筛选
            
            # 执行相似度检索
            distances, indices = index.search(query_emb, search_k)
            
            # 处理检索结果
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx < 0 or idx >= len(text_chunks):  # 无效索引
                    continue
                
                # 应用相似度阈值
                if distance < similarity_threshold:
                    continue
                
                chunk_text = text_chunks[idx]
                
                # 截断过长的文本块
                if len(chunk_text) > CONFIG["MAX_CHUNK_LENGTH"]:
                    chunk_text = chunk_text[:CONFIG["MAX_CHUNK_LENGTH"]] + "..."
                
                result = {
                    "chunk": chunk_text,
                    "similarity": float(distance),
                    "doc_name": doc_name,
                    "doc_type": doc_type,
                    "chunk_index": int(idx),
                    "total_chunks_in_doc": len(text_chunks),
                    "source_info": f"来自文档：{doc_name} ({doc_type})"
                }
                
                all_results.append(result)
        
        if not all_results:
            logger.warning("⚠️  未检索到相似文本块，尝试降低相似度阈值")
            # 如果没找到结果，放宽阈值重新检索
            if similarity_threshold > 0.3:
                return retrieve_similar_chunks(
                    query, top_k, doc_filter, similarity_threshold - 0.1
                )
            return []
        
        # 5. 按相似度排序
        all_results.sort(key=lambda x: x["similarity"], reverse=True)
        
        # 6. 去重（基于文本内容的去重）
        unique_results = []
        seen_chunks = set()
        
        for result in all_results:
            chunk_hash = hash(result["chunk"][:200])  # 取前200字符的哈希作为去重依据
            if chunk_hash not in seen_chunks:
                seen_chunks.add(chunk_hash)
                unique_results.append(result)
        
        # 7. 重排序
        final_results = rerank_results(query, unique_results, top_k)
        
        logger.info(f"✅ 检索完成，找到 {len(final_results)} 个相关文本块")
        for i, result in enumerate(final_results[:3], 1):
            logger.debug(f"  结果 {i}: {result['doc_name']} (相似度: {result['similarity']:.3f})")
        
        return final_results
        
    except Exception as e:
        logger.error(f"❌ 检索过程中出错：{str(e)}")
        return []

def retrieve_chunks_with_context(
    query: str, 
    top_k: int = 5,
    include_context: bool = True
) -> List[Dict[str, Any]]:
    """
    检索相似文本块，并包含上下文信息
    :param query: 用户查询
    :param top_k: 返回数量
    :param include_context: 是否包含上下文
    :return: 包含上下文的结果列表
    """
    results = retrieve_similar_chunks(query, top_k=top_k)
    
    if not include_context or not results:
        return results
    
    # 加载所有文本块数据以获取上下文
    store_manager = VectorStoreManager()
    all_indices = store_manager.load_single_indices()
    
    doc_chunks_map = {}
    for doc_name, _, _, chunks_data in all_indices:
        doc_chunks_map[doc_name] = chunks_data.get("chunks", [])
    
    # 为每个结果添加上下文
    for result in results:
        doc_name = result["doc_name"]
        chunk_idx = result["chunk_index"]
        chunks = doc_chunks_map.get(doc_name, [])
        
        if not chunks:
            continue
        
        # 添加上下文（前后各1个chunk）
        start_idx = max(0, chunk_idx - 1)
        end_idx = min(len(chunks), chunk_idx + 2)  # +2因为切片是前闭后开
        
        context_chunks = chunks[start_idx:end_idx]
        context_text = "\n\n...\n\n".join(context_chunks)
        
        result["context"] = context_text
        result["context_range"] = f"{start_idx+1}-{end_idx}"
    
    return results

def get_available_documents() -> List[Dict[str, str]]:
    """
    获取所有可用的文档列表
    :return: 文档信息列表
    """
    try:
        store_manager = VectorStoreManager()
        indices = store_manager.load_single_indices()
        
        documents = []
        for doc_name, doc_type, _, chunks_data in indices:
            documents.append({
                "name": doc_name,
                "type": doc_type.upper(),
                "chunks_count": chunks_data.get("total_chunks", 0),
                "title": chunks_data.get("doc_name", doc_name)
            })
        
        # 按文档类型排序
        documents.sort(key=lambda x: x["type"])
        return documents
        
    except Exception as e:
        logger.error(f"❌ 获取文档列表失败：{str(e)}")
        return []

# 测试检索功能
if __name__ == "__main__":
    print("=" * 60)
    print("🔍 RAG检索模块测试")
    print("=" * 60)
    
    # 测试1：获取可用文档
    print("\n📋 可用文档列表：")
    docs = get_available_documents()
    for doc in docs:
        print(f"  - {doc['name']} ({doc['type']}): {doc['chunks_count']} 个文本块")
    
    # 测试2：基本检索
    print("\n🧪 测试基本检索：")
    test_queries = [
        "Transformer模型的核心机制是什么？",
        "大语言模型的训练需要哪些数据？",
        "RAG系统的工作流程是怎样的？"
    ]
    
    for query in test_queries[:1]:  # 只测试第一个
        print(f"\n查询：'{query}'")
        results = retrieve_similar_chunks(query, top_k=3)
        
        if results:
            print(f"找到 {len(results)} 个结果：")
            for i, result in enumerate(results, 1):
                print(f"\n{i}. [{result['doc_name']}] 相似度: {result['similarity']:.3f}")
                print(f"   来源：{result['source_info']}")
                print(f"   内容：{result['chunk'][:150]}...")
        else:
            print("⚠️  未找到相关结果")
    
    print("\n" + "=" * 60)
    print("✅ 检索模块测试完成")
    print("=" * 60)