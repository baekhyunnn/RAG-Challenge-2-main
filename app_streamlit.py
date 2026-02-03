import os as global_os
import sys
import time
import importlib
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

class StreamlitRAG:
    def __init__(self):
        self.vector_store = None
        self.llm = None
        self.qa_chain = None
        self.doc_count = 0
        self.api_key = global_os.getenv("DASHSCOPE_API_KEY")
        self.initialize_system()
    
    def initialize_system(self):
        """系统初始化连接"""
        return self.init_components()

    def init_components(self) -> bool:
        """加载向量库组件"""
        try:
            from langchain_community.embeddings import DashScopeEmbeddings
            from langchain_chroma import Chroma
            
            if not self.api_key:
                return False

            embeddings = DashScopeEmbeddings(model="text-embedding-v1", dashscope_api_key=self.api_key)
            # 指向与 ingestion.py 相同的存储目录
            self.vector_store = Chroma(persist_directory="./vector_store", embedding_function=embeddings)
            
            try:
                self.doc_count = self.vector_store._collection.count()
            except:
                self.doc_count = 0
            
            if self.api_key:
                self.init_llm()
            return True
        except Exception as e:
            print(f"初始化组件失败: {e}")
            return False

    def init_llm(self):
        """配置大模型问答链"""
        try:
            from langchain_community.llms import Tongyi
            from langchain.chains.retrieval_qa.base import RetrievalQA
            from langchain.prompts import PromptTemplate
            
            self.llm = Tongyi(model="qwen-plus", dashscope_api_key=self.api_key, temperature=0.1)
            
            template = """请基于以下提供的上下文信息准确回答问题。如果上下文不足以回答，请直说不知道。
            上下文内容：\n{context}\n
            用户问题：{question}\n
            专业回答："""
            
            prompt = PromptTemplate(template=template, input_variables=["context", "question"])
            
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                retriever=self.vector_store.as_retriever(search_kwargs={"k": 7}),
                chain_type_kwargs={"prompt": prompt},
                return_source_documents=True
            )
        except Exception as e:
            print(f"LLM 引擎初始化失败: {e}")

    def ask_question(self, query: str):
        """执行问答检索"""
        if not self.qa_chain: return None
        try:
            start_time = time.time()
            # 适配新版 LangChain 调用
            response = self.qa_chain.invoke({"query": query})
            elapsed = round(time.time() - start_time, 2)
            
            return {
                "answer": response["result"],
                "sources": response.get("source_documents", []),
                "time": elapsed
            }
        except Exception as e:
            print(f"问答过程报错: {e}")
            return None

    def process_uploaded_file(self, uploaded_file):
        """处理文件上传并调用 ingestion 脚本"""
        try:
            # 1. 保存上传的文件
            save_dir = global_os.path.join("data", "pdf")
            global_os.makedirs(save_dir, exist_ok=True)
            with open(global_os.path.join(save_dir, uploaded_file.name), "wb") as f:
                f.write(uploaded_file.getbuffer())

            # 2. 注入 src 路径解决 ModuleNotFoundError
            root_path = global_os.getcwd()
            src_path = global_os.path.join(root_path, "src")
            if src_path not in sys.path:
                sys.path.insert(0, src_path)
            
            # 3. 动态加载并运行同步脚本
            import ingestion
            importlib.reload(ingestion)
            ingestion.main()
            
            # 4. 重新加载本地数据库
            return self.init_components()
        except Exception as e:
            st.error(f"知识库同步失败: {e}")
            return False