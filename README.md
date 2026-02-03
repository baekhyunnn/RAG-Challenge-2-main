企业级 AI 知识库助理 (基于 RAG 架构)
1. 项目概述
本项目是一个基于 检索增强生成 (RAG) 技术的智能问答系统。它通过本地向量数据库和 LLM（通义千问）的结合，实现了对企业私有 PDF 文档的精准问答，有效解决了大模型在垂直领域下的“幻觉”问题。

2. 核心技术栈
LLM / Embedding: 通义千问 (DashScope API)

向量数据库: ChromaDB (持久化版本)

RAG 框架: LangChain & Streamlit

文档解析: PyMuPDF + PyPDF (双引擎容错)

3. 技术亮点与实践细节
在开发过程中，我针对 RAG 链路中的多个环节进行了手动优化，而非简单的 API 调用：

文档切分策略：放弃了传统的等长切分，采用了递归字符分块逻辑，并设置了 200 字符的语义重叠区（Overlap），显著改善了跨块信息丢失导致的检索失败。

向量索引持久化：针对 Windows 环境下常见的 FAISS 路径兼容性问题，重构为 ChromaDB 原生存储方案，实现了数据的本地化持久存储与高效加载。

检索链路调优：针对特定文档下模型回复“不知道”的情况，通过实验将 Top-K 检索深度提升至 7，并配合 Prompt 提示词优化，增强了系统获取上下文的完整度。

结构化输出保障：在 prompts.py 中设计了标准的 JSON Schema 校验，并加入了 AnswerSchemaFix 修复层，通过 LLM 自愈能力处理非标准格式输出，保障了前端解析的稳定性。

4. 项目结构
src/ingestion.py: 文档扫描与向量化自动流水线。

src/pdf_parsing.py: 处理 PDF 文本提取，支持双解析器退避逻辑。

src/retrieval.py: 检索引擎核心，负责向量比对与相似度召回。

src/prompts.py: 存放经过优化的系统提示词与结构化输出定义。

5. 如何运行
依赖安装：pip install -r requirements.txt。

Key 配置：在 .env 中配置 DASHSCOPE_API_KEY。

数据预处理：将 PDF 放入 data/pdf，运行 python src/ingestion.py。

启动应用：运行 streamlit run app_streamlit_beautiful.py。