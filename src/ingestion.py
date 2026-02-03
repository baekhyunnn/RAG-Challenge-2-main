import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings

# 加载环境变量
load_dotenv()

# 获取项目路径
PROJECT_ROOT = Path(__file__).parent.parent
PDF_DIR = PROJECT_ROOT / "data" / "pdf"
VECTOR_STORE_DIR = PROJECT_ROOT / "vector_store"

# 确保能导入同目录下的 pdf_parsing
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from pdf_parsing import parse_pdf

def main():
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("❌ 错误: 请在 .env 文件中配置 DASHSCOPE_API_KEY")
        return

    # 1. 初始化 Embedding 模型
    embeddings = DashScopeEmbeddings(model="text-embedding-v1", dashscope_api_key=api_key)

    # 2. 扫描并解析 PDF
    pdf_files = list(PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"📂 文件夹 {PDF_DIR} 中未找到 PDF 文件")
        return

    all_chunks = []
    all_metadatas = []

    for pdf_path in pdf_files:
        print(f"📄 正在解析: {pdf_path.name}")
        result = parse_pdf(pdf_path)
        if result and "plain_text" in result:
            # 简单分块逻辑
            text = result["plain_text"]
            chunks = [text[i:i+1000] for i in range(0, len(text), 800)]
            all_chunks.extend(chunks)
            all_metadatas.extend([{"source": pdf_path.name} for _ in chunks])

    # 3. 写入 Chroma 数据库
    if all_chunks:
        print(f"🧪 正在构建向量库，共 {len(all_chunks)} 个块...")
        Chroma.from_texts(
            texts=all_chunks,
            embedding=embeddings,
            persist_directory=str(VECTOR_STORE_DIR),
            metadatas=all_metadatas
        )
        print(f"✅ 成功！向量库已保存在: {VECTOR_STORE_DIR}")
    else:
        print("⚠️ 未提取到任何文本内容")

if __name__ == "__main__":
    main()