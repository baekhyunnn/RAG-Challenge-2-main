# pdf_parsing.py
import os
import re
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import pymupdf  # PyMuPDF
import pypdf  # PyPDF作为备选
from tqdm import tqdm

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
_log = logging.getLogger(__name__)

def parse_pdf_with_pymupdf(pdf_path: Path) -> Optional[Dict[str, Any]]:
    """
    使用PyMuPDF解析PDF文档
    :param pdf_path: PDF文件路径
    :return: 解析结果字典，包含文本和元数据
    """
    try:
        _log.info(f"🔍 正在使用PyMuPDF解析: {pdf_path.name}")
        
        # 打开PDF文档
        pdf_doc = pymupdf.open(pdf_path)
        
        # 提取文本
        structured_paragraphs = []
        total_text = ""
        
        # 逐页提取文本
        for page_num in range(pdf_doc.page_count):
            try:
                page = pdf_doc.load_page(page_num)
                text = page.get_text()
                
                if text and text.strip():
                    # 清理文本
                    cleaned_text = re.sub(r'\s+', ' ', text.strip())
                    structured_paragraphs.append({
                        "page": page_num + 1,
                        "text": cleaned_text
                    })
                    total_text += cleaned_text + "\n\n"
            except Exception as page_error:
                _log.warning(f"⚠️  解析第{page_num+1}页失败: {page_error}")
                continue
        
        # 关闭文档（重要！）
        pdf_doc.close()
        
        if not total_text.strip():
            _log.warning(f"⚠️  {pdf_path.name} 无有效文本内容")
            return None
        
        # 构建结果
        result = {
            "plain_text": total_text.strip(),
            "structured_paragraphs": structured_paragraphs,
            "metainfo": {
                "filename": pdf_path.name,
                "filepath": str(pdf_path),
                "page_count": pdf_doc.page_count,
                "paragraph_count": len(structured_paragraphs),
                "text_length": len(total_text),
                "parser": "pymupdf"
            }
        }
        
        _log.info(f"✅ PyMuPDF解析成功：{pdf_path.name} | 页码数：{pdf_doc.page_count} | 有效段落数：{len(structured_paragraphs)}")
        return result
        
    except Exception as e:
        _log.error(f"❌ PyMuPDF解析失败：{pdf_path.name} | 错误原因：{e}")
        return None

def parse_pdf_with_pypdf(pdf_path: Path) -> Optional[Dict[str, Any]]:
    """
    使用PyPDF解析PDF文档（备选方案）
    :param pdf_path: PDF文件路径
    :return: 解析结果字典
    """
    try:
        _log.info(f"🔍 正在使用PyPDF解析: {pdf_path.name}")
        
        with open(pdf_path, 'rb') as file:
            reader = pypdf.PdfReader(file)
            
            structured_paragraphs = []
            total_text = ""
            
            # 逐页提取文本
            for page_num, page in enumerate(reader.pages):
                try:
                    text = page.extract_text()
                    
                    if text and text.strip():
                        # 清理文本
                        cleaned_text = re.sub(r'\s+', ' ', text.strip())
                        structured_paragraphs.append({
                            "page": page_num + 1,
                            "text": cleaned_text
                        })
                        total_text += cleaned_text + "\n\n"
                except Exception as page_error:
                    _log.warning(f"⚠️  解析第{page_num+1}页失败: {page_error}")
                    continue
            
            if not total_text.strip():
                _log.warning(f"⚠️  {pdf_path.name} 无有效文本内容")
                return None
            
            # 构建结果
            result = {
                "plain_text": total_text.strip(),
                "structured_paragraphs": structured_paragraphs,
                "metainfo": {
                    "filename": pdf_path.name,
                    "filepath": str(pdf_path),
                    "page_count": len(reader.pages),
                    "paragraph_count": len(structured_paragraphs),
                    "text_length": len(total_text),
                    "parser": "pypdf"
                }
            }
            
            _log.info(f"✅ PyPDF解析成功：{pdf_path.name} | 页码数：{len(reader.pages)} | 有效段落数：{len(structured_paragraphs)}")
            return result
            
    except Exception as e:
        _log.error(f"❌ PyPDF解析失败：{pdf_path.name} | 错误原因：{e}")
        return None

def parse_pdf(pdf_path: Path, fallback: bool = True) -> Optional[Dict[str, Any]]:
    """
    解析PDF文档，主函数
    :param pdf_path: PDF文件路径
    :param fallback: 是否启用备选解析器
    :return: 解析结果字典
    """
    if not pdf_path.exists():
        _log.error(f"❌ 文件不存在：{pdf_path}")
        return None
    
    if not pdf_path.suffix.lower() == '.pdf':
        _log.error(f"❌ 非PDF文件：{pdf_path}")
        return None
    
    # 优先使用PyMuPDF
    result = parse_pdf_with_pymupdf(pdf_path)
    
    # 如果失败且启用备选，尝试PyPDF
    if not result and fallback:
        _log.info(f"🔄 PyMuPDF解析失败，尝试PyPDF：{pdf_path.name}")
        result = parse_pdf_with_pypdf(pdf_path)
    
    if result:
        # 保存解析结果（可选）
        output_dir = Path("data/parsed")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / f"{pdf_path.stem}_parsed.json"
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            _log.debug(f"📝 解析结果已保存：{output_file}")
        except Exception as e:
            _log.warning(f"⚠️  保存解析结果失败：{e}")
    
    return result

def batch_parse_pdfs(pdf_dir: Path, output_dir: Path = None, ignore_failed: bool = True) -> List[Dict[str, Any]]:
    """
    批量解析PDF文档
    :param pdf_dir: PDF目录路径
    :param output_dir: 输出目录
    :param ignore_failed: 是否忽略失败的文件
    :return: 解析结果列表
    """
    if not pdf_dir.exists():
        _log.error(f"❌ PDF目录不存在：{pdf_dir}")
        return []
    
    # 收集PDF文件
    pdf_files = list(pdf_dir.glob("*.pdf")) + list(pdf_dir.glob("*.PDF"))
    
    if not pdf_files:
        _log.warning(f"⚠️  目录中未找到PDF文件：{pdf_dir}")
        return []
    
    _log.info(f"📂 找到 {len(pdf_files)} 个PDF文件，开始批量解析...")
    
    # 创建输出目录
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # 批量解析
    results = []
    failed_files = []
    
    for pdf_file in tqdm(pdf_files, desc="📄 解析PDF"):
        try:
            result = parse_pdf(pdf_file, fallback=True)
            
            if result:
                results.append(result)
                _log.debug(f"✅ 解析成功：{pdf_file.name}")
            else:
                failed_files.append(pdf_file.name)
                _log.warning(f"⚠️  解析失败：{pdf_file.name}")
                
                if not ignore_failed:
                    raise Exception(f"PDF解析失败：{pdf_file.name}")
                    
        except Exception as e:
            failed_files.append(pdf_file.name)
            _log.error(f"❌ 处理 {pdf_file.name} 时出错：{e}")
            
            if not ignore_failed:
                raise
    
    # 生成报告
    _log.info(f"📊 批量解析完成：成功 {len(results)} 个，失败 {len(failed_files)} 个")
    
    if failed_files:
        _log.warning(f"❌ 失败文件列表：{', '.join(failed_files)}")
    
    return results

def extract_tables_from_pdf(pdf_path: Path) -> List[Dict]:
    """
    从PDF中提取表格数据（基础版本）
    :param pdf_path: PDF文件路径
    :return: 表格数据列表
    """
    try:
        import pandas as pd
        
        _log.info(f"🔍 正在提取表格：{pdf_path.name}")
        
        doc = pymupdf.open(pdf_path)
        tables = []
        
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            
            # 尝试提取表格
            tabs = page.find_tables()
            
            if tabs.tables:
                for i, tab in enumerate(tabs.tables):
                    try:
                        # 转换为DataFrame
                        df = tab.to_pandas()
                        
                        tables.append({
                            "page": page_num + 1,
                            "table_index": i + 1,
                            "rows": df.shape[0],
                            "cols": df.shape[1],
                            "data": df.to_dict(orient="records"),
                            "html": df.to_html(index=False)
                        })
                        
                        _log.debug(f"📊 第{page_num+1}页找到表格{i+1}：{df.shape[0]}行×{df.shape[1]}列")
                        
                    except Exception as tab_error:
                        _log.warning(f"⚠️  处理表格失败：{tab_error}")
                        continue
        
        doc.close()
        
        _log.info(f"✅ 表格提取完成：{pdf_path.name}，共找到 {len(tables)} 个表格")
        return tables
        
    except Exception as e:
        _log.error(f"❌ 表格提取失败：{pdf_path.name}，错误：{e}")
        return []

# 测试函数
def test_parsing():
    """测试PDF解析功能"""
    print("=" * 60)
    print("🧪 PDF解析模块测试")
    print("=" * 60)
    
    # 测试单个文件
    test_pdf = Path("data/pdf/AI大模型面试题(102).pdf")
    
    if test_pdf.exists():
        print(f"\n测试文件：{test_pdf.name}")
        
        # 测试解析
        result = parse_pdf(test_pdf)
        
        if result:
            print(f"✅ 解析成功！")
            print(f"   页码数：{result['metainfo']['page_count']}")
            print(f"   段落数：{result['metainfo']['paragraph_count']}")
            print(f"   文本长度：{result['metainfo']['text_length']} 字符")
            print(f"   解析器：{result['metainfo']['parser']}")
            
            # 显示前3段文本
            paragraphs = result['plain_text'].split('\n\n')
            for i, para in enumerate(paragraphs[:3], 1):
                print(f"\n   段落 {i}（前100字符）：")
                print(f"   {para[:100]}...")
        else:
            print("❌ 解析失败！")
    else:
        print(f"⚠️  测试文件不存在：{test_pdf}")
        print("💡 请确保PDF文件已放置在 data/pdf/ 目录下")
    
    print("\n" + "=" * 60)
    print("✅ 测试完成")
    print("=" * 60)

if __name__ == "__main__":
    test_parsing()