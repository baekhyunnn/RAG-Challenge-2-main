from pydantic import BaseModel, Field
from typing import List, Union


def build_system_prompt(instruction: str = "", example: str = "", pydantic_schema: str = "") -> str:
    """
    构建标准化的系统提示词，强制模型输出符合特定 JSON Schema 的结构化结果。
    
    Args:
        instruction: 任务核心指令描述。
        example: 期望输出的 JSON 示例。
        pydantic_schema: 定义输出结构的 JSON Schema 字符串。
    Returns:
        拼接后的完整 System Prompt。
    """
    delimiter = "\n\n---\n\n"
    schema = f"你的回答必须是合法的 JSON 格式，并严格遵循如下 Schema 定义，字段顺序需保持一致：\n```\n{pydantic_schema}\n```"
    
    if example:
        example = delimiter + "期望的输出示例如下：\n" + example.strip()
    if schema:
        schema = delimiter + schema.strip()
    
    system_prompt = instruction.strip() + schema + example
    return system_prompt


class AnswerWithRAGContextPrompt:
    """通用知识库问答提示词"""
    instruction = """
你是一个专业的数据分析专家和企业智库助手。
请根据提供的上下文内容（Context）准确回答用户提出的问题。
如果上下文内容不足以回答该问题，请明确告知用户“根据现有文档无法回答”，严禁编造答案。
"""

    class AnswerSchema(BaseModel):
        """标准回答结构定义"""
        step_by_step_analysis: str = Field(description="基于上下文的逐步逻辑分析过程")
        reasoning_summary: str = Field(description="对推理过程的简要总结")
        relevant_pages: List[int] = Field(description="参考的原文页码列表")
        final_answer: str = Field(description="最终生成的正式回答内容")

    pydantic_schema = """
class AnswerSchema(BaseModel):
    step_by_step_analysis: str  # 逻辑分析过程
    reasoning_summary: str      # 推理总结
    relevant_pages: List[int]   # 参考页码
    final_answer: str           # 正式回答内容
"""
    
    system_prompt = build_system_prompt(instruction, pydantic_schema=pydantic_schema)
    user_prompt = "请基于以下上下文回答问题：\n\n【上下文】：\n{context}\n\n---\n\n【用户问题】：\n{question}"


class AnswerSchemaFixPrompt:
    """大模型输出自修复提示词 """
    system_prompt = """
你是一个 JSON 格式化助手。
你的任务是将大模型输出的原始内容（可能包含前言或截断）修复并格式化为合法的 JSON 对象。
你的回答必须严格以 "{" 开头，以 "}" 结尾，且中间不包含任何 Markdown 标识。
"""

    user_prompt = """
下面是定义的 JSON Schema 定义:
\"\"\"
{system_prompt}
\"\"\"

---

下面是需要你修复并格式化为合法 JSON 的 LLM 原始输出内容：
\"\"\"
{response}
\"\"\"
"""


class RerankingPrompt:
    """检索重排序专家提示词 """
    system_prompt_rerank_single_block = """
你是一个 RAG 检索重排专家。
你将收到一个查询问题和一个检索到的文本块，请根据其与查询的相关性进行评分。

评分准则：
1. 分析文本块中是否包含解决查询所需的关键信息点。
2. 相关性分数范围为 0-1（步长 0.1）：
   - 0.0: 完全无关。
   - 0.5: 提及了相关概念但无实质性答案。
   - 0.8: 包含直接相关的关键数据或描述。
   - 1.0: 完美匹配，包含解决问题所需的完整信息。
3. 仅基于提供的文本内容进行客观评价。
"""