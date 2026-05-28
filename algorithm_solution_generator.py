# -*- coding: utf-8 -*-
"""
AI 算法题解与复杂度分析生成器
基于 FastAPI + DeepSeek + LangChain RAG
"""

import os
import json
import re
from typing import List, Optional, Dict
from pydantic import BaseModel, ValidationError
from openai import OpenAI, APIError, AuthenticationError, RateLimitError
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# 环境变量配置
DEFAULT_API_BASE = "https://api.deepseek.com"
DEFAULT_MODEL_NAME = "deepseek-chat"

KNOWLEDGE_BASE_PATH = os.path.join(os.path.dirname(__file__), "docs", "算法题型知识库.md")

documents_cache = []

def load_algorithm_knowledge_base():
    """加载算法知识库到内存"""
    global documents_cache
    
    if documents_cache:
        return
    
    if not os.path.exists(KNOWLEDGE_BASE_PATH):
        print(f"警告: 知识库文件不存在: {KNOWLEDGE_BASE_PATH}")
        return
    
    try:
        with open(KNOWLEDGE_BASE_PATH, "r", encoding="utf-8") as f:
            content = f.read()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n## ", "\n### ", "\n\n", "\n"]
        )
        
        chunks = text_splitter.split_text(content)
        documents_cache = [Document(page_content=chunk) for chunk in chunks]
        print(f"已加载 {len(documents_cache)} 个知识库文档片段")
        
    except Exception as e:
        print(f"加载知识库失败: {str(e)}")

def simple_similarity(text1: str, text2: str) -> float:
    """简单的关键词相似度计算"""
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if not words1 or not words2:
        return 0.0
    
    intersection = words1 & words2
    union = words1 | words2
    
    return len(intersection) / len(union) if union else 0.0

def search_algorithm_knowledge(query: str, top_k: int = 3) -> List[str]:
    """从知识库中检索相关算法题型"""
    if not documents_cache:
        load_algorithm_knowledge_base()
    
    if not documents_cache:
        return []
    
    scores = []
    for doc in documents_cache:
        score = simple_similarity(query, doc.page_content)
        scores.append((score, doc.page_content))
    
    scores.sort(key=lambda x: x[0], reverse=True)
    
    results = []
    for score, content in scores[:top_k]:
        if score > 0.05:
            results.append(content)
    
    return results

class AlgorithmRequest(BaseModel):
    """算法题解请求模型"""
    problem: str
    language: str = "Python"
    difficulty: str = "medium"

def extract_json_from_markdown(text: str) -> str:
    """从 Markdown 代码块中提取 JSON"""
    # 匹配 ```json ... ```
    pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
    match = re.search(pattern, text)
    if match:
        return match.group(1).strip()
    
    # 匹配 ``` ... ```
    pattern = r'```\s*([\s\S]*?)\s*```'
    match = re.search(pattern, text)
    if match:
        return match.group(1).strip()
    
    return text

def parse_json_safely(text: str) -> Optional[dict]:
    """安全解析 JSON，处理各种异常情况"""
    try:
        # 先尝试直接解析
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # 尝试提取 JSON
    extracted = extract_json_from_markdown(text)
    try:
        return json.loads(extracted)
    except json.JSONDecodeError:
        pass
    
    # 尝试修复常见问题
    try:
        # 移除末尾多余逗号
        fixed = re.sub(r',\s*([}\]])', r'\1', extracted)
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass
    
    return None

def generate_algorithm_solution(request: AlgorithmRequest) -> dict:
    """生成算法题解"""
    # 验证输入
    if not request.problem or not request.problem.strip():
        return {
            "problem_type": [],
            "core_idea": "错误：题目描述不能为空",
            "data_structure": "",
            "step_by_step_solution": [],
            "reference_code": "",
            "time_complexity": "",
            "space_complexity": "",
            "edge_cases": [],
            "common_mistakes": [],
            "optimization": "",
            "rag_context": ""
        }
    
    # 获取环境变量
    api_key = os.getenv("OPENAI_API_KEY")
    api_base = os.getenv("OPENAI_BASE_URL", DEFAULT_API_BASE)
    model_name = os.getenv("MODEL_NAME", DEFAULT_MODEL_NAME)
    
    # 检查 API Key
    if not api_key:
        return {
            "problem_type": [],
            "core_idea": "错误：未配置 OPENAI_API_KEY 环境变量",
            "data_structure": "",
            "step_by_step_solution": [],
            "reference_code": "",
            "time_complexity": "",
            "space_complexity": "",
            "edge_cases": [],
            "common_mistakes": [],
            "optimization": "",
            "rag_context": ""
        }
    
    # RAG 检索
    retrieved_docs = search_algorithm_knowledge(request.problem, top_k=3)
    context_str = "\n\n".join(retrieved_docs) if retrieved_docs else "未检索到强相关模板"
    
    # 构建 Prompt
    system_prompt = f"""你是一名资深算法工程师和 LeetCode 题解专家。
你的任务是根据用户输入的算法题描述，生成结构化的算法题解。

## RAG 检索到的算法知识库内容
{context_str}

## Few-shot 示例（严格参考此格式生成）
示例输入：
{{
  "problem": "给定一个整数数组 nums 和一个目标值 target，请你在该数组中找出和为目标值的两个整数，并返回它们的数组下标。",
  "language": "Python",
  "difficulty": "easy"
}}

示例输出：
{{
  "problem_type": ["数组", "哈希表"],
  "core_idea": "使用哈希表记录已经遍历过的元素及其下标，在遍历当前元素时判断 target - nums[i] 是否已经出现。",
  "data_structure": "使用哈希表（字典），因为需要在 O(1) 平均时间内判断补数是否存在。",
  "step_by_step_solution": [
    "1. 初始化一个空哈希表，用于存储元素值到下标的映射。",
    "2. 遍历数组 nums。",
    "3. 对每个元素 nums[i]，计算 complement = target - nums[i]。",
    "4. 如果 complement 已经在哈希表中，返回对应下标和当前下标。",
    "5. 否则将当前元素和下标存入哈希表。"
  ],
  "reference_code": "class Solution:\\n    def twoSum(self, nums: List[int], target: int) -> List[int]:\\n        seen = {{}}\\n        for i, num in enumerate(nums):\\n            need = target - num\\n            if need in seen:\\n                return [seen[need], i]\\n            seen[num] = i\\n        return []",
  "time_complexity": "O(n)，其中 n 是数组 nums 的长度，需要遍历数组一次。",
  "space_complexity": "O(n)，最坏情况下需要将 n 个元素存入哈希表。",
  "edge_cases": [
    {{"input": "nums = [2,7,11,15], target = 9", "output": "[0,1]", "explanation": "nums[0] + nums[1] = 2 + 7 = 9"}},
    {{"input": "nums = [3,2,4], target = 6", "output": "[1,2]", "explanation": "nums[1] + nums[2] = 2 + 4 = 6"}},
    {{"input": "nums = [3,3], target = 6", "output": "[0,1]", "explanation": "nums[0] + nums[1] = 3 + 3 = 6，注意数组中可能有重复元素"}}
  ],
  "common_mistakes": ["先把当前元素放入哈希表再查找，可能导致同一个元素被使用两次。", "没有考虑数组中存在重复元素的情况。"],
  "optimization": "暴力解法需要 O(n²) 时间复杂度，使用哈希表可以将时间复杂度优化到 O(n)，空间复杂度为 O(n)。",
  "rag_context": "检索到数组与哈希表模板。"
}}

## 用户输入
{{
  "problem": "{request.problem}",
  "language": "{request.language}",
  "difficulty": "{request.difficulty}"
}}

## 输出要求（必须严格遵守）
1. 只输出 JSON，不要输出任何其他文字，不要输出 Markdown 代码块。
2. 必须包含以下字段：problem_type, core_idea, data_structure, step_by_step_solution, reference_code, time_complexity, space_complexity, edge_cases, common_mistakes, optimization, rag_context。
3. problem_type 是字符串数组，如 ["数组", "哈希表"]。
4. step_by_step_solution 是字符串数组，包含解题步骤。
5. edge_cases 至少包含 3 个对象，每个对象包含 input, output, explanation 字段。
6. common_mistakes 是字符串数组。
7. reference_code 包含完整可运行的代码，使用用户指定的语言。
8. time_complexity 和 space_complexity 说明复杂度并解释变量含义。
9. rag_context 说明 RAG 检索结果摘要。
10. 如果题目描述不完整，在 core_idea 中说明并基于常见假设给出解法。

开始输出："""

    try:
        client = OpenAI(
            api_key=api_key,
            base_url=api_base
        )
        
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"请生成算法题解，只输出 JSON。"}
            ],
            temperature=0.3,
            max_tokens=3000,
            response_format={"type": "json_object"}
        )
        
        result_text = response.choices[0].message.content.strip()
        
        # 解析 JSON
        result = parse_json_safely(result_text)
        
        if result is None:
            return {
                "problem_type": ["解析错误"],
                "core_idea": f"无法解析模型输出: {result_text[:200]}...",
                "data_structure": "",
                "step_by_step_solution": [],
                "reference_code": result_text,
                "time_complexity": "",
                "space_complexity": "",
                "edge_cases": [],
                "common_mistakes": [],
                "optimization": "",
                "rag_context": context_str[:200] if context_str else ""
            }
        
        # 确保所有字段存在
        required_fields = [
            "problem_type", "core_idea", "data_structure",
            "step_by_step_solution", "reference_code",
            "time_complexity", "space_complexity",
            "edge_cases", "common_mistakes", "optimization", "rag_context"
        ]
        
        for field in required_fields:
            if field not in result or result[field] is None:
                result[field] = "" if field != "problem_type" and field != "step_by_step_solution" and field != "edge_cases" and field != "common_mistakes" else []
        
        # 确保 edge_cases 格式正确
        if not isinstance(result["edge_cases"], list):
            result["edge_cases"] = []
        else:
            valid_cases = []
            for case in result["edge_cases"]:
                if isinstance(case, dict) and "input" in case and "output" in case:
                    valid_cases.append({
                        "input": case.get("input", ""),
                        "output": case.get("output", ""),
                        "explanation": case.get("explanation", "")
                    })
            result["edge_cases"] = valid_cases
        
        # 设置 rag_context
        if not result["rag_context"] or result["rag_context"] == "":
            result["rag_context"] = context_str[:200] if context_str else "未检索到强相关模板"
        
        return result
        
    except AuthenticationError:
        return {
            "problem_type": [],
            "core_idea": "错误：API Key 无效或已过期",
            "data_structure": "",
            "step_by_step_solution": [],
            "reference_code": "",
            "time_complexity": "",
            "space_complexity": "",
            "edge_cases": [],
            "common_mistakes": [],
            "optimization": "",
            "rag_context": context_str[:200] if context_str else ""
        }
    except RateLimitError:
        return {
            "problem_type": [],
            "core_idea": "错误：API 请求超限，请稍后重试",
            "data_structure": "",
            "step_by_step_solution": [],
            "reference_code": "",
            "time_complexity": "",
            "space_complexity": "",
            "edge_cases": [],
            "common_mistakes": [],
            "optimization": "",
            "rag_context": context_str[:200] if context_str else ""
        }
    except APIError as e:
        return {
            "problem_type": [],
            "core_idea": f"错误：API 服务异常: {str(e)}",
            "data_structure": "",
            "step_by_step_solution": [],
            "reference_code": "",
            "time_complexity": "",
            "space_complexity": "",
            "edge_cases": [],
            "common_mistakes": [],
            "optimization": "",
            "rag_context": context_str[:200] if context_str else ""
        }
    except Exception as e:
        return {
            "problem_type": [],
            "core_idea": f"错误：{str(e)}",
            "data_structure": "",
            "step_by_step_solution": [],
            "reference_code": "",
            "time_complexity": "",
            "space_complexity": "",
            "edge_cases": [],
            "common_mistakes": [],
            "optimization": "",
            "rag_context": context_str[:200] if context_str else ""
        }

# 初始化加载知识库
load_algorithm_knowledge_base()
