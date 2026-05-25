# -*- coding: utf-8 -*-
"""
FastAPI 主应用 - AI 测试用例生成 API (纯内存 RAG 版)

使用 FastAPI + LangChain + 纯内存向量检索部署到 Vercel
"""

import os
import json
import math
import glob
from typing import List, Dict, Tuple
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from openai import OpenAI
from mangum import Mangum
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

# 加载 .env 文件
load_dotenv()


# 创建 FastAPI 应用
app = FastAPI(
    title="AI-Driven SDET Toolkit API",
    description="基于纯内存 RAG 的自动化测试助手 API",
    version="2.1.0"
)

# 配置 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 请求体模型
class GenerateCasesRequest(BaseModel):
    requirement: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=3, ge=1, le=10)


# 全局变量：内存向量存储
documents: List[Dict] = []
embeddings_model = None


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    计算余弦相似度（纯 Python 实现）
    
    参数:
        vec1: 向量1
        vec2: 向量2
    
    返回:
        similarity: 余弦相似度 [-1, 1]
    """
    if len(vec1) != len(vec2):
        raise ValueError("向量长度不一致")
    
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)


def load_documents_to_memory():
    """
    从 docs 目录加载文档并向量化到内存
    
    返回:
        success: 是否成功
        count: 加载的文档块数量
    """
    global documents, embeddings_model
    
    docs_dir = os.getenv("DOCS_DIR", "./docs")
    
    try:
        if embeddings_model is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                print("警告: OPENAI_API_KEY 未设置，无法加载向量")
                return False, 0
            
            base_url = os.getenv("OPENAI_BASE_URL", "https://api.qnaigc.com/v1")
            model_name = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
            
            embeddings_model = OpenAIEmbeddings(
                openai_api_key=api_key,
                openai_api_base=base_url,
                model=model_name
            )
        
        # 加载文档
        doc_files = []
        file_patterns = [
            os.path.join(docs_dir, "**/*.md"),
            os.path.join(docs_dir, "**/*.txt")
        ]
        
        for pattern in file_patterns:
            doc_files.extend(glob.glob(pattern, recursive=True))
        
        if not doc_files:
            print(f"警告: 在目录 {docs_dir} 中未找到文档")
            return False, 0
        
        print(f"找到 {len(doc_files)} 个文档文件")
        
        all_docs = []
        for file_path in doc_files:
            try:
                loader = TextLoader(file_path, encoding="utf-8")
                docs = loader.load()
                all_docs.extend(docs)
                print(f"  已加载: {file_path}")
            except Exception as e:
                print(f"  加载失败 {file_path}: {str(e)}")
        
        if not all_docs:
            print("未加载到任何文档内容")
            return False, 0
        
        # 文档切片
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
        )
        
        chunks = splitter.split_documents(all_docs)
        print(f"文档切片完成: {len(all_docs)} 个文档 -> {len(chunks)} 个块")
        
        # 向量化
        print("正在向量化文档块...")
        texts = [chunk.page_content for chunk in chunks]
        vectors = embeddings_model.embed_documents(texts)
        
        # 存储到内存
        documents = []
        for idx, (text, vector) in enumerate(zip(texts, vectors)):
            documents.append({
                "id": idx,
                "content": text,
                "vector": vector
            })
        
        print(f"成功向量化并加载 {len(documents)} 个文档块到内存")
        return True, len(documents)
    
    except Exception as e:
        print(f"加载文档到内存失败: {str(e)}")
        return False, 0


def search_memory(query: str, top_k: int = 3) -> List[str]:
    """
    在内存中检索相关文档
    
    参数:
        query: 查询文本
        top_k: 返回 top K 个最相关文档
    
    返回:
        results: 相关文档内容列表
    """
    global documents, embeddings_model
    
    if not documents or embeddings_model is None:
        return []
    
    try:
        query_vector = embeddings_model.embed_query(query)
        
        scores = []
        for doc in documents:
            sim = cosine_similarity(query_vector, doc["vector"])
            scores.append((sim, doc["content"]))
        
        scores.sort(key=lambda x: x[0], reverse=True)
        top_docs = scores[:top_k]
        
        return [content for (sim, content) in top_docs]
    
    except Exception as e:
        print(f"检索失败: {str(e)}")
        return []


@app.on_event("startup")
async def startup_event():
    """应用启动时加载文档到内存"""
    print("=" * 60)
    print("正在加载 RAG 知识库到内存...")
    print("=" * 60)
    success, count = load_documents_to_memory()
    if success:
        print(f"知识库加载成功！共 {count} 个文档块")
    else:
        print("知识库加载失败，将使用无 RAG 模式")


@app.get("/")
async def root():
    return {
        "name": "AI-Driven SDET Toolkit API",
        "version": "2.1.0",
        "status": "running",
        "rag_enabled": len(documents) > 0,
        "document_count": len(documents)
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "vectorstore_loaded": len(documents) > 0,
        "document_count": len(documents)
    }


@app.post("/api/generate_cases")
async def generate_cases(request: GenerateCasesRequest):
    try:
        # 1. 检索相关测试规范
        retrieved_docs = search_memory(request.requirement, top_k=request.top_k)
        
        # 2. 构建 RAG 增强的 System Prompt
        context_str = "\n\n".join(retrieved_docs) if retrieved_docs else ""
        
        system_prompt = f"""你是一个专业的软件测试工程师。请根据用户的需求和提供的测试规范，生成标准化的测试用例。

## 测试规范参考
{context_str if context_str else "（无相关规范，请根据通用测试标准生成）"}

## Few-shot 示例（参考此格式生成）
**示例需求**：用户登录功能
**示例输出**：
{{
  "test_cases": [
    {{
      "case_id": "TC001",
      "test_point": "正常登录-用户名密码正确",
      "precondition": "用户已注册，账号状态正常",
      "steps": ["1. 打开登录页面", "2. 输入正确的用户名", "3. 输入正确的密码", "4. 点击登录按钮"],
      "expected_result": "登录成功，页面跳转到首页",
      "priority": "P0",
      "test_type": "功能测试"
    }},
    {{
      "case_id": "TC002",
      "test_point": "异常登录-密码错误",
      "precondition": "用户已注册",
      "steps": ["1. 打开登录页面", "2. 输入正确的用户名", "3. 输入错误的密码", "4. 点击登录按钮"],
      "expected_result": "登录失败，提示'密码错误'",
      "priority": "P0",
      "test_type": "异常测试"
    }},
    {{
      "case_id": "TC003",
      "test_point": "边界测试-空用户名",
      "precondition": "无",
      "steps": ["1. 打开登录页面", "2. 用户名输入框留空", "3. 输入任意密码", "4. 点击登录按钮"],
      "expected_result": "登录失败，提示'请输入用户名'",
      "priority": "P1",
      "test_type": "边界测试"
    }}
  ]
}}

## 生成要求
1. 生成 5 个测试用例，覆盖功能测试、异常测试、边界测试
2. 每个用例包含：case_id, test_point, precondition, steps, expected_result, priority, test_type
3. 返回纯 JSON 格式，不要包含其他文字
4. priority 使用 P0/P1/P2
5. test_type 使用：功能测试/异常测试/边界测试/性能测试/安全测试"""

        user_prompt = f"请为以下需求生成测试用例：{request.requirement}"

        client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.qnaigc.com/v1")
        )
        model_name = os.getenv("MODEL_NAME", "deepseek-v3")

        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=2000
        )

        response_text = response.choices[0].message.content

        # 解析 JSON
        json_start = response_text.find("{")
        json_end = response_text.rfind("}") + 1
        if json_start != -1 and json_end != -1:
            json_str = response_text[json_start:json_end]
            json_data = json.loads(json_str)
        else:
            json_data = json.loads(response_text)

        test_cases = json_data.get("test_cases", [])

        return {
            "success": True,
            "message": "测试用例生成成功",
            "data": test_cases,
            "total_cases": len(test_cases),
            "rag_enabled": len(documents) > 0,
            "retrieved_docs_count": len(retrieved_docs)
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"生成测试用例失败: {str(e)}"
        )


# Mangum handler for Vercel
handler = Mangum(app)
