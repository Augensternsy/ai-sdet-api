# -*- coding: utf-8 -*-
"""
FastAPI 主应用 - AI 测试用例生成 API

功能说明：
1. 提供 RESTful API 接口，供前端调用生成测试用例
2. 基于已有的 test_case_generator.py 核心逻辑
3. 支持跨域请求（CORS）
4. 环境变量配置，不硬编码敏感信息

启动方式：
uvicorn main:app --host 0.0.0.0 --port 8001

环境变量：
- OPENAI_API_KEY: API 密钥
- OPENAI_BASE_URL: API 基础 URL（可选）
- MODEL_NAME: 模型名称（可选，默认 deepseek-v3）
"""

import os
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# 导入核心生成逻辑
from test_case_generator import (
    load_vectorstore,
    retrieve_context,
    build_system_prompt,
    call_llm_api,
    parse_and_validate_json
)


# 创建 FastAPI 应用
app = FastAPI(
    title="AI-Driven SDET Toolkit API",
    description="基于 RAG 与 Agent 的自动化测试助手 API",
    version="1.0.0"
)

# 配置 CORS 中间件，允许所有跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源（生产环境应限制具体域名）
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有 HTTP 方法
    allow_headers=["*"],  # 允许所有 HTTP 头
)


# 请求体模型
class GenerateCasesRequest(BaseModel):
    """生成测试用例的请求参数"""
    requirement: str = Field(
        ...,
        description="用户需求文本，例如：'登录接口，包含用户名和密码'",
        min_length=1,
        max_length=2000,
        examples=["登录接口，包含用户名和密码"]
    )
    top_k: int = Field(
        default=3,
        description="检索的规范上下文数量（默认 3）",
        ge=1,
        le=10
    )


# 全局向量数据库实例（懒加载）
_vectorstore = None


def get_vectorstore():
    """获取或初始化向量数据库"""
    global _vectorstore
    if _vectorstore is None:
        try:
            _vectorstore = load_vectorstore()
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"向量数据库加载失败: {str(e)}"
            )
    return _vectorstore


@app.get("/")
async def root():
    """根路径，返回 API 信息"""
    return {
        "name": "AI-Driven SDET Toolkit API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {
        "status": "healthy",
        "vectorstore_loaded": _vectorstore is not None
    }


@app.post("/api/generate_cases")
async def generate_cases(request: GenerateCasesRequest):
    """
    生成测试用例接口

    接收用户需求，结合 RAG 向量库检索的规范上下文，
    调用 LLM 生成标准 JSON 格式的测试用例。

    - **requirement**: 用户需求文本
    - **top_k**: 检索的规范上下文数量（可选，默认 3）
    """
    try:
        print(f"收到请求: requirement={request.requirement}, top_k={request.top_k}")
        
        # 步骤 1: 获取向量数据库
        print("步骤 1: 获取向量数据库...")
        vectorstore = get_vectorstore()
        print("步骤 1 完成")

        # 步骤 2: 检索相关上下文
        print("步骤 2: 检索相关上下文...")
        context, _ = retrieve_context(vectorstore, request.requirement, request.top_k)
        print("步骤 2 完成")

        # 步骤 3: 构建 Prompt
        print("步骤 3: 构建 Prompt...")
        system_prompt, user_prompt = build_system_prompt(context, request.requirement)
        print("步骤 3 完成")

        # 步骤 4: 调用 LLM
        print("步骤 4: 调用 LLM...")
        response_text = call_llm_api(system_prompt, user_prompt)
        print("步骤 4 完成")

        # 步骤 5: 解析 JSON
        print("步骤 5: 解析 JSON...")
        json_data = parse_and_validate_json(response_text)
        print("步骤 5 完成")

        if json_data is None:
            raise HTTPException(
                status_code=500,
                detail="JSON 解析失败，请重试"
            )

        # 步骤 6: 构建响应
        test_cases = json_data.get("test_cases", [])

        return {
            "success": True,
            "message": "测试用例生成成功",
            "data": test_cases,
            "total_cases": len(test_cases)
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"Exception: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"生成测试用例失败: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8002,
        reload=False
    )
