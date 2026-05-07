# -*- coding: utf-8 -*-
"""
FastAPI 主应用 - AI 测试用例生成 API (Vercel 简化版)

功能说明：
1. 提供 RESTful API 接口，供前端调用生成测试用例
2. 直接调用 LLM 生成测试用例（暂不使用 RAG）
3. 支持跨域请求（CORS）
4. 环境变量配置，不硬编码敏感信息

环境变量：
- OPENAI_API_KEY: API 密钥
- OPENAI_BASE_URL: API 基础 URL（可选）
- MODEL_NAME: 模型名称（可选，默认 deepseek-v3）
"""

import os
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from openai import OpenAI

# 创建 FastAPI 应用
app = FastAPI(
    title="AI-Driven SDET Toolkit API",
    description="基于 RAG 与 Agent 的自动化测试助手 API",
    version="1.0.0"
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


# 初始化 OpenAI 客户端
def get_client():
    return OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.qnaigc.com/v1")
    )


@app.get("/")
async def root():
    return {
        "name": "AI-Driven SDET Toolkit API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.post("/api/generate_cases")
async def generate_cases(request: GenerateCasesRequest):
    try:
        system_prompt = """你是一个专业的软件测试工程师。请根据用户的需求生成标准化的测试用例。

要求：
1. 生成 5 个测试用例
2. 每个用例包含：case_id, test_point, precondition, steps, expected_result, priority, test_type
3. 返回纯 JSON 格式，不要包含其他文字
4. priority 使用 P0/P1/P2
5. test_type 使用：功能测试/异常测试/边界测试/性能测试/安全测试

JSON 格式示例：
{
  "test_cases": [
    {
      "case_id": "TC001",
      "test_point": "正常登录",
      "precondition": "用户已注册",
      "steps": ["打开登录页面", "输入用户名", "输入密码", "点击登录"],
      "expected_result": "登录成功，跳转到首页",
      "priority": "P0",
      "test_type": "功能测试"
    }
  ]
}"""

        user_prompt = f"请为以下需求生成测试用例：{request.requirement}"

        client = get_client()
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
            "total_cases": len(test_cases)
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"生成测试用例失败: {str(e)}"
        )
