# -*- coding: utf-8 -*-
"""
测试用例自动生成器

功能说明：
1. 从 Chroma 向量库中检索最相关的测试规范上下文
2. 使用结构化 Prompt + Few-Shot 提示词引导 LLM
3. 调用 OpenAI/DeepSeek API 生成标准 JSON 格式的测试用例
4. 输出包含：用例编号、测试点、前置条件、操作步骤、预期结果

环境变量配置：
- OPENAI_API_KEY: OpenAI API 密钥
- OPENAI_BASE_URL: API 基础 URL（可选，用于兼容 DeepSeek 等）
- MODEL_NAME: 模型名称（默认：gpt-3.5-turbo）

依赖安装：
pip install openai langchain langchain-community chromadb
"""

import os
import json
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from openai import OpenAI


def load_vectorstore(persist_directory="./chroma_db"):
    """
    加载已保存的 Chroma 向量数据库
    
    参数:
        persist_directory: 向量数据库持久化目录
    
    返回:
        vectorstore: Chroma 向量数据库实例
    """
    print("正在加载向量数据库...")
    
    # 创建 Embedding 模型（与构建时使用的模型保持一致）
    embeddings = HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-base-zh-v1.5",
        model_kwargs={'device': 'cpu'}
    )
    
    # 加载已保存的向量数据库
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
    
    print(f"  ✓ 向量数据库加载成功: {persist_directory}")
    return vectorstore


def retrieve_context(vectorstore, query, top_k=3):
    """
    从向量库中检索最相关的测试规范上下文
    
    参数:
        vectorstore: Chroma 向量数据库实例
        query: 用户输入的查询文本
        top_k: 返回最相关的 Top-K 条结果
    
    返回:
        context_text: 拼接后的上下文文本
        results: 原始检索结果列表
    """
    print(f"\n正在检索相关规范上下文...")
    print(f"  查询: {query}")
    print(f"  检索数量: Top-{top_k}")
    
    # 执行相似度检索
    results = vectorstore.similarity_search(query, k=top_k)
    
    # 拼接检索结果为上下文文本
    context_parts = []
    for i, result in enumerate(results, 1):
        source_file = result.metadata.get('source_file', '未知')
        content = result.page_content
        context_parts.append(f"[规范片段 {i}] (来源: {source_file})\n{content}")
    
    context_text = "\n\n".join(context_parts)
    
    print(f"  ✓ 检索到 {len(results)} 条相关规范")
    return context_text, results


def build_system_prompt(context, user_requirement):
    """
    构建结构化的 System Prompt，包含 Few-Shot 示例
    
    参数:
        context: 从向量库检索到的测试规范上下文
        user_requirement: 用户的测试需求描述
    
    返回:
        system_prompt: 完整的系统提示词
        user_prompt: 用户提示词
    """
    
    # System Prompt：定义角色、任务、约束和 Few-Shot 示例
    system_prompt = f"""你是一个资深测试开发专家，拥有 10 年以上的软件测试经验。

你的任务是根据用户提供的测试需求，结合相关的测试规范，设计高质量的标准测试用例。

## 测试用例要求：
1. 运用等价类划分法和边界值分析法设计测试用例
2. 覆盖正常流程和异常流程
3. 测试用例应该具有可执行性和可验证性
4. 预期结果应该明确且可验证

## 输出格式约束：
你必须且只能输出合法的 JSON 格式，不要包含任何其他文字说明。
JSON 格式如下：
{{
  "test_cases": [
    {{
      "case_id": "TC001",
      "test_point": "测试点描述",
      "precondition": "前置条件",
      "steps": ["步骤1", "步骤2", "步骤3"],
      "expected_result": "预期结果",
      "priority": "P0/P1/P2",
      "test_type": "功能测试/边界测试/异常测试"
    }}
  ]
}}

## 参考的测试规范上下文：
{context}
"""
    
    # Few-Shot 示例：提供 2 个完整的测试用例示例
    few_shot_examples = """
## 示例 1：
用户输入：用户登录功能，包含用户名和密码

输出：
{{
  "test_cases": [
    {{
      "case_id": "TC001",
      "test_point": "正常登录 - 正确的用户名和密码",
      "precondition": "用户已注册，账号存在且状态正常",
      "steps": [
        "打开登录页面",
        "输入正确的用户名",
        "输入正确的密码",
        "点击登录按钮"
      ],
      "expected_result": "登录成功，跳转到用户首页，显示欢迎信息",
      "priority": "P0",
      "test_type": "功能测试"
    }},
    {{
      "case_id": "TC002",
      "test_point": "边界值 - 用户名长度为最小值（1个字符）",
      "precondition": "系统中存在用户名为单个字符的用户",
      "steps": [
        "打开登录页面",
        "输入1个字符的用户名",
        "输入正确的密码",
        "点击登录按钮"
      ],
      "expected_result": "登录成功，系统正常处理最短用户名",
      "priority": "P1",
      "test_type": "边界测试"
    }},
    {{
      "case_id": "TC003",
      "test_point": "异常测试 - 密码错误",
      "precondition": "用户已注册，账号存在",
      "steps": [
        "打开登录页面",
        "输入正确的用户名",
        "输入错误的密码",
        "点击登录按钮"
      ],
      "expected_result": "登录失败，显示'用户名或密码错误'提示，不跳转到首页",
      "priority": "P0",
      "test_type": "异常测试"
    }},
    {{
      "case_id": "TC004",
      "test_point": "边界值 - 密码长度超过最大值",
      "precondition": "无",
      "steps": [
        "打开登录页面",
        "输入正确的用户名",
        "输入超过最大长度限制的密码（如129个字符）",
        "点击登录按钮"
      ],
      "expected_result": "系统应拒绝超长密码，显示'密码长度不能超过128个字符'提示",
      "priority": "P1",
      "test_type": "边界测试"
    }},
    {{
      "case_id": "TC005",
      "test_point": "异常测试 - 用户名为空",
      "precondition": "无",
      "steps": [
        "打开登录页面",
        "不输入用户名",
        "输入密码",
        "点击登录按钮"
      ],
      "expected_result": "登录失败，显示'用户名不能为空'提示，登录按钮可继续点击",
      "priority": "P1",
      "test_type": "异常测试"
    }}
  ]
}}

## 示例 2：
用户输入：文件上传功能，支持 JPG 和 PNG 格式，最大 5MB

输出：
{{
  "test_cases": [
    {{
      "case_id": "TC001",
      "test_point": "正常上传 - 符合要求的JPG图片",
      "precondition": "用户已登录系统，进入文件上传页面",
      "steps": [
        "点击上传按钮",
        "选择一个2MB的JPG格式图片",
        "确认上传"
      ],
      "expected_result": "上传成功，显示上传进度条，完成后显示预览图",
      "priority": "P0",
      "test_type": "功能测试"
    }},
    {{
      "case_id": "TC002",
      "test_point": "边界值 - 文件大小恰好为5MB",
      "precondition": "用户已登录系统",
      "steps": [
        "点击上传按钮",
        "选择一个恰好5MB的PNG图片",
        "确认上传"
      ],
      "expected_result": "上传成功，5MB是允许的最大值，系统正常处理",
      "priority": "P1",
      "test_type": "边界测试"
    }},
    {{
      "case_id": "TC003",
      "test_point": "边界值 - 文件大小超过5MB（5.1MB）",
      "precondition": "用户已登录系统",
      "steps": [
        "点击上传按钮",
        "选择一个5.1MB的JPG图片",
        "确认上传"
      ],
      "expected_result": "上传失败，显示'文件大小不能超过5MB'提示",
      "priority": "P0",
      "test_type": "边界测试"
    }},
    {{
      "case_id": "TC004",
      "test_point": "异常测试 - 上传不支持的文件格式（PDF）",
      "precondition": "用户已登录系统",
      "steps": [
        "点击上传按钮",
        "选择一个PDF文件",
        "确认上传"
      ],
      "expected_result": "上传失败，显示'仅支持JPG和PNG格式'提示",
      "priority": "P0",
      "test_type": "异常测试"
    }},
    {{
      "case_id": "TC005",
      "test_point": "异常测试 - 上传空文件（0字节）",
      "precondition": "用户已登录系统",
      "steps": [
        "点击上传按钮",
        "选择一个0字节的JPG文件",
        "确认上传"
      ],
      "expected_result": "上传失败，显示'文件不能为空'提示",
      "priority": "P1",
      "test_type": "异常测试"
    }}
  ]
}}
"""
    
    # User Prompt：用户的具体需求
    user_prompt = f"""{few_shot_examples}

现在请为以下需求设计测试用例：

用户需求：{user_requirement}

请严格按照上述 JSON 格式输出测试用例，至少包含 5 个测试用例，覆盖正常流程、边界值和异常情况。
只输出 JSON，不要包含任何其他说明文字。"""
    
    return system_prompt, user_prompt


def call_llm_api(system_prompt, user_prompt):
    """
    调用 OpenAI/DeepSeek API 生成测试用例
    
    参数:
        system_prompt: 系统提示词
        user_prompt: 用户提示词
    
    返回:
        response_text: LLM 返回的文本
    """
    print("\n正在调用 LLM 生成测试用例...")
    
    # 从环境变量读取配置
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model_name = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
    
    if not api_key:
        raise ValueError("未设置 OPENAI_API_KEY 环境变量")
    
    print(f"  API 地址: {base_url}")
    print(f"  模型: {model_name}")
    
    # 创建 OpenAI 客户端
    client = OpenAI(
        api_key=api_key,
        base_url=base_url
    )
    
    # 调用 API
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.7,
        max_tokens=4000
    )
    
    response_text = response.choices[0].message.content
    print(f"  ✓ LLM 响应接收完成")
    
    return response_text


def parse_and_validate_json(response_text):
    """
    解析并验证 LLM 返回的 JSON
    
    参数:
        response_text: LLM 返回的文本
    
    返回:
        json_data: 解析后的 JSON 数据
    """
    print("\n正在解析 JSON...")
    
    # 尝试提取 JSON（处理可能包含 markdown 代码块的情况）
    json_str = response_text.strip()
    
    # 如果包含 ```json 标记，提取其中的内容
    if "```json" in json_str:
        json_str = json_str.split("```json")[1].split("```")[0].strip()
    elif "```" in json_str:
        json_str = json_str.split("```")[1].split("```")[0].strip()
    
    # 解析 JSON
    try:
        json_data = json.loads(json_str)
        print(f"  ✓ JSON 解析成功")
        
        # 验证 JSON 结构
        if "test_cases" not in json_data:
            print("  ⚠ 警告: JSON 缺少 'test_cases' 字段")
        
        test_cases = json_data.get("test_cases", [])
        print(f"  ✓ 包含 {len(test_cases)} 个测试用例")
        
        # 验证每个测试用例的必需字段
        required_fields = ["case_id", "test_point", "precondition", "steps", "expected_result"]
        for i, case in enumerate(test_cases):
            missing_fields = [field for field in required_fields if field not in case]
            if missing_fields:
                print(f"  ⚠ 警告: 用例 {case.get('case_id', i+1)} 缺少字段: {missing_fields}")
        
        return json_data
        
    except json.JSONDecodeError as e:
        print(f"  ✗ JSON 解析失败: {str(e)}")
        print(f"\n原始响应:\n{response_text}")
        return None


def save_test_cases(json_data, output_file=None):
    """
    保存测试用例到文件
    
    参数:
        json_data: 测试用例 JSON 数据
        output_file: 输出文件路径
    """
    if output_file is None:
        # 自动生成文件名
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"test_cases_{timestamp}.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n  ✓ 测试用例已保存到: {output_file}")
    return output_file


def generate_test_cases(user_requirement, vectorstore=None, top_k=3, save_to_file=True):
    """
    主函数：生成测试用例的完整流程
    
    参数:
        user_requirement: 用户的测试需求描述
        vectorstore: Chroma 向量数据库实例（如果为 None 则自动加载）
        top_k: 检索的上下文数量
        save_to_file: 是否保存到文件
    
    返回:
        json_data: 生成的测试用例 JSON 数据
    """
    print("=" * 60)
    print("测试用例自动生成器")
    print("=" * 60)
    
    # 步骤 1: 加载向量数据库
    if vectorstore is None:
        vectorstore = load_vectorstore()
    
    # 步骤 2: 检索相关上下文
    context, _ = retrieve_context(vectorstore, user_requirement, top_k)
    
    # 步骤 3: 构建 Prompt
    print("\n正在构建结构化 Prompt...")
    system_prompt, user_prompt = build_system_prompt(context, user_requirement)
    print("  ✓ Prompt 构建完成")
    
    # 步骤 4: 调用 LLM
    response_text = call_llm_api(system_prompt, user_prompt)
    
    # 步骤 5: 解析 JSON
    json_data = parse_and_validate_json(response_text)
    
    if json_data is None:
        print("\n错误: JSON 解析失败")
        return None
    
    # 步骤 6: 保存到文件
    output_file = None
    if save_to_file:
        output_file = save_test_cases(json_data)
    
    # 打印结果摘要
    print("\n" + "=" * 60)
    print("测试用例生成完成")
    print("=" * 60)
    
    test_cases = json_data.get("test_cases", [])
    print(f"\n生成的测试用例摘要:")
    for case in test_cases:
        print(f"  - {case.get('case_id', '?')}: {case.get('test_point', '?')}")
        print(f"    优先级: {case.get('priority', '?')} | 类型: {case.get('test_type', '?')}")
    
    if output_file:
        print(f"\n详细结果请查看: {output_file}")
    
    return json_data


def main():
    """
    主函数：演示测试用例生成器的使用
    """
    
    # 测试用例示例
    test_requirements = [
        "登录接口，包含用户名和密码",
        "文件上传功能，支持 JPG 和 PNG 格式，最大 5MB",
        "用户注册功能，需要邮箱验证，密码要求至少8位包含大小写字母和数字"
    ]
    
    # 生成第一个测试用例
    requirement = test_requirements[0]
    print(f"\n需求: {requirement}\n")
    
    try:
        result = generate_test_cases(requirement)
        
        if result:
            print("\n\n完整 JSON 输出:")
            print(json.dumps(result, ensure_ascii=False, indent=2))
            
    except Exception as e:
        print(f"\n错误: {str(e)}")
        print("\n请确保已设置环境变量:")
        print("  - OPENAI_API_KEY: 你的 API 密钥")
        print("  - OPENAI_BASE_URL: API 地址（可选，默认 OpenAI）")
        print("  - MODEL_NAME: 模型名称（可选，默认 gpt-3.5-turbo）")
        print("\n示例（使用 DeepSeek）:")
        print("  set OPENAI_API_KEY=your-api-key")
        print("  set OPENAI_BASE_URL=https://api.deepseek.com/v1")
        print("  set MODEL_NAME=deepseek-chat")


if __name__ == "__main__":
    main()
