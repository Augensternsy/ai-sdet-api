from http.server import BaseHTTPRequestHandler
import json
import os
from openai import OpenAI

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/' or self.path == '/health':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            response = {
                "name": "AI-Driven SDET Toolkit API",
                "version": "1.0.0",
                "status": "running"
            }
            self.wfile.write(json.dumps(response).encode())
        else:
            self.send_response(404)
            self.end_headers()
    
    def do_POST(self):
        if self.path == '/api/generate_cases':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            request_body = json.loads(post_data.decode('utf-8'))
            requirement = request_body.get('requirement', '')
            
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

                user_prompt = f"请为以下需求生成测试用例：{requirement}"

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

                result = {
                    "success": True,
                    "message": "测试用例生成成功",
                    "data": test_cases,
                    "total_cases": len(test_cases)
                }

                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps(result).encode())

            except Exception as e:
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                error_response = {
                    "success": False,
                    "message": f"生成测试用例失败: {str(e)}"
                }
                self.wfile.write(json.dumps(error_response).encode())
        else:
            self.send_response(404)
            self.end_headers()
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
