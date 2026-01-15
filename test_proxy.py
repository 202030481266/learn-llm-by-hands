import litellm
import os

# 2. 设置 Base URL
# 注意：大多数中转站要求末尾必须加 /v1
# 如果你的报错是 "Just a moment"，极大概率是因为你漏了 /v1，导致请求打到了网页首页
# 开启调试模式，这会打印出具体的 Request URL，非常重要
litellm.set_verbose = True

try:
    # 3. 关键点：在模型前加 "openai/" 前缀
    # 这告诉 litellm："虽然我要调 Claude，但请用 OpenAI 的格式（JSON schema）发请求"
    # 这样 litellm 就会把请求发到 .../v1/chat/completions 而不是 Anthropic 的原生路径
    response = litellm.completion(
        model="anthropic/claude-sonnet-4-5-20250929", 
        messages=[
            {"role": "user", "content": "如果你能看到这句话，请回复'连接成功'并告诉我你是谁。"}
        ],
        max_tokens=50
    )
    
    print("-" * 30)
    print("✅ 测试成功！")
    print("回复内容:", response.choices[0].message.content)

except litellm.APIConnectionError as e:
    print("\n❌ 连接错误 (可能是 URL 不对或被拦截):")
    print(e)
    # 如果错误里包含 HTML，说明 URL 填错了，打到了网页界面
    if "Just a moment" in str(e) or "<!DOCTYPE html>" in str(e):
        print("\n👉 分析: 你依然收到了 Cloudflare 的 HTML 页面。")
        print("   请检查 api_base 是否漏掉了 '/v1'，或者该中转站是否开启了高强度防御。")

except litellm.APIError as e:
    print("\n❌ API 业务错误 (连上了，但参数或余额有问题):")
    print(e)

except Exception as e:
    print("\n❌ 其他未知错误:")
    print(e)