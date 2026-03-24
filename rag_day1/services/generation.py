from openai import OpenAI
from core.config import API_KEY, BASE_URL
import os
import logging
CHAT_MODEL = os.getenv("ZHIPU_CHAT_MODEL", "glm-4-flash")

client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL
)

def build_prompt(query: str, context_docs: list[str]):
    context_text = "\n\n".join(
        [f"[资料{i+1}] {doc}" for i, doc in enumerate(context_docs)]
    )

    system_prompt = (
        "你是一个严谨的问答助手。"
        "你必须优先根据提供的资料回答。"
        "如果资料不足以回答问题，就明确说：根据现有资料无法确定。"
        "不要编造，不要补充资料中没有的信息。"
    )

    user_prompt = f"""
下面是检索到的资料：

{context_text}

用户问题：
{query}

请你遵守以下要求回答：
1. 只根据上面的资料回答
2. 回答简洁清楚
3. 如果资料不足，请直接说“根据现有资料无法确定”
"""

    return system_prompt, user_prompt


def generate_answer(query: str, context_docs: list[str]):
    logging.info("开始构建 prompt")
    system_prompt, user_prompt = build_prompt(query, context_docs)

    logging.info("开始调用 chat model")
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.2
    )

    logging.info("chat model 调用完成")
    return response.choices[0].message.content