from github import Github
import requests


# 初始化GitHub客户端
def get_github_client(token: str = None):
    """创建GitHub API客户端"""
    from dotenv import load_dotenv
    import os

    load_dotenv()  # 从.env文件加载变量
    token = os.getenv("GITHUB_TOKEN")
    if token:
        return Github(token)
    else:  # 匿名访问（有频率限制）
        return Github()


def query_llm(prompt: str) -> str:
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "deepseek-r1:32b", "prompt": prompt, "stream": False},
    )
    response_json = response.json()
    return response_json.get("response", "")
