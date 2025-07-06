import os
import logging
from typing import List, Dict, Any, Optional
from openai import OpenAI
import json

class LLMClient:
    """统一的大模型客户端接口"""
    
    def __init__(self, model_name: str = "deepseek-chat", api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        
        # 根据模型名称选择不同的客户端
        if "deepseek" in model_name.lower():
            self.client = OpenAI(
                api_key=self.api_key,
                base_url="https://api.deepseek.com"
            )
        elif "gpt" in model_name.lower():
            self.client = OpenAI(api_key=self.api_key)
        # 可以添加其他接口
        else:
            # 默认使用OpenAI格式
            self.client = OpenAI(api_key=self.api_key)
    
    def generate_answer(self, 
                       prompt: str, 
                       temperature: float = 0.1, 
                       max_tokens: int = 500,
                       json_response = False) -> str:
        """生成答案
        :param prompt: 输入提示
        :param temperature: 生成的随机性，0.0-1.0之间
        :param max_tokens: 最大生成token数
        :param json_response: 是否返回JSON格式的响应(仅限deepseek-chat)
        """
        
        if json_response:
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format={
                        'type': 'json_object'
                    }
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                logging.error(f"LLM生成失败: {e}")
                return f"生成失败: {str(e)}"
        else:
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                logging.error(f"LLM生成失败: {e}")
                return f"生成失败: {str(e)}"
    
    def batch_generate(self, 
                      prompts: List[str], 
                      temperature: float = 0.1) -> List[str]:
        """批量生成答案"""
        results = []
        for prompt in prompts:
            answer = self.generate_answer(prompt, temperature)
            results.append(answer)
        return results