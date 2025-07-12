import os
import logging
from typing import List, Dict, Any, Optional
from openai import OpenAI
import re

# 添加tiktoken支持
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    logging.warning("tiktoken未安装，将使用简单估算方法。建议安装: pip install tiktoken")


class LLMClient:
    """简化版大模型客户端接口，专注于DeepSeek优化"""
    
    # 更新的模型配置
    MODEL_CONFIGS = {
        # DeepSeek 系列
        "deepseek-chat": {
            "context_length": 65536,    # 64K上下文
            "max_output": 8192,             # 最大输出8K
            "default_output": 4096,         # 默认输出4K
            "encoding": "cl100k_base"
        },
        "deepseek-reasoner": {
            "context_length": 65536,    # 64K上下文
            "max_output": 65536,        # 最大输出64K
            "default_output": 32768,    # 默认输出32K
            "encoding": "cl100k_base"
        },
        
        # OpenAI GPT 系列（保持向后兼容）
        "gpt-3.5-turbo": {
            "context_length": 4096,
            "max_output": 1024,
            "default_output": 512,
            "encoding": "cl100k_base"
        },
        "gpt-4": {
            "context_length": 8192,
            "max_output": 2048,
            "default_output": 1024,
            "encoding": "cl100k_base"
        },
        "gpt-4o-mini": {
            "context_length": 128000,
            "max_output": 4096,
            "default_output": 2048,
            "encoding": "o200k_base"
        },
        "gpt-4o": {
            "context_length": 128000,
            "max_output": 4096,
            "default_output": 2048,
            "encoding": "o200k_base"
        },
    }
    
    def __init__(self, 
                 model_name: str = "deepseek-chat", 
                 api_key: Optional[str] = None,
                 max_context_ratio: float = 0.85):  # 为输出预留15%空间
        """
        初始化简化版LLM客户端
        
        Args:
            model_name: 模型名称
            api_key: API密钥
            max_context_ratio: 上下文占总长度的比例
        """
        self.model_name = model_name
        self.api_key = api_key or self._get_api_key(model_name)
        self.max_context_ratio = max_context_ratio
        
        # 获取模型配置
        self.model_config = self.MODEL_CONFIGS.get(model_name, {
            "context_length": 4096,
            "max_output": 1024,
            "default_output": 512,
            "encoding": "cl100k_base"
        })
        
        self.context_length = self.model_config["context_length"]
        self.max_output_tokens = self.model_config["max_output"]
        self.default_output_tokens = self.model_config["default_output"]
        self.max_context_tokens = int(self.context_length * max_context_ratio)
        
        self.logger = logging.getLogger(__name__)
        
        # 初始化tokenizer
        self.tokenizer = self._init_tokenizer()
        
        # 初始化客户端
        self.client = self._init_client()
        
        self.logger.info(f"🤖 初始化LLM客户端: {model_name}")
        self.logger.info(f"📏 上下文长度: {self.context_length:,}, 可用: {self.max_context_tokens:,}")
        self.logger.info(f"📤 最大输出: {self.max_output_tokens:,}, 默认: {self.default_output_tokens:,}")
        self.logger.info(f"🔧 Token计算: {'tiktoken' if self.tokenizer else '简单估算'}")
    
    def _get_api_key(self, model_name: str) -> Optional[str]:
        """获取API密钥"""
        if "deepseek" in model_name.lower():
            return os.getenv("DEEPSEEK_API_KEY")
        elif "gpt" in model_name.lower():
            return os.getenv("OPENAI_API_KEY")
        else:
            return os.getenv("LLM_API_KEY")
    
    def _init_tokenizer(self):
        """初始化tokenizer"""
        if not TIKTOKEN_AVAILABLE:
            return None
        
        try:
            encoding_name = self.model_config["encoding"]
            tokenizer = tiktoken.get_encoding(encoding_name)
            self.logger.debug(f"使用tiktoken编码器: {encoding_name}")
            return tokenizer
        except Exception as e:
            self.logger.warning(f"tiktoken初始化失败: {e}，使用简单估算")
            return None
    
    def _init_client(self):
        """初始化客户端"""
        if "deepseek" in self.model_name.lower():
            return OpenAI(
                api_key=self.api_key,
                base_url="https://api.deepseek.com"
            )
        elif "gpt" in self.model_name.lower():
            return OpenAI(api_key=self.api_key)
        else:
            return OpenAI(api_key=self.api_key)
    
    def count_tokens(self, text: str) -> int:
        """
        计算token数量
        优先使用tiktoken，fallback到简化估算
        """
        if self.tokenizer:
            try:
                return len(self.tokenizer.encode(text))
            except Exception:
                pass
        
        # 简化的token估算：按您的要求
        # 1 个英文字符 ≈ 0.3 个 token
        # 1 个中文字符 ≈ 0.6 个 token
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        english_chars = len(re.findall(r'[a-zA-Z]', text))
        other_chars = len(text) - chinese_chars - english_chars
        
        estimated_tokens = int(
            chinese_chars * 0.6 +      # 中文字符
            english_chars * 0.3 +      # 英文字符
            other_chars * 0.4          # 其他字符（数字、标点等）
        )
        
        return max(estimated_tokens, 1)  # 至少1个token
    
    def truncate_context(self, context_text: str, max_tokens: Optional[int] = None) -> str:
        """
        简化的上下文截断
        保留最后的内容（最新的对话通常最重要）
        """
        if max_tokens is None:
            max_tokens = self.max_context_tokens
        
        current_tokens = self.count_tokens(context_text)
        
        if current_tokens <= max_tokens:
            return context_text
        
        # 按行截断，从后往前保留
        lines = context_text.split('\n')
        selected_lines = []
        current_tokens = 0
        
        for line in reversed(lines):
            line_tokens = self.count_tokens(line + '\n')
            if current_tokens + line_tokens <= max_tokens:
                selected_lines.insert(0, line)
                current_tokens += line_tokens
            else:
                break
        
        result = '\n'.join(selected_lines)
        final_tokens = self.count_tokens(result)
        
        if final_tokens != current_tokens:  # 只在截断时记录
            self.logger.info(f"📏 上下文截断: {self.count_tokens(context_text)} -> {final_tokens} tokens")
        
        return result
    
    def get_context_info(self) -> Dict[str, Any]:
        """获取LLM上下文配置信息"""
        return {
            "model_name": self.model_name,
            "context_length": self.context_length,
            "max_output_tokens": self.max_output_tokens,
            "default_output_tokens": self.default_output_tokens,
            "max_context_tokens": self.max_context_tokens,
            "tokenizer_available": self.tokenizer is not None,
            "encoding": self.model_config.get("encoding", "unknown")
        }

    def generate_answer(self, 
                   prompt: str, 
                   max_tokens: Optional[int] = None,
                   temperature: float = 0.1,
                   generate_strategy: str = "default",
                   **kwargs) -> str:
        """
        生成答案 - 简化版本，移除不支持的参数
        
        Args:
            prompt: 完整提示词
            max_tokens: 最大输出token数（None时使用默认值）
            temperature: 温度参数
            generate_strategy: 生成策略 ("default", "max")
            **kwargs: 其他DeepSeek API参数
            
        Returns:
            生成的答案
        """
        
        # 设置输出token数
        if max_tokens is None:
            if generate_strategy == "max":
                max_tokens = self.max_output_tokens
            else:
                # 默认使用模型的默认输出token数
                max_tokens = self.default_output_tokens
        else:
            max_tokens = min(max_tokens, self.max_output_tokens)  # 不超过模型限制
        
        # 确保prompt不超过上下文限制
        prompt_tokens = self.count_tokens(prompt)
        max_prompt_tokens = self.context_length - max_tokens - 100  # 预留100缓冲
        
        if prompt_tokens > max_prompt_tokens:
            self.logger.warning(f"⚠️ Prompt过长 ({prompt_tokens} > {max_prompt_tokens})，截断中...")
            prompt = self.truncate_context(prompt, max_prompt_tokens)
            prompt_tokens = self.count_tokens(prompt)
            self.logger.info(f"📏 Prompt截断后: {prompt_tokens} tokens")
        
        try:
            # 构建请求参数（遵循DeepSeek API规范）
            request_params = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature
            }
            
            # 添加其他支持的参数
            supported_params = ['frequency_penalty', 'presence_penalty', 'top_p', 'stop']
            for param in supported_params:
                if param in kwargs:
                    request_params[param] = kwargs[param]
            
            # 如果是DeepSeek模型且明确要求JSON格式，添加response_format
            if "deepseek" in self.model_name.lower() and kwargs.get('json_format', False):
                request_params["response_format"] = {"type": "json_object"}
                # 确保prompt中包含JSON指令
                if "json" not in prompt.lower():
                    prompt += "\n\n请以JSON格式回复。"
                    request_params["messages"] = [{"role": "user", "content": prompt}]
            
            self.logger.debug(f"🚀 发送请求: {prompt_tokens} tokens -> max {max_tokens} tokens")
            
            response = self.client.chat.completions.create(**request_params)
            answer = response.choices[0].message.content.strip()
            
            # 记录token使用
            if hasattr(response, 'usage') and response.usage:
                usage = response.usage
                self.logger.debug(f"📊 Token使用: 输入={usage.prompt_tokens}, "
                                f"输出={usage.completion_tokens}, "
                                f"总计={usage.total_tokens}")
            else:
                # 估算输出token数
                estimated_output = self.count_tokens(answer)
                self.logger.debug(f"📊 Token估算: 输入≈{prompt_tokens}, 输出≈{estimated_output}")
            
            return answer
            
        except Exception as e:
            self.logger.error(f"❌ 生成失败: {e}")
            return f"生成失败: {str(e)}"

    def get_context_info(self) -> Dict[str, Any]:
        """获取LLM上下文配置信息"""
        return {
            "model_name": self.model_name,
            "context_length": self.context_length,
            "max_output_tokens": self.max_output_tokens,
            "default_output_tokens": self.default_output_tokens,
            "tokenizer_available": self.tokenizer is not None,
            "encoding": self.model_config.get("encoding", "unknown")
        }
    
    def batch_generate(self, 
                      prompts: List[str], 
                      max_tokens: Optional[int] = None,
                      temperature: float = 0.1,
                      **kwargs) -> List[str]:
        """批量生成答案"""
        results = []
        
        self.logger.info(f"🔄 批量生成开始: {len(prompts)} 个请求")
        
        for i, prompt in enumerate(prompts, 1):
            self.logger.debug(f"处理 {i}/{len(prompts)}")
            answer = self.generate_answer(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
            results.append(answer)
        
        self.logger.info(f"✅ 批量生成完成")
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型配置信息"""
        return {
            "model_name": self.model_name,
            "context_length": self.context_length,
            "max_context_tokens": self.max_context_tokens,
            "max_output_tokens": self.max_output_tokens,
            "default_output_tokens": self.default_output_tokens,
            "max_context_ratio": self.max_context_ratio,
            "tokenizer_available": self.tokenizer is not None,
            "encoding": self.model_config["encoding"]
        }
    
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """分析文本的token信息"""
        total_tokens = self.count_tokens(text)
        
        # 字符统计
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        english_chars = len(re.findall(r'[a-zA-Z]', text))
        
        return {
            "total_tokens": total_tokens,
            "character_count": len(text),
            "chinese_chars": chinese_chars,
            "english_chars": english_chars,
            "tokens_per_char": total_tokens / len(text) if text else 0,
            "fits_in_context": total_tokens <= self.max_context_tokens,
            "usage_ratio": total_tokens / self.context_length,
            "can_process": total_tokens <= (self.context_length - self.default_output_tokens)
        }


# 便捷函数
def create_deepseek_client(model: str = "deepseek-chat", 
                          api_key: Optional[str] = None) -> LLMClient:
    """创建DeepSeek客户端的便捷函数"""
    return LLMClient(model_name=model, api_key=api_key)


def estimate_tokens(text: str) -> int:
    """快速估算token数的独立函数"""
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    english_chars = len(re.findall(r'[a-zA-Z]', text))
    other_chars = len(text) - chinese_chars - english_chars
    
    return int(chinese_chars * 0.6 + english_chars * 0.3 + other_chars * 0.4)


# 使用示例
if __name__ == "__main__":
    # 测试客户端
    client = LLMClient("deepseek-chat")
    
    # 测试文本分析
    test_text = "Hello world! 你好世界！这是一个测试文本。"
    analysis = client.analyze_text(test_text)
    print(f"文本分析: {analysis}")
    
    # 测试生成
    prompt = "请简单介绍一下人工智能。"
    answer = client.generate_answer(prompt, max_tokens=200)
    print(f"生成结果: {answer}")
    
    # 显示模型信息
    info = client.get_model_info()
    print(f"模型信息: {info}")
