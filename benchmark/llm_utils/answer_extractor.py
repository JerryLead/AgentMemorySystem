import re
from typing import Optional

class AnswerExtractor:
    """答案提取器"""
    
    def __init__(self):
        # 定义各种答案提取模式
        self.patterns = {
            "final_answer": [
                r"最终答案[：:]\s*(.+?)(?:\n|$)",
                r"答案[：:]\s*(.+?)(?:\n|$)",
                r"Answer[：:]\s*(.+?)(?:\n|$)",
            ],
            "reasoning_answer": [
                r"最终答案[：:]\s*(.+?)(?:\n|$)",
                r"答案[：:]\s*(.+?)(?:\n|推理|$)",
            ],
            "direct_answer": [
                r"^(.+?)(?:\n|$)",  # 第一行作为答案
            ]
        }
    
    def extract_answer(self, 
                      llm_response: str, 
                      category: int = 1) -> str:
        """从LLM响应中提取答案"""
        
        if not llm_response or not llm_response.strip():
            return "无答案"
        
        # 清理响应文本
        cleaned_response = llm_response.strip()
        
        # 根据问题类别选择提取策略
        if category in [3, 4]:  # 推理类问题
            answer = self._extract_with_patterns(cleaned_response, "reasoning_answer")
        else:
            answer = self._extract_with_patterns(cleaned_response, "final_answer")
        
        # 如果没有匹配到模式，使用第一句话
        if not answer:
            answer = self._extract_first_sentence(cleaned_response)
        
        return self._clean_answer(answer)
    
    def _extract_with_patterns(self, text: str, pattern_type: str) -> Optional[str]:
        """使用模式提取答案"""
        patterns = self.patterns.get(pattern_type, [])
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(1).strip()
        
        return None
    
    def _extract_first_sentence(self, text: str) -> str:
        """提取第一个句子作为答案"""
        sentences = re.split(r'[。.!！?？\n]', text)
        if sentences:
            return sentences[0].strip()
        return text[:100]  # 如果没有句子分隔符，取前100字符
    
    def _clean_answer(self, answer: str) -> str:
        """清理答案文本"""
        if not answer:
            return "无答案"
        
        # 移除多余的空白字符
        answer = re.sub(r'\s+', ' ', answer).strip()
        
        # 移除引号
        answer = answer.strip('"\'""''')
        
        # 移除开头的标记词
        answer = re.sub(r'^(答案是?[：:]?|答案[：:]?|Answer[：:]?)\s*', '', answer, flags=re.IGNORECASE)
        
        return answer