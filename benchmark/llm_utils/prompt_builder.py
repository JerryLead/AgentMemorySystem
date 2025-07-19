import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
from dev import MemoryUnit

class PromptBuilder:
    """提示词构建器"""
    
    def __init__(self):
        self.templates = {
            "qa_with_context": self._build_qa_template(),
            "qa_with_reasoning": self._build_reasoning_template(),
            "adversarial_qa": self._build_adversarial_template()
        }
    
    def _build_qa_template(self) -> str:
        return """你是一个智能问答助手。请根据提供的对话历史记录回答问题。

        ## 对话记录上下文
        {context}

        ## 问题
        {question}

        ## 回答要求
        1. 基于上述对话记录回答问题
        2. 如果对话记录中没有相关信息，请回答"对话中未提及相关信息"
        3. 回答要简洁准确，直接给出答案
        4. 如果涉及时间信息，请使用具体的日期格式

        ## 答案
        """

    def _build_reasoning_template(self) -> str:
        return """你是一个智能推理助手。请根据提供的对话历史记录，通过推理回答问题。

        ## 对话记录上下文
        {context}

        ## 问题
        {question}

        ## 推理要求
        1. 仔细分析对话记录中的信息
        2. 找出与问题相关的所有线索
        3. 进行逻辑推理得出答案
        4. 如果需要推理多个步骤，请简要说明推理过程

        ## 推理过程和答案
        推理过程：[简要说明]
        最终答案：[直接答案]
        """

    def _build_adversarial_template(self) -> str:
        return """你是一个谨慎的问答助手。请根据提供的对话历史记录回答问题。

        ## 对话记录上下文
        {context}

        ## 问题
        {question}

        ## 回答要求
        1. 仔细检查对话记录是否包含回答问题所需的信息
        2. 如果对话记录中确实包含相关信息，请给出准确答案
        3. 如果对话记录中没有足够信息回答问题，请回答"对话中未提及相关信息"
        4. 不要基于常识或外部知识进行推测

        ## 答案
        """

    def build_prompt(self, 
                    question: str, 
                    context_units: List[Tuple[MemoryUnit, float]], 
                    category: int = 1,
                    evidence: List[str] = None) -> str:
        """构建提示词"""
        
        # 格式化上下文
        context = self._format_context(context_units)
        
        # 根据问题类别选择模板
        if category == 5:  # 对抗性问题
            template = self.templates["adversarial_qa"]
        elif category in [3, 4]:  # 需要推理的问题
            template = self.templates["qa_with_reasoning"]
        else:  # 普通问答
            template = self.templates["qa_with_context"]
        
        return template.format(
            context=context,
            question=question
        )
    
    def _format_context(self, context_units: List[Tuple[MemoryUnit, float]]) -> str:
        """格式化上下文信息"""
        if not context_units:
            return "没有相关的对话记录。"
        
        context_lines = []
        for i, (unit, score) in enumerate(context_units, 1):
            data_source = unit.metadata.get('data_source', 'unknown')
            
            if data_source == 'locomo_summary':
                session = unit.raw_data.get('session', 'unknown')
                content = unit.raw_data.get('text_content', '')
                speakers = unit.raw_data.get('speakers', 'Unknown')
                line = f"{i}. [会话摘要-{session}] {speakers}: {content}"
                
            elif data_source == 'locomo_dialog':
                speaker = unit.raw_data.get('speaker', 'unknown')
                content = unit.raw_data.get('message_content', '')
                session = unit.metadata.get('session', 'unknown')
                timestamp = unit.metadata.get('timestamp', '')
                line = f"{i}. [对话-{session}] {speaker} ({timestamp}): {content}"
                
            elif data_source == 'locomo_qa':
                question = unit.raw_data.get('question', '')
                answer = unit.raw_data.get('answer', '')
                line = f"{i}. [相关QA] Q: {question} A: {answer}"
                
            else:
                content = unit.raw_data.get('text_content', '')[:200]
                line = f"{i}. [记录] {content}"
            
            context_lines.append(line)
        
        return "\n".join(context_lines)