import logging
import json
import sys
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent.parent))

from benchmark.task_eval.semantic_graph_qa_evaluator import SemanticGraphQAEvaluator
from benchmark.extractor.dataset_inserter import ConversationSemanticStorage
from benchmark.llm_utils.llm_client import LLMClient


def quick_test_single_question():
    """快速测试单个问题"""
    
    # 初始化
    storage = ConversationSemanticStorage()
    llm_client = LLMClient(model_name="deepseek-chat")
    evaluator = SemanticGraphQAEvaluator(storage.semantic_graph, llm_client)
    
    # 测试问题
    test_question = {
        "question": "What LGBTQ support group did Caroline attend?",
        "answer": "Connected LGBTQ Activists",
        "category": 2,
        "evidence": ["D10:3"]
    }
    
    conversation_id = "conv-26"
    
    print(f"🔍 测试问题: {test_question['question']}")
    print(f"📝 标准答案: {test_question['answer']}")
    print(f"🏷️ 类别: {test_question['category']}")
    
    # 检索上下文
    print("\n1️⃣ 检索相关上下文...")
    context = evaluator.retrieve_context_for_question(
        question=test_question["question"],
        category=test_question["category"],
        conversation_id=conversation_id,
        evidence=test_question.get("evidence", [])
    )
    
    print(f"✅ 检索到 {len(context['retrieved_nodes'])} 个节点")
    print(f"✅ 检索到 {len(context['retrieved_relations'])} 个关系")
    print(f"📊 证据召回率: {context.get('evidence_recall', 0):.2f}")
    
    # 显示部分检索内容
    print("\n📋 检索到的主要内容:")
    for i, node in enumerate(context["retrieved_nodes"][:3]):
        print(f"  [{i+1}] {node['data_type']}: {node['content'][:100]}...")
    
    # 生成答案
    print("\n2️⃣ 生成答案...")
    answer_result = evaluator.generate_answer(context)
    
    generated_answer = answer_result["generated_answer"]
    print(f"🤖 生成答案: {generated_answer}")
    
    # 简单评估
    print("\n3️⃣ 评估结果...")
    from benchmark.task_eval.evaluation import f1_score, exact_match_score
    
    f1 = f1_score(generated_answer, test_question["answer"])
    em = exact_match_score(generated_answer, test_question["answer"])
    
    print(f"📊 F1分数: {f1:.4f}")
    print(f"📊 精确匹配: {em}")
    
    # 显示提示词（用于调试）
    print(f"\n🔧 使用的提示词预览:")
    print(answer_result.get("prompt", "")[:300] + "...")


def test_category_strategies():
    """测试不同类别的策略"""
    
    storage = ConversationSemanticStorage()
    llm_client = LLMClient(model_name="deepseek-chat")
    evaluator = SemanticGraphQAEvaluator(storage.semantic_graph, llm_client)
    
    # 测试每个类别的一个问题
    test_questions = [
        {
            "question": "How does Caroline's LGBTQ activism connect to her adoption plans?",
            "answer": "Caroline chose an adoption agency that supports LGBTQ+ community",
            "category": 1,  # 多跳推理
            "evidence": []
        },
        {
            "question": "What is the name of Caroline's LGBTQ activist group?",
            "answer": "Connected LGBTQ Activists",
            "category": 2,  # 单跳事实
            "evidence": ["D10:3"]
        },
        {
            "question": "When did Caroline join the LGBTQ activist group?",
            "answer": "last Tuesday",
            "category": 3,  # 时间相关
            "evidence": ["D10:3"]
        }
    ]
    
    conversation_id = "conv-26"
    
    for i, test_q in enumerate(test_questions):
        print(f"\n{'='*50}")
        print(f"🧪 测试类别 {test_q['category']} - {evaluator.category_strategies[test_q['category']]}")
        print(f"❓ 问题: {test_q['question']}")
        
        # 检索和生成
        context = evaluator.retrieve_context_for_question(
            question=test_q["question"],
            category=test_q["category"],
            conversation_id=conversation_id,
            evidence=test_q.get("evidence", [])
        )
        
        answer_result = evaluator.generate_answer(context)
        
        print(f"🎯 标准答案: {test_q['answer']}")
        print(f"🤖 生成答案: {answer_result['generated_answer']}")
        
        # 评估
        from benchmark.task_eval.evaluation import f1_score
        f1 = f1_score(answer_result['generated_answer'], test_q['answer'])
        print(f"📊 F1分数: {f1:.4f}")
        print(f"📊 检索节点数: {len(context['retrieved_nodes'])}")
        print(f"📊 策略: {context['strategy']}")


if __name__ == "__main__":
    print("🧪 Semantic Graph QA Testing")
    print("1. 单问题测试")
    print("2. 类别策略测试")
    
    choice = input("\n选择测试类型 (1 或 2): ").strip()
    
    if choice == "1":
        quick_test_single_question()
    elif choice == "2":
        test_category_strategies()
    else:
        print("❌ 无效选择")