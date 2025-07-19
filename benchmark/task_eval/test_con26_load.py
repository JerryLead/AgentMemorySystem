#!/usr/bin/env python3
"""
测试conv-26数据集加载情况，验证datetime字段填充
"""

import json
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent.parent))

from dev import SemanticGraph, MemoryUnit
from benchmark.llm_utils.llm_client import LLMClient
from benchmark.task_eval.QA_evaluator import NewEvalExperiment

def test_conv26_datetime_loading():
    """测试conv-26数据集的datetime加载情况"""
    
    # 设置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    print("=" * 60)
    print("🧪 测试conv-26数据集datetime字段加载")
    print("=" * 60)
    
    # 初始化实验框架
    try:
        llm_client = LLMClient(model_name="deepseek-chat")
        experiment = NewEvalExperiment(
            llm_client=llm_client,
            output_dir="test_results/datetime_test"
        )
        print("✅ 实验框架初始化成功")
    except Exception as e:
        print(f"❌ 实验框架初始化失败: {e}")
        return
    
    # 测试数据加载
    conv_id = "conv-26"
    fused_dir = "benchmark/dataset/locomo/fused_llm"
    
    print(f"\n📁 加载数据集: {conv_id}")
    print(f"📁 Fused目录: {fused_dir}")
    
    try:
        # 调用加载函数
        graph, qa_pairs = experiment.load_single_conversation_from_fused(conv_id, fused_dir)
        
        if not qa_pairs:
            print("❌ 没有加载到QA数据")
            return
        
        print(f"✅ 成功加载 {len(qa_pairs)} 个QA问题")
        
        # 检查图谱中的数据
        print(f"\n🔍 检查语义图谱数据...")
        
        # 获取所有memory space
        ms_names = [f"conversation_{conv_id}"]
        all_units = graph.get_units_in_memory_space(ms_names)
        
        print(f"📊 总计加载了 {len(all_units)} 个内存单元")
        
        # 分析数据类型分布
        type_stats = {}
        datetime_stats = {
            "有时间": 0,
            "无时间": 0,
            "时间样例": []
        }
        
        for unit in all_units:
            data_type = unit.raw_data.get("data_type", "unknown")
            datetime_val = unit.raw_data.get("datetime", "")
            
            # 统计数据类型
            type_stats[data_type] = type_stats.get(data_type, 0) + 1
            
            # 统计时间字段
            if datetime_val and datetime_val.strip():
                datetime_stats["有时间"] += 1
                if len(datetime_stats["时间样例"]) < 5:  # 只收集前5个样例
                    datetime_stats["时间样例"].append({
                        "uid": unit.uid,
                        "data_type": data_type,
                        "datetime": datetime_val,
                        "session": unit.raw_data.get("session", ""),
                        "speaker": unit.raw_data.get("speaker", "")
                    })
            else:
                datetime_stats["无时间"] += 1
        
        print(f"\n📊 数据类型分布:")
        for data_type, count in sorted(type_stats.items()):
            print(f"   - {data_type}: {count} 个")
        
        print(f"\n⏰ 时间字段统计:")
        print(f"   - 有时间信息: {datetime_stats['有时间']} 个")
        print(f"   - 无时间信息: {datetime_stats['无时间']} 个")
        print(f"   - 时间填充率: {datetime_stats['有时间'] / len(all_units) * 100:.1f}%")
        
        print(f"\n📝 时间信息样例:")
        for i, sample in enumerate(datetime_stats["时间样例"], 1):
            print(f"   [{i}] {sample['data_type']} - {sample['session']}")
            print(f"       时间: {sample['datetime']}")
            print(f"       说话人: {sample['speaker']}")
            print(f"       UID: {sample['uid']}")
            print()
        
        # 详细检查对话数据
        print(f"\n🔍 详细检查对话数据...")
        dialogue_units = [u for u in all_units if u.raw_data.get("data_type") == "dialogue"]
        
        print(f"📋 对话单元: {len(dialogue_units)} 个")
        
        # 按session分组检查
        session_groups = {}
        for unit in dialogue_units:
            session = unit.raw_data.get("session", "unknown")
            if session not in session_groups:
                session_groups[session] = []
            session_groups[session].append(unit)
        
        print(f"📊 Session分布:")
        for session, units in sorted(session_groups.items()):
            if units:
                first_unit = units[0]
                datetime_val = first_unit.raw_data.get("datetime", "")
                print(f"   - {session}: {len(units)} 个对话, 时间: {datetime_val if datetime_val else '❌ 无时间'}")
        
        # 构建测试结果
        test_result = {
            "测试信息": {
                "测试时间": datetime.now().isoformat(),
                "对话ID": conv_id,
                "Fused目录": fused_dir,
                "QA问题数量": len(qa_pairs),
                "总内存单元数": len(all_units)
            },
            "数据类型统计": type_stats,
            "时间字段统计": {
                "有时间信息": datetime_stats["有时间"],
                "无时间信息": datetime_stats["无时间"],
                "时间填充率": f"{datetime_stats['有时间'] / len(all_units) * 100:.1f}%"
            },
            "时间信息样例": datetime_stats["时间样例"],
            "Session分析": {},
            "对话单元详情": []
        }
        
        # 详细的session分析
        for session, units in sorted(session_groups.items()):
            if units:
                first_unit = units[0]
                test_result["Session分析"][session] = {
                    "对话数量": len(units),
                    "时间信息": first_unit.raw_data.get("datetime", ""),
                    "是否有时间": bool(first_unit.raw_data.get("datetime", "").strip()),
                    "说话人列表": list(set([u.raw_data.get("speaker", "") for u in units if u.raw_data.get("speaker")]))
                }
        
        # 详细的对话单元信息（前10个）
        for i, unit in enumerate(dialogue_units[:10]):
            test_result["对话单元详情"].append({
                "索引": i + 1,
                "UID": unit.uid,
                "对话ID": unit.raw_data.get("dialogue_id", ""),
                "Session": unit.raw_data.get("session", ""),
                "说话人": unit.raw_data.get("speaker", ""),
                "时间信息": unit.raw_data.get("datetime", ""),
                "文本内容": unit.raw_data.get("text_content", "")[:100] + "..." if len(unit.raw_data.get("text_content", "")) > 100 else unit.raw_data.get("text_content", ""),
                "是否有时间": bool(unit.raw_data.get("datetime", "").strip())
            })
        
        # 保存测试结果
        output_dir = Path("test_results/datetime_test")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        result_file = output_dir / f"{conv_id}_datetime_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(test_result, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 测试结果已保存: {result_file}")
        print(f"📁 结果文件大小: {result_file.stat().st_size / 1024:.1f} KB")
        
        # 输出关键发现
        print(f"\n🎯 关键发现:")
        
        if datetime_stats["有时间"] > 0:
            print(f"✅ 时间字段修复成功！")
            print(f"   - {datetime_stats['有时间']}/{len(all_units)} 个单元有时间信息")
            print(f"   - 时间填充率: {datetime_stats['有时间'] / len(all_units) * 100:.1f}%")
        else:
            print(f"❌ 时间字段仍然为空！")
            print(f"   - 所有 {len(all_units)} 个单元都没有时间信息")
        
        # 检查是否主要是对话单元有时间
        dialogue_with_time = len([u for u in dialogue_units if u.raw_data.get("datetime", "").strip()])
        if dialogue_with_time > 0:
            print(f"✅ 对话单元时间填充: {dialogue_with_time}/{len(dialogue_units)} 个")
        else:
            print(f"❌ 对话单元完全没有时间信息")
        
        return test_result
        
    except Exception as e:
        print(f"❌ 测试过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return None

def inspect_raw_fused_data():
    """检查原始fused数据的结构"""
    
    print(f"\n" + "=" * 60)
    print("🔍 检查原始fused数据结构")
    print("=" * 60)
    
    fused_file = Path("benchmark/dataset/locomo/fused_llm/conv-26_fused.json")
    
    if not fused_file.exists():
        print(f"❌ 文件不存在: {fused_file}")
        return
    
    try:
        with open(fused_file, 'r', encoding='utf-8') as f:
            fused_data = json.load(f)
        
        print(f"✅ 成功加载原始数据")
        print(f"📁 文件大小: {fused_file.stat().st_size / 1024 / 1024:.1f} MB")
        
        # 检查顶层结构
        print(f"\n📊 顶层数据结构:")
        for key in fused_data.keys():
            if isinstance(fused_data[key], dict):
                print(f"   - {key}: 字典 (包含 {len(fused_data[key])} 个键)")
            elif isinstance(fused_data[key], list):
                print(f"   - {key}: 列表 (包含 {len(fused_data[key])} 个元素)")
            else:
                print(f"   - {key}: {type(fused_data[key]).__name__}")
        
        # 检查conversations结构
        conversations = fused_data.get("conversations", {})
        print(f"\n📋 conversations结构:")
        for key in conversations.keys():
            if isinstance(conversations[key], dict):
                print(f"   - {key}: 字典 (包含 {len(conversations[key])} 个键)")
            elif isinstance(conversations[key], list):
                print(f"   - {key}: 列表 (包含 {len(conversations[key])} 个元素)")
            else:
                print(f"   - {key}: {type(conversations[key]).__name__}")
        
        # 检查raw_conversation中的session时间
        raw_conversation = conversations.get("raw_conversation", {})
        print(f"\n⏰ raw_conversation中的时间信息:")
        
        session_times = {}
        session_dialogues = {}
        
        for key, value in raw_conversation.items():
            if key.endswith("_date_time"):
                session_key = key.replace("_date_time", "")
                session_times[session_key] = value
                print(f"   - {key}: {value}")
            elif key.startswith("session_") and isinstance(value, list):
                session_key = key
                session_dialogues[session_key] = len(value)
                print(f"   - {key}: {len(value)} 个对话")
        
        print(f"\n📊 Session时间映射验证:")
        for session_key in sorted(session_times.keys()):
            time_val = session_times.get(session_key, "")
            dialogue_count = session_dialogues.get(session_key, 0)
            print(f"   - {session_key}: 时间='{time_val}', 对话数={dialogue_count}")
        
        # 检查几个session的对话样例
        print(f"\n📝 对话样例检查 (前3个session):")
        count = 0
        for session_key, dialogues in raw_conversation.items():
            if session_key.startswith("session_") and isinstance(dialogues, list) and count < 3:
                session_time = session_times.get(session_key, "无时间")
                print(f"\n   Session: {session_key} (时间: {session_time})")
                
                for i, dialogue in enumerate(dialogues[:2]):  # 每个session显示前2个对话
                    dia_id = dialogue.get("dia_id", f"d{i}")
                    speaker = dialogue.get("speaker", "")
                    text = dialogue.get("text", "")[:50] + "..." if len(dialogue.get("text", "")) > 50 else dialogue.get("text", "")
                    print(f"     [{i+1}] {dia_id} - {speaker}: {text}")
                
                count += 1
        
        return {
            "文件路径": str(fused_file),
            "文件大小MB": fused_file.stat().st_size / 1024 / 1024,
            "顶层键": list(fused_data.keys()),
            "conversations键": list(conversations.keys()),
            "session时间映射": session_times,
            "session对话数": session_dialogues
        }
        
    except Exception as e:
        print(f"❌ 检查原始数据失败: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("🚀 开始conv-26数据集datetime字段测试")
    
    # 1. 检查原始数据结构
    raw_info = inspect_raw_fused_data()
    
    # 2. 测试加载过程
    test_result = test_conv26_datetime_loading()
    
    if test_result:
        print(f"\n🎉 测试完成！结果已保存到 test_results/datetime_test/ 目录")
    else:
        print(f"\n❌ 测试失败！")
    
    print(f"\n{'=' * 60}")
    print("📋 测试总结:")
    if raw_info:
        print(f"   - 原始数据检查: ✅ 成功")
        print(f"   - Session时间映射: {len(raw_info.get('session时间映射', {}))} 个")
    else:
        print(f"   - 原始数据检查: ❌ 失败")
    
    if test_result:
        print(f"   - 数据加载测试: ✅ 成功")
        time_stats = test_result.get("时间字段统计", {})
        print(f"   - 时间填充率: {time_stats.get('时间填充率', '0%')}")
    else:
        print(f"   - 数据加载测试: ❌ 失败")
    
    print("=" * 60)