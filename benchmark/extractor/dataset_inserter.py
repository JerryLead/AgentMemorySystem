import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from dev.semantic_graph import SemanticGraph
from dev.memory_unit import MemoryUnit


class ConversationSemanticStorage:
    """
    对话语义存储器 - 按 sample_id 将 LoCoMo 数据存储到 semantic_map/graph 中
    支持原始数据和抽取结果的混合存储，可指定存储特定对话
    """
    
    def __init__(self, output_dir: str = "benchmark/conversation_semantic_storage"):
        """初始化对话语义存储器"""
        # 首先设置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化语义图谱
        self.semantic_graph = SemanticGraph()
        
        self.logger.info("ConversationSemanticStorage 初始化完成")
    
    # def store_conversation(self, 
    #                       sample_id: str,
    #                       raw_dataset_file: str = "benchmark/dataset/locomo/locomo10.json",
    #                       extracted_dataset_file: str = "benchmark/dataset/locomo/extraction/locomo_extracted_full_dataset.json",
    #                       include_raw: bool = True,
    #                       include_extracted: bool = True) -> Dict[str, Any]:
    #     """
    #     存储指定对话到语义图谱中
        
    #     Args:
    #         sample_id: 对话ID (如 "conv-26", "conv-30")
    #         raw_dataset_file: 原始数据文件路径
    #         extracted_dataset_file: 抽取结果数据文件路径
    #         include_raw: 是否包含原始数据
    #         include_extracted: 是否包含抽取结果
            
    #     Returns:
    #         存储统计信息
    #     """
    #     self.logger.info(f"🚀 开始存储对话 {sample_id}")
        
    #     # 初始化存储统计
    #     storage_stats = {
    #         "sample_id": sample_id,
    #         "include_raw": include_raw,
    #         "include_extracted": include_extracted,
    #         "raw_dataset_file": raw_dataset_file,
    #         "extracted_dataset_file": extracted_dataset_file,
    #         "storage_breakdown": {
    #             # 原始数据
    #             "conversations": 0,
    #             "observations": 0,
    #             "events": 0,
    #             "summaries": 0,
    #             "qa_pairs": 0,
    #             # 抽取数据
    #             "entities": 0,
    #             "relationships": 0,
    #             "keywords": 0,
    #             "statistics": 0
    #         },
    #         "namespace_usage": {},
    #         "processing_time": {
    #             "start_time": datetime.now().isoformat()
    #         }
    #     }
        
    #     start_time = datetime.now()
        
    #     try:
    #         # 加载数据
    #         raw_data = None
    #         extracted_data = None
            
    #         if include_raw:
    #             raw_data = self._load_raw_conversation(raw_dataset_file, sample_id)
    #             if raw_data:
    #                 self.logger.info(f"✅ 成功加载原始数据: {sample_id}")
    #             else:
    #                 self.logger.warning(f"⚠️ 未找到原始数据: {sample_id}")
            
    #         if include_extracted:
    #             extracted_data = self._load_extracted_conversation(extracted_dataset_file, sample_id)
    #             if extracted_data:
    #                 self.logger.info(f"✅ 成功加载抽取数据: {sample_id}")
    #             else:
    #                 self.logger.warning(f"⚠️ 未找到抽取数据: {sample_id}")
            
    #         # 存储原始数据
    #         if raw_data:
    #             raw_stats = self._store_raw_conversation_data(raw_data, sample_id)
    #             for key, value in raw_stats.items():
    #                 storage_stats["storage_breakdown"][key] += value
            
    #         # 存储抽取结果
    #         if extracted_data:
    #             extracted_stats = self._store_extracted_conversation_data(extracted_data, sample_id)
    #             for key, value in extracted_stats.items():
    #                 storage_stats["storage_breakdown"][key] += value
                
    #             # 建立实体关系
    #             self._establish_entity_relationships(extracted_data, sample_id)
            
    #         # 构建语义索引
    #         self.logger.info("构建语义索引...")
    #         self.semantic_graph.build_semantic_map_index()
            
    #         # 统计命名空间使用情况
    #         storage_stats["namespace_usage"] = self._get_namespace_usage_stats()
            
    #         end_time = datetime.now()
    #         storage_stats["processing_time"]["end_time"] = end_time.isoformat()
    #         storage_stats["processing_time"]["duration_seconds"] = (end_time - start_time).total_seconds()
            
    #         # 保存存储统计
    #         stats_file = self.output_dir / f"{sample_id}_storage_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    #         with open(stats_file, 'w', encoding='utf-8') as f:
    #             json.dump(storage_stats, f, ensure_ascii=False, indent=2)
            
    #         self.logger.info(f"🎉 对话 {sample_id} 存储完成！统计信息保存至: {stats_file}")
    #         return storage_stats
            
    #     except Exception as e:
    #         self.logger.error(f"❌ 存储对话 {sample_id} 失败: {e}")
    #         raise

    # 更新存储统计信息结构
    def store_conversation(self, 
                        sample_id: str,
                        raw_dataset_file: str = "benchmark/dataset/locomo/locomo10.json",
                        extracted_dataset_file: str = "benchmark/dataset/locomo/extraction/locomo_extracted_full_dataset.json",
                        include_raw: bool = True,
                        include_extracted: bool = True) -> Dict[str, Any]:
        """
        存储指定对话到语义图谱中
        
        Args:
            sample_id: 对话ID (如 "conv-26", "conv-30")
            raw_dataset_file: 原始数据文件路径
            extracted_dataset_file: 抽取结果数据文件路径
            include_raw: 是否包含原始数据
            include_extracted: 是否包含抽取结果
            
        Returns:
            存储统计信息
        """
        self.logger.info(f"🚀 开始存储对话 {sample_id} (QA数据将保留作为测试集)")
        
        # 初始化存储统计
        storage_stats = {
            "sample_id": sample_id,
            "include_raw": include_raw,
            "include_extracted": include_extracted,
            "raw_dataset_file": raw_dataset_file,
            "extracted_dataset_file": extracted_dataset_file,
            "storage_breakdown": {
                # 原始数据
                "conversations": 0,
                "observations": 0,
                "events": 0,
                "summaries": 0,
                # "qa_pairs": 0,  # 移除QA统计
                # 抽取数据
                "entities": 0,
                "relationships": 0,
                "keywords": 0,
                "statistics": 0
            },
            "qa_info": {
                "total_qa_pairs": 0,
                "note": "QA数据保留作为测试集，未插入语义图谱"
            },
            "namespace_usage": {},
            "processing_time": {
                "start_time": datetime.now().isoformat()
            }
        }
        
        start_time = datetime.now()
        
        try:
            # 加载数据
            raw_data = None
            extracted_data = None
            
            if include_raw:
                raw_data = self._load_raw_conversation(raw_dataset_file, sample_id)
                if raw_data:
                    self.logger.info(f"✅ 成功加载原始数据: {sample_id}")
                    # 统计QA数据但不插入
                    qa_data = raw_data.get('qa', [])
                    storage_stats["qa_info"]["total_qa_pairs"] = len(qa_data)
                    self.logger.info(f"📊 发现 {len(qa_data)} 个QA对，将保留作为测试集")
                else:
                    self.logger.warning(f"⚠️ 未找到原始数据: {sample_id}")
            
            if include_extracted:
                extracted_data = self._load_extracted_conversation(extracted_dataset_file, sample_id)
                if extracted_data:
                    self.logger.info(f"✅ 成功加载抽取数据: {sample_id}")
                else:
                    self.logger.warning(f"⚠️ 未找到抽取数据: {sample_id}")
            
            # 存储原始数据（不包括QA）
            if raw_data:
                raw_stats = self._store_raw_conversation_data(raw_data, sample_id)
                for key, value in raw_stats.items():
                    storage_stats["storage_breakdown"][key] += value
            
            # 存储抽取结果
            if extracted_data:
                extracted_stats = self._store_extracted_conversation_data(extracted_data, sample_id)
                for key, value in extracted_stats.items():
                    storage_stats["storage_breakdown"][key] += value
                
                # 建立实体关系
                self._establish_entity_relationships(extracted_data, sample_id)
            
            # 构建语义索引
            self.logger.info("构建语义索引...")
            self.semantic_graph.build_semantic_map_index()
            
            # 统计命名空间使用情况
            storage_stats["namespace_usage"] = self._get_namespace_usage_stats()
            
            end_time = datetime.now()
            storage_stats["processing_time"]["end_time"] = end_time.isoformat()
            storage_stats["processing_time"]["duration_seconds"] = (end_time - start_time).total_seconds()
            
            # 保存存储统计
            stats_file = self.output_dir / f"{sample_id}_storage_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(storage_stats, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"🎉 对话 {sample_id} 存储完成！QA数据已保留作为测试集")
            self.logger.info(f"📈 统计信息保存至: {stats_file}")
            return storage_stats
            
        except Exception as e:
            self.logger.error(f"❌ 存储对话 {sample_id} 失败: {e}")
            raise
    
    # def store_all_conversations(self,
    #                            raw_dataset_file: str = "benchmark/dataset/locomo/locomo10.json",
    #                            extracted_dataset_file: str = "benchmark/dataset/locomo/extraction/locomo_extracted_full_dataset.json",
    #                            include_raw: bool = True,
    #                            include_extracted: bool = True) -> Dict[str, Any]:
    #     """
    #     存储所有对话到语义图谱中
        
    #     Args:
    #         raw_dataset_file: 原始数据文件路径
    #         extracted_dataset_file: 抽取结果数据文件路径
    #         include_raw: 是否包含原始数据
    #         include_extracted: 是否包含抽取结果
            
    #     Returns:
    #         所有对话的存储统计信息
    #     """
    #     self.logger.info("🚀 开始存储所有对话")
        
    #     # 获取所有可用的sample_id
    #     all_sample_ids = set()
        
    #     if include_raw:
    #         with open(raw_dataset_file, 'r', encoding='utf-8') as f:
    #             raw_dataset = json.load(f)
    #         for sample in raw_dataset:
    #             sample_id = sample.get('sample_id')
    #             if sample_id:
    #                 all_sample_ids.add(sample_id)
        
    #     if include_extracted:
    #         with open(extracted_dataset_file, 'r', encoding='utf-8') as f:
    #             extracted_dataset = json.load(f)
    #         extracted_samples = extracted_dataset.get("samples", {})
    #         all_sample_ids.update(extracted_samples.keys())
        
    #     self.logger.info(f"发现 {len(all_sample_ids)} 个对话: {sorted(all_sample_ids)}")
        
    #     # 存储每个对话
    #     all_stats = {
    #         "total_conversations": len(all_sample_ids),
    #         "processed_conversations": 0,
    #         "failed_conversations": [],
    #         "conversation_stats": {},
    #         "overall_storage_breakdown": {
    #             "conversations": 0,
    #             "observations": 0,
    #             "events": 0,
    #             "summaries": 0,
    #             "qa_pairs": 0,
    #             "entities": 0,
    #             "relationships": 0,
    #             "keywords": 0,
    #             "statistics": 0
    #         },
    #         "processing_time": {
    #             "start_time": datetime.now().isoformat()
    #         }
    #     }
        
    #     start_time = datetime.now()
        
    #     for sample_id in sorted(all_sample_ids):
    #         try:
    #             stats = self.store_conversation(
    #                 sample_id=sample_id,
    #                 raw_dataset_file=raw_dataset_file,
    #                 extracted_dataset_file=extracted_dataset_file,
    #                 include_raw=include_raw,
    #                 include_extracted=include_extracted
    #             )
                
    #             all_stats["conversation_stats"][sample_id] = stats
    #             all_stats["processed_conversations"] += 1
                
    #             # 累加统计信息
    #             for key, value in stats["storage_breakdown"].items():
    #                 all_stats["overall_storage_breakdown"][key] += value
                
    #             self.logger.info(f"✅ {sample_id} 处理完成")
                
    #         except Exception as e:
    #             self.logger.error(f"❌ {sample_id} 处理失败: {e}")
    #             all_stats["failed_conversations"].append(sample_id)
        
    #     end_time = datetime.now()
    #     all_stats["processing_time"]["end_time"] = end_time.isoformat()
    #     all_stats["processing_time"]["duration_seconds"] = (end_time - start_time).total_seconds()
        
    #     # 保存总体统计
    #     stats_file = self.output_dir / f"all_conversations_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    #     with open(stats_file, 'w', encoding='utf-8') as f:
    #         json.dump(all_stats, f, ensure_ascii=False, indent=2)
        
    #     self.logger.info(f"🎉 所有对话存储完成！总体统计保存至: {stats_file}")
    #     return all_stats

    # 更新 store_all_conversations 方法的统计结构
    def store_all_conversations(self,
                            raw_dataset_file: str = "benchmark/dataset/locomo/locomo10.json",
                            extracted_dataset_file: str = "benchmark/dataset/locomo/extraction/locomo_extracted_full_dataset.json",
                            include_raw: bool = True,
                            include_extracted: bool = True) -> Dict[str, Any]:
        """
        存储所有对话到语义图谱中
        
        Args:
            raw_dataset_file: 原始数据文件路径
            extracted_dataset_file: 抽取结果数据文件路径
            include_raw: 是否包含原始数据
            include_extracted: 是否包含抽取结果
            
        Returns:
            所有对话的存储统计信息
        """
        self.logger.info("🚀 开始存储所有对话 (QA数据将保留作为测试集)")
        
        # 获取所有可用的sample_id
        all_sample_ids = set()
        
        if include_raw:
            with open(raw_dataset_file, 'r', encoding='utf-8') as f:
                raw_dataset = json.load(f)
            for sample in raw_dataset:
                sample_id = sample.get('sample_id')
                if sample_id:
                    all_sample_ids.add(sample_id)
        
        if include_extracted:
            with open(extracted_dataset_file, 'r', encoding='utf-8') as f:
                extracted_dataset = json.load(f)
            extracted_samples = extracted_dataset.get("samples", {})
            all_sample_ids.update(extracted_samples.keys())
        
        self.logger.info(f"发现 {len(all_sample_ids)} 个对话: {sorted(all_sample_ids)}")
        
        # 存储每个对话
        all_stats = {
            "total_conversations": len(all_sample_ids),
            "processed_conversations": 0,
            "failed_conversations": [],
            "conversation_stats": {},
            "overall_storage_breakdown": {
                "conversations": 0,
                "observations": 0,
                "events": 0,
                "summaries": 0,
                # "qa_pairs": 0,  # 移除QA统计
                "entities": 0,
                "relationships": 0,
                "keywords": 0,
                "statistics": 0
            },
            "overall_qa_info": {
                "total_qa_pairs": 0,
                "note": "所有QA数据保留作为测试集，未插入语义图谱"
            },
            "processing_time": {
                "start_time": datetime.now().isoformat()
            }
        }
        
        start_time = datetime.now()
        
        for sample_id in sorted(all_sample_ids):
            try:
                stats = self.store_conversation(
                    sample_id=sample_id,
                    raw_dataset_file=raw_dataset_file,
                    extracted_dataset_file=extracted_dataset_file,
                    include_raw=include_raw,
                    include_extracted=include_extracted
                )
                
                all_stats["conversation_stats"][sample_id] = stats
                all_stats["processed_conversations"] += 1
                
                # 累加存储统计信息
                for key, value in stats["storage_breakdown"].items():
                    all_stats["overall_storage_breakdown"][key] += value
                
                # 累加QA信息
                all_stats["overall_qa_info"]["total_qa_pairs"] += stats["qa_info"]["total_qa_pairs"]
                
                self.logger.info(f"✅ {sample_id} 处理完成")
                
            except Exception as e:
                self.logger.error(f"❌ {sample_id} 处理失败: {e}")
                all_stats["failed_conversations"].append(sample_id)
        
        end_time = datetime.now()
        all_stats["processing_time"]["end_time"] = end_time.isoformat()
        all_stats["processing_time"]["duration_seconds"] = (end_time - start_time).total_seconds()
        
        # 保存总体统计
        stats_file = self.output_dir / f"all_conversations_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(all_stats, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"🎉 所有对话存储完成！")
        self.logger.info(f"📊 总计保留 {all_stats['overall_qa_info']['total_qa_pairs']} 个QA对作为测试集")
        self.logger.info(f"📈 总体统计保存至: {stats_file}")
        return all_stats
    
    def _load_raw_conversation(self, file_path: str, sample_id: str) -> Optional[Dict]:
        """从原始数据集中加载指定对话"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                dataset = json.load(f)
            
            for sample in dataset:
                if sample.get('sample_id') == sample_id:
                    return sample
            
            return None
        except Exception as e:
            self.logger.error(f"加载原始对话 {sample_id} 失败: {e}")
            return None
    
    def _load_extracted_conversation(self, file_path: str, sample_id: str) -> Optional[Dict]:
        """从抽取结果数据集中加载指定对话"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                dataset = json.load(f)
            
            samples = dataset.get("samples", {})
            return samples.get(sample_id)
        except Exception as e:
            self.logger.error(f"加载抽取对话 {sample_id} 失败: {e}")
            return None
        
    # 在 _store_raw_conversation_data 方法中，注释掉 QA 数据存储
    def _store_raw_conversation_data(self, sample: Dict, sample_id: str) -> Dict[str, int]:
        """存储单个对话的原始数据"""
        stats = {
            "conversations": 0,
            "observations": 0,
            "events": 0,
            "summaries": 0,
            # "qa_pairs": 0  # 移除QA统计
        }
        
        # 创建对话专用空间
        self.semantic_graph.create_memory_space_in_map(f"conversation_{sample_id}")
        self.semantic_graph.create_memory_space_in_map(f"raw_data_{sample_id}")
        
        # 1. 存储对话数据
        conv_count = self._store_conversation_data(sample, sample_id)
        stats["conversations"] += conv_count
        
        # 2. 存储观察记录
        obs_count = self._store_observation_data(sample, sample_id)
        stats["observations"] += obs_count
        
        # 3. 存储事件记录
        event_count = self._store_event_data(sample, sample_id)
        stats["events"] += event_count
        
        # 4. 存储会话摘要
        summary_count = self._store_summary_data(sample, sample_id)
        stats["summaries"] += summary_count
        
        # 5. 存储问答数据 - 注释掉
        # qa_count = self._store_qa_data(sample, sample_id)
        # stats["qa_pairs"] += qa_count
        
        return stats
    
    # def _store_raw_conversation_data(self, sample: Dict, sample_id: str) -> Dict[str, int]:
    #     """存储单个对话的原始数据"""
    #     stats = {
    #         "conversations": 0,
    #         "observations": 0,
    #         "events": 0,
    #         "summaries": 0,
    #         "qa_pairs": 0
    #     }
        
    #     # 创建对话专用空间
    #     self.semantic_graph.create_memory_space_in_map(f"conversation_{sample_id}")
    #     self.semantic_graph.create_memory_space_in_map(f"raw_data_{sample_id}")
        
    #     # 1. 存储对话数据
    #     conv_count = self._store_conversation_data(sample, sample_id)
    #     stats["conversations"] += conv_count
        
    #     # 2. 存储观察记录
    #     obs_count = self._store_observation_data(sample, sample_id)
    #     stats["observations"] += obs_count
        
    #     # 3. 存储事件记录
    #     event_count = self._store_event_data(sample, sample_id)
    #     stats["events"] += event_count
        
    #     # 4. 存储会话摘要
    #     summary_count = self._store_summary_data(sample, sample_id)
    #     stats["summaries"] += summary_count
        
    #     # 5. 存储问答数据
    #     qa_count = self._store_qa_data(sample, sample_id)
    #     stats["qa_pairs"] += qa_count
        
    #     return stats
    
    def _store_extracted_conversation_data(self, sample_data: Dict, sample_id: str) -> Dict[str, int]:
        """存储单个对话的抽取结果数据"""
        stats = {
            "entities": 0,
            "relationships": 0,
            "keywords": 0,
            "statistics": 0
        }
        
        # 创建抽取数据专用空间
        self.semantic_graph.create_memory_space_in_map(f"conversation_{sample_id}")
        self.semantic_graph.create_memory_space_in_map(f"extracted_data_{sample_id}")
        
        # 1. 存储抽取的实体
        entity_stats = self._store_extracted_entities(sample_data, sample_id)
        stats["entities"] += entity_stats
        
        # 2. 存储抽取的关系
        rel_stats = self._store_extracted_relationships(sample_data, sample_id)
        stats["relationships"] += rel_stats
        
        # 3. 存储关键词
        keyword_stats = self._store_keywords(sample_data, sample_id)
        stats["keywords"] += keyword_stats
        
        # 4. 存储统计信息
        stats_count = self._store_statistics(sample_data, sample_id)
        stats["statistics"] += stats_count
        
        return stats
    
    def _store_conversation_data(self, sample: Dict, sample_id: str) -> int:
        """存储对话数据"""
        conversation = sample.get('conversation', {})
        if not conversation:
            return 0
        
        # 获取说话者信息
        speaker_a = conversation.get('speaker_a', 'Speaker A')
        speaker_b = conversation.get('speaker_b', 'Speaker B')
        
        # 按会话存储对话
        session_keys = [k for k in conversation.keys() 
                       if k.startswith('session_') and not k.endswith('_date_time')]
        
        # 1. 创建完整对话记忆单元
        full_text_parts = []
        for session_key in sorted(session_keys, key=lambda x: int(x.split('_')[1])):
            session_messages = conversation.get(session_key, [])
            session_datetime = conversation.get(f"{session_key}_date_time", "")
            
            if session_messages and session_datetime:
                session_text = f"\n=== {session_key.upper()} ({session_datetime}) ===\n"
                for msg in session_messages:
                    if isinstance(msg, dict):
                        speaker = msg.get('speaker', 'Unknown')
                        content = msg.get('content', msg.get('text', ''))
                        session_text += f"{speaker}: {content}\n"
                full_text_parts.append(session_text)
        
        full_conversation = "\n".join(full_text_parts)
        
        if full_conversation.strip():
            unit = MemoryUnit(
                uid=f"{sample_id}_full_conversation",
                raw_data={
                    "text_content": full_conversation,
                    "speakers": [speaker_a, speaker_b],
                    "session_count": len(session_keys),
                    "total_length": len(full_conversation),
                    "sample_id": sample_id,
                    "data_type": "full_conversation",
                    "data_source": "raw"
                },
                metadata={
                    "conversation_id": sample_id,
                    "data_layer": "raw",
                    "content_type": "full_conversation",
                    "speakers": [speaker_a, speaker_b],
                    "created": datetime.now().isoformat()
                }
            )
            
            # 添加到对话专用空间
            self.semantic_graph.add_unit(unit, space_names=[
                f"conversation_{sample_id}",  # 对话专用空间
                f"raw_data_{sample_id}",      # 原始数据空间
                "all_conversations"           # 全局对话空间
            ])
            
            return 1
        
        return 0
    
    def _store_observation_data(self, sample: Dict, sample_id: str) -> int:
        """存储观察记录"""
        observations = sample.get('observation', {})
        if not observations:
            return 0
        
        stored_count = 0
        
        for obs_session_key, obs_data in observations.items():
            if not isinstance(obs_data, dict):
                continue
                
            for speaker, obs_list in obs_data.items():
                if not isinstance(obs_list, list):
                    continue
                    
                for obs_idx, obs_item in enumerate(obs_list):
                    if isinstance(obs_item, list) and len(obs_item) >= 1:
                        obs_content = obs_item[0]
                        evidence_id = obs_item[1] if len(obs_item) > 1 else ""
                        
                        if not obs_content or not obs_content.strip():
                            continue
                        
                        # 创建观察记录单元
                        unit = MemoryUnit(
                            uid=f"{sample_id}_{obs_session_key}_{speaker}_obs_{obs_idx}",
                            raw_data={
                                "text_content": obs_content,
                                "session_id": obs_session_key,
                                "speaker": speaker,
                                "evidence_id": evidence_id,
                                "observation_index": obs_idx,
                                "sample_id": sample_id,
                                "data_type": "observation",
                                "data_source": "raw"
                            },
                            metadata={
                                "conversation_id": sample_id,
                                "data_layer": "raw",
                                "content_type": "observation",
                                "session_id": obs_session_key,
                                "speaker": speaker,
                                "created": datetime.now().isoformat()
                            }
                        )
                        
                        # 添加到对话专用空间
                        self.semantic_graph.add_unit(unit, space_names=[
                            f"conversation_{sample_id}",
                            f"raw_data_{sample_id}",
                            "all_observations"
                        ])
                        
                        stored_count += 1
        
        return stored_count
    
    def _store_event_data(self, sample: Dict, sample_id: str) -> int:
        """存储事件记录"""
        event_summary = sample.get('event_summary', {})
        if not event_summary:
            return 0
        
        stored_count = 0
        
        for event_session_key, event_data in event_summary.items():
            if not isinstance(event_data, dict):
                continue
                
            event_date = event_data.get('date', '')
            
            for speaker, events in event_data.items():
                if speaker == 'date' or not isinstance(events, list):
                    continue
                    
                for event_idx, event_desc in enumerate(events):
                    if not isinstance(event_desc, str) or not event_desc.strip():
                        continue
                    
                    # 创建事件记录单元
                    unit = MemoryUnit(
                        uid=f"{sample_id}_{event_session_key}_{speaker}_event_{event_idx}",
                        raw_data={
                            "text_content": event_desc,
                            "session_id": event_session_key,
                            "event_date": event_date,
                            "speaker": speaker,
                            "event_index": event_idx,
                            "sample_id": sample_id,
                            "data_type": "event",
                            "data_source": "raw"
                        },
                        metadata={
                            "conversation_id": sample_id,
                            "data_layer": "raw",
                            "content_type": "event",
                            "session_id": event_session_key,
                            "speaker": speaker,
                            "event_date": event_date,
                            "created": datetime.now().isoformat()
                        }
                    )
                    
                    # 添加到对话专用空间
                    self.semantic_graph.add_unit(unit, space_names=[
                        f"conversation_{sample_id}",
                        f"raw_data_{sample_id}",
                        "all_events"
                    ])
                    
                    stored_count += 1
        
        return stored_count
    
    def _store_summary_data(self, sample: Dict, sample_id: str) -> int:
        """存储会话摘要"""
        session_summary = sample.get('session_summary', {})
        if not session_summary:
            return 0
        
        stored_count = 0
        
        for summary_key, summary_content in session_summary.items():
            if not isinstance(summary_content, str) or not summary_content.strip():
                continue
            
            # 创建摘要记录单元
            unit = MemoryUnit(
                uid=f"{sample_id}_{summary_key}_summary",
                raw_data={
                    "text_content": summary_content,
                    "summary_key": summary_key,
                    "sample_id": sample_id,
                    "data_type": "summary",
                    "data_source": "raw"
                },
                metadata={
                    "conversation_id": sample_id,
                    "data_layer": "raw", 
                    "content_type": "summary",
                    "summary_type": summary_key,
                    "created": datetime.now().isoformat()
                }
            )
            
            # 添加到对话专用空间
            self.semantic_graph.add_unit(unit, space_names=[
                f"conversation_{sample_id}",
                f"raw_data_{sample_id}",
                "all_summaries"
            ])
            
            stored_count += 1
        
        return stored_count
    
    # 修改 _store_qa_data 方法为空实现
    def _store_qa_data(self, sample: Dict, sample_id: str) -> int:
        """
        存储问答数据 - 已禁用
        QA数据将作为测试集使用，不插入到语义图谱中
        """
        self.logger.info(f"跳过QA数据存储 - QA数据将用作测试集 (sample_id: {sample_id})")
        return 0
    
    # 添加一个新的方法来获取QA数据作为测试集
    def get_qa_test_data(self, 
                        sample_ids: Optional[List[str]] = None,
                        raw_dataset_file: str = "benchmark/dataset/locomo/locomo10.json") -> Dict[str, List[Dict]]:
        """
        获取QA数据作为测试集
        
        Args:
            sample_ids: 指定的对话ID列表，如果为None则获取所有对话的QA
            raw_dataset_file: 原始数据文件路径
            
        Returns:
            QA测试数据 {sample_id: [qa_items]}
        """
        self.logger.info("📋 提取QA数据作为测试集...")
        
        qa_test_data = {}
        
        try:
            with open(raw_dataset_file, 'r', encoding='utf-8') as f:
                raw_dataset = json.load(f)
            
            for sample in raw_dataset:
                sample_id = sample.get('sample_id')
                if not sample_id:
                    continue
                    
                # 如果指定了sample_ids，只处理指定的对话
                if sample_ids and sample_id not in sample_ids:
                    continue
                
                qa_data = sample.get('qa', [])
                if qa_data:
                    qa_test_data[sample_id] = qa_data
                    self.logger.info(f"✅ {sample_id}: 提取了 {len(qa_data)} 个QA对")
            
            total_qa = sum(len(qa_list) for qa_list in qa_test_data.values())
            self.logger.info(f"📊 总计提取 {total_qa} 个QA对作为测试集")
            
            return qa_test_data
            
        except Exception as e:
            self.logger.error(f"❌ 提取QA测试数据失败: {e}")
            return {}
    
    # def _store_qa_data(self, sample: Dict, sample_id: str) -> int:
    #     """存储问答数据"""
    #     qa_data = sample.get('qa', [])
    #     if not qa_data:
    #         return 0
        
    #     stored_count = 0
        
    #     # 将QA数据按类型分组
    #     qa_by_category = {}
    #     qa_adversarial = []
        
    #     for qa_idx, qa_item in enumerate(qa_data):
    #         if isinstance(qa_item, dict):
    #             if 'category' in qa_item:
    #                 category = qa_item['category']
    #                 if category not in qa_by_category:
    #                     qa_by_category[category] = []
    #                 qa_by_category[category].append((qa_idx, qa_item))
    #             elif 'adversarial_answer' in qa_item:
    #                 qa_adversarial.append((qa_idx, qa_item))
        
    #     # 存储分类问答
    #     for category, qa_items in qa_by_category.items():
    #         if qa_items:
    #             category_text = f"Category {category} Questions for {sample_id}:\n"
    #             category_text += f"Total questions in this category: {len(qa_items)}\n"
    #             category_text += "Question indices: " + ", ".join([str(idx) for idx, _ in qa_items])
                
    #             unit = MemoryUnit(
    #                 uid=f"{sample_id}_qa_category_{category}",
    #                 raw_data={
    #                     "text_content": category_text,
    #                     "category": category,
    #                     "question_count": len(qa_items),
    #                     "question_indices": [idx for idx, _ in qa_items],
    #                     "sample_id": sample_id,
    #                     "data_type": "qa_category",
    #                     "data_source": "raw"
    #                 },
    #                 metadata={
    #                     "conversation_id": sample_id,
    #                     "data_layer": "raw",
    #                     "content_type": "qa_category",
    #                     "qa_category": category,
    #                     "created": datetime.now().isoformat()
    #                 }
    #             )
                
    #             self.semantic_graph.add_unit(unit, space_names=[
    #                 f"conversation_{sample_id}",
    #                 f"raw_data_{sample_id}",
    #                 "all_qa"
    #             ])
                
    #             stored_count += 1
        
    #     return stored_count
    
    def _store_extracted_entities(self, sample_data: Dict, sample_id: str) -> int:
        """存储抽取的实体"""
        entities = sample_data.get('entities', [])
        if not entities:
            return 0
        
        stored_count = 0
        
        for entity_idx, entity in enumerate(entities):
            entity_name = entity.get('name', f'entity_{entity_idx}')
            entity_type = entity.get('type', 'unknown')
            description = entity.get('description', '')
            confidence = entity.get('confidence', 0.0)
            source_text = entity.get('source_text', '')
            
            # 创建实体的文本表示
            entity_text = f"Entity: {entity_name}\nType: {entity_type}\nDescription: {description}"
            if source_text:
                entity_text += f"\nSource: {source_text[:200]}..."
            
            # 创建实体记忆单元
            unit = MemoryUnit(
                uid=f"{sample_id}_entity_{entity_idx}_{self._safe_name(entity_name)}",
                raw_data={
                    "text_content": entity_text,
                    "entity_name": entity_name,
                    "entity_type": entity_type,
                    "description": description,
                    "confidence": confidence,
                    "source_text": source_text,
                    "sample_id": sample_id,
                    "data_type": "extracted_entity",
                    "data_source": "extracted"
                },
                metadata={
                    "conversation_id": sample_id,
                    "data_layer": "knowledge",
                    "content_type": "entity",
                    "entity_type": entity_type,
                    "confidence": confidence,
                    "created": datetime.now().isoformat()
                }
            )
            
            # 添加到对话专用空间
            self.semantic_graph.add_unit(unit, space_names=[
                f"conversation_{sample_id}",
                f"extracted_data_{sample_id}",
                "all_entities"
            ])
            
            stored_count += 1
        
        return stored_count
    
    def _store_extracted_relationships(self, sample_data: Dict, sample_id: str) -> int:
        """存储抽取的关系"""
        relationships = sample_data.get('relationships', [])
        if not relationships:
            return 0
        
        stored_count = 0
        
        for rel_idx, relationship in enumerate(relationships):
            source = relationship.get('source', '')
            target = relationship.get('target', '')
            rel_type = relationship.get('type', 'RELATED_TO')
            description = relationship.get('description', '')
            strength = relationship.get('strength', 0.0)
            source_text = relationship.get('source_text', '')
            keywords = relationship.get('keywords', [])
            
            # 创建关系的文本表示
            rel_text = f"Relationship: {source} --[{rel_type}]--> {target}\nDescription: {description}"
            if source_text:
                rel_text += f"\nSource: {source_text[:200]}..."
            if keywords:
                rel_text += f"\nKeywords: {', '.join(keywords)}"
            
            # 创建关系记忆单元
            unit = MemoryUnit(
                uid=f"{sample_id}_rel_{rel_idx}_{self._safe_name(source)}_{self._safe_name(target)}",
                raw_data={
                    "text_content": rel_text,
                    "source_entity": source,
                    "target_entity": target,
                    "relationship_type": rel_type,
                    "description": description,
                    "strength": strength,
                    "keywords": keywords,
                    "source_text": source_text,
                    "sample_id": sample_id,
                    "data_type": "extracted_relationship",
                    "data_source": "extracted"
                },
                metadata={
                    "conversation_id": sample_id,
                    "data_layer": "knowledge",
                    "content_type": "relationship",
                    "relationship_type": rel_type,
                    "strength": strength,
                    "created": datetime.now().isoformat()
                }
            )
            
            # 添加到对话专用空间
            self.semantic_graph.add_unit(unit, space_names=[
                f"conversation_{sample_id}",
                f"extracted_data_{sample_id}",
                "all_relationships"
            ])
            
            stored_count += 1
        
        return stored_count
    
    def _store_keywords(self, sample_data: Dict, sample_id: str) -> int:
        """存储关键词"""
        keywords = sample_data.get('content_keywords', [])
        if not keywords:
            return 0
        
        keyword_text = f"Sample Keywords for {sample_id}: {', '.join(keywords)}"
        
        # 创建关键词记忆单元
        unit = MemoryUnit(
            uid=f"{sample_id}_keywords",
            raw_data={
                "text_content": keyword_text,
                "keywords": keywords,
                "keyword_count": len(keywords),
                "sample_id": sample_id,
                "data_type": "keywords",
                "data_source": "extracted"
            },
            metadata={
                "conversation_id": sample_id,
                "data_layer": "semantic",
                "content_type": "keywords",
                "created": datetime.now().isoformat()
            }
        )
        
        # 添加到对话专用空间
        self.semantic_graph.add_unit(unit, space_names=[
            f"conversation_{sample_id}",
            f"extracted_data_{sample_id}",
            "all_keywords"
        ])
        
        return 1
    
    def _store_statistics(self, sample_data: Dict, sample_id: str) -> int:
        """存储统计信息"""
        extraction_stats = sample_data.get('extraction_statistics', {})
        entity_stats = sample_data.get('entity_statistics', {})
        graph_structure = sample_data.get('graph_structure', {})
        
        if not any([extraction_stats, entity_stats, graph_structure]):
            return 0
        
        stats_text = f"Statistics for {sample_id}:\n"
        stats_text += f"- Entities: {extraction_stats.get('total_entities', 0)}\n"
        stats_text += f"- Relationships: {extraction_stats.get('total_relationships', 0)}\n"
        stats_text += f"- Keywords: {extraction_stats.get('total_keywords', 0)}\n"
        stats_text += f"- Graph Nodes: {graph_structure.get('networkx_nodes', 0)}\n"
        stats_text += f"- Graph Edges: {graph_structure.get('networkx_edges', 0)}\n"
        
        # 创建统计信息记忆单元
        unit = MemoryUnit(
            uid=f"{sample_id}_statistics",
            raw_data={
                "text_content": stats_text,
                "extraction_statistics": extraction_stats,
                "entity_statistics": entity_stats,
                "graph_structure": graph_structure,
                "sample_id": sample_id,
                "data_type": "statistics",
                "data_source": "extracted"
            },
            metadata={
                "conversation_id": sample_id,
                "data_layer": "semantic",
                "content_type": "statistics",
                "created": datetime.now().isoformat()
            }
        )
        
        # 添加到对话专用空间
        self.semantic_graph.add_unit(unit, space_names=[
            f"conversation_{sample_id}",
            f"extracted_data_{sample_id}",
            "all_statistics"
        ])
        
        return 1
    
    def _establish_entity_relationships(self, sample_data: Dict, sample_id: str):
        """在语义图中建立实体间的显式关系"""
        entities = sample_data.get('entities', [])
        relationships = sample_data.get('relationships', [])
        
        if not entities or not relationships:
            return
        
        # 创建实体名称到UID的映射
        entity_name_to_uid = {}
        for entity_idx, entity in enumerate(entities):
            entity_name = entity.get('name', f'entity_{entity_idx}')
            entity_uid = f"{sample_id}_entity_{entity_idx}_{self._safe_name(entity_name)}"
            entity_name_to_uid[entity_name.lower()] = entity_uid
        
        # 建立关系
        for rel_idx, relationship in enumerate(relationships):
            source_name = relationship.get('source', '').lower()
            target_name = relationship.get('target', '').lower()
            rel_type = relationship.get('type', 'RELATED_TO')
            description = relationship.get('description', '')
            strength = relationship.get('strength', 0.0)
            keywords = relationship.get('keywords', [])
            
            # 查找对应的实体UID
            source_uid = entity_name_to_uid.get(source_name)
            target_uid = entity_name_to_uid.get(target_name)
            
            if source_uid and target_uid:
                # 在语义图中添加关系
                self.semantic_graph.add_relationship(
                    source_uid=source_uid,
                    target_uid=target_uid,
                    relationship_name=rel_type,
                    description=description,
                    strength=strength,
                    keywords=keywords,
                    conversation_id=sample_id,
                    created=datetime.now().isoformat()
                )
                self.logger.debug(f"已建立关系: {source_name} --[{rel_type}]--> {target_name}")
    
    def _get_namespace_usage_stats(self) -> Dict[str, int]:
        """获取命名空间使用统计"""
        stats = {}
        
        for space_name, space in self.semantic_graph.semantic_map.memory_spaces.items():
            stats[space_name] = len(space.get_memory_uids())
        
        return stats
    
    def _safe_name(self, name: str) -> str:
        """创建安全的名称"""
        if not name:
            return "unknown"
        safe = "".join(c for c in name if c.isalnum() or c in "._-")
        return safe[:50]
    
    def get_conversation_list(self,
                             raw_dataset_file: str = "benchmark/dataset/locomo/locomo10.json",
                             extracted_dataset_file: str = "benchmark/dataset/locomo/extraction/locomo_extracted_full_dataset.json") -> List[str]:
        """获取所有可用的对话ID列表"""
        all_sample_ids = set()
        
        # 从原始数据获取
        try:
            with open(raw_dataset_file, 'r', encoding='utf-8') as f:
                raw_dataset = json.load(f)
            for sample in raw_dataset:
                sample_id = sample.get('sample_id')
                if sample_id:
                    all_sample_ids.add(sample_id)
        except Exception as e:
            self.logger.warning(f"无法加载原始数据: {e}")
        
        # 从抽取数据获取
        try:
            with open(extracted_dataset_file, 'r', encoding='utf-8') as f:
                extracted_dataset = json.load(f)
            extracted_samples = extracted_dataset.get("samples", {})
            all_sample_ids.update(extracted_samples.keys())
        except Exception as e:
            self.logger.warning(f"无法加载抽取数据: {e}")
        
        return sorted(list(all_sample_ids))


class ConversationSemanticQuerier:
    """
    对话语义查询器 - 支持按对话进行RAG检索
    """
    
    def __init__(self, semantic_graph: SemanticGraph):
        """初始化查询器"""
        self.semantic_graph = semantic_graph
        self.logger = logging.getLogger(__name__)
    
    def query_conversation(self, 
                          query_text: str,
                          conversation_id: str,
                          data_sources: List[str] = None,
                          data_types: List[str] = None,
                          k: int = 5) -> Dict[str, Any]:
        """
        查询指定对话
        
        Args:
            query_text: 查询文本
            conversation_id: 对话ID (如 "conv-26")
            data_sources: 数据源过滤 ("raw", "extracted")
            data_types: 数据类型过滤
            k: 返回结果数量
            
        Returns:
            查询结果
        """
        results = {
            "query": query_text,
            "conversation_id": conversation_id,
            "results": []
        }
        
        try:
            # 在对话专用空间中搜索
            space_name = f"conversation_{conversation_id}"
            
            namespace_results = self.semantic_graph.search_similarity_in_graph(
                query_text=query_text,
                k=k * 2,
                space_name=space_name
            )
            
            # 按数据源过滤
            if data_sources:
                filtered_results = []
                for unit, score in namespace_results:
                    unit_data_source = unit.raw_data.get('data_source', '')
                    if unit_data_source in data_sources:
                        filtered_results.append((unit, score))
                namespace_results = filtered_results
            
            # 按数据类型过滤
            if data_types:
                filtered_results = []
                for unit, score in namespace_results:
                    unit_data_type = unit.raw_data.get('data_type', '')
                    if unit_data_type in data_types:
                        filtered_results.append((unit, score))
                namespace_results = filtered_results
            
            # 格式化结果
            formatted_results = [
                {
                    "unit_id": unit.uid,
                    "content": unit.raw_data.get("text_content", "")[:500],
                    "data_type": unit.raw_data.get("data_type", "unknown"),
                    "data_source": unit.raw_data.get("data_source", "unknown"),
                    "similarity_score": float(score),
                    "metadata": unit.metadata
                }
                for unit, score in namespace_results[:k]
            ]
            
            results["results"] = formatted_results
            
        except Exception as e:
            self.logger.error(f"查询对话 {conversation_id} 失败: {e}")
        
        return results
    
    def query_all_conversations(self, 
                               query_text: str,
                               data_sources: List[str] = None,
                               data_types: List[str] = None,
                               k: int = 5) -> Dict[str, Any]:
        """
        查询所有对话
        
        Args:
            query_text: 查询文本
            data_sources: 数据源过滤
            data_types: 数据类型过滤
            k: 返回结果数量
            
        Returns:
            查询结果
        """
        results = {
            "query": query_text,
            "conversation_results": {}
        }
        
        # 获取所有对话空间
        conversation_spaces = [
            space_name for space_name in self.semantic_graph.semantic_map.memory_spaces.keys()
            if space_name.startswith("conversation_")
        ]
        
        for space_name in conversation_spaces:
            conversation_id = space_name.replace("conversation_", "")
            conv_results = self.query_conversation(
                query_text=query_text,
                conversation_id=conversation_id,
                data_sources=data_sources,
                data_types=data_types,
                k=k
            )
            
            if conv_results["results"]:
                results["conversation_results"][conversation_id] = conv_results
        
        return results


# def main():
#     """主函数 - 演示对话存储和查询"""
    
#     # 1. 创建对话存储器
#     storage = ConversationSemanticStorage()
    
#     # 2. 获取可用对话列表
#     conversations = storage.get_conversation_list()
#     print(f"🔍 发现 {len(conversations)} 个对话: {conversations}")
    
#     # 3. 存储指定对话
#     test_conversation = "conv-26"
#     print(f"\n🚀 存储对话 {test_conversation}")
    
#     stats = storage.store_conversation(
#         sample_id=test_conversation,
#         include_raw=True,
#         include_extracted=True
#     )
    
#     print(f"✅ {test_conversation} 存储完成！")
#     print(f"存储统计: {stats['storage_breakdown']}")
#     print(f"命名空间使用: {stats['namespace_usage']}")
    
#     # 4. 创建查询器并测试
#     querier = ConversationSemanticQuerier(storage.semantic_graph)
    
#     print(f"\n🔍 查询对话 {test_conversation}")
    
#     # 查询原始数据
#     raw_results = querier.query_conversation(
#         query_text="Caroline LGBTQ support",
#         conversation_id=test_conversation,
#         data_sources=["raw"],
#         k=3
#     )
    
#     print(f"原始数据查询结果: {len(raw_results['results'])} 条")
#     for result in raw_results['results'][:2]:
#         print(f"- [{result['similarity_score']:.3f}] {result['data_type']}: {result['content'][:80]}...")
    
#     # 查询抽取结果
#     extracted_results = querier.query_conversation(
#         query_text="Caroline",
#         conversation_id=test_conversation,
#         data_sources=["extracted"],
#         data_types=["extracted_entity"],
#         k=3
#     )
    
#     print(f"\n抽取结果查询: {len(extracted_results['results'])} 条")
#     for result in extracted_results['results'][:2]:
#         print(f"- [{result['similarity_score']:.3f}] {result['data_type']}: {result['content'][:80]}...")
    
#     print(f"\n🎯 演示特定对话存储和查询完成！")
    
#     # 5. 可选：存储所有对话
#     choice = input("\n是否存储所有对话？(y/n): ")
#     if choice.lower() == 'y':
#         print("\n🚀 开始存储所有对话")
#         all_stats = storage.store_all_conversations()
#         print(f"✅ 所有对话存储完成！处理了 {all_stats['processed_conversations']}/{all_stats['total_conversations']} 个对话")
#         print(f"总体存储统计: {all_stats['overall_storage_breakdown']}")

# 更新main函数
def main():
    """主函数 - 演示对话存储和QA测试集提取"""
    
    # 1. 创建对话存储器
    storage = ConversationSemanticStorage()
    
    # 2. 获取可用对话列表
    conversations = storage.get_conversation_list()
    print(f"🔍 发现 {len(conversations)} 个对话: {conversations}")
    
    # 3. 存储指定对话（不包括QA）
    test_conversation = "conv-26"
    print(f"\n🚀 存储对话 {test_conversation} (不包括QA)")
    
    stats = storage.store_conversation(
        sample_id=test_conversation,
        include_raw=True,
        include_extracted=True
    )
    
    print(f"✅ {test_conversation} 存储完成！")
    print(f"存储统计: {stats['storage_breakdown']}")
    print(f"QA信息: {stats['qa_info']}")
    print(f"命名空间使用: {stats['namespace_usage']}")
    
    # 4. 提取QA数据作为测试集
    print(f"\n📋 提取QA数据作为测试集...")
    qa_test_data = storage.get_qa_test_data([test_conversation])
    
    if test_conversation in qa_test_data:
        qa_count = len(qa_test_data[test_conversation])
        print(f"📊 {test_conversation} 的QA测试集包含 {qa_count} 个问答对")
        
        # 显示前几个QA样例
        for i, qa in enumerate(qa_test_data[test_conversation][:3]):
            category = qa.get('category', 'unknown')
            question = qa.get('question', 'N/A')[:50] + '...' if len(qa.get('question', '')) > 50 else qa.get('question', 'N/A')
            print(f"  QA[{i}] (cat-{category}): {question}")
    
    # 5. 创建查询器并测试检索
    print(f"\n🔍 测试语义检索功能...")
    querier = ConversationSemanticQuerier(storage.semantic_graph)
    
    # 在原始数据中搜索
    search_results = querier.query_conversation(
        query_text="Caroline LGBTQ support",
        conversation_id=test_conversation,
        data_sources=["raw"],
        k=3
    )
    
    print(f"原始数据检索结果: {len(search_results['results'])} 条")
    for i, result in enumerate(search_results['results'][:2]):
        content_preview = result['content'][:60] + '...' if len(result['content']) > 60 else result['content']
        print(f"  [{i}] [{result['similarity_score']:.3f}] {result['data_type']}: {content_preview}")
    
    print(f"\n🎯 演示完成！")
    print(f"✅ 对话数据已存储到语义图谱（不包括QA）")
    print(f"✅ QA数据已准备好作为测试集使用")
    
    # 6. 可选：存储所有对话
    choice = input("\n是否存储所有对话？(y/n): ")
    if choice.lower() == 'y':
        print("\n🚀 开始存储所有对话（不包括QA）")
        all_stats = storage.store_all_conversations()
        print(f"✅ 所有对话存储完成！处理了 {all_stats['processed_conversations']}/{all_stats['total_conversations']} 个对话")
        print(f"📊 总体存储统计: {all_stats['overall_storage_breakdown']}")
        print(f"📋 总计QA测试集: {all_stats['overall_qa_info']['total_qa_pairs']} 个问答对")

if __name__ == "__main__":
    main()

# # 只存储原始数据
# storage.store_conversation("conv-26", include_raw=True, include_extracted=False)

# # 只存储抽取结果
# storage.store_conversation("conv-26", include_raw=False, include_extracted=True)

# # 存储两种数据
# storage.store_conversation("conv-26", include_raw=True, include_extracted=True)

# # 查询指定对话
# querier.query_conversation(
#     query_text="Caroline LGBTQ",
#     conversation_id="conv-26",
#     data_sources=["raw", "extracted"],
#     k=5
# )

# # 查询所有对话
# querier.query_all_conversations(
#     query_text="LGBTQ support",
#     data_sources=["raw"],
#     k=5
# )

# 1. 存储特定对话
storage = ConversationSemanticStorage()
storage.store_conversation("conv-26")  # 只存储 conv-26

# 2. 存储多个对话
for conv_id in ["conv-26", "conv-30", "conv-41"]:
    storage.store_conversation(conv_id)

# # 3. 查询特定对话
# querier = ConversationSemanticQuerier(storage.semantic_graph)
# results = querier.query_conversation("Caroline", "conv-26")