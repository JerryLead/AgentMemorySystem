import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass
import numpy as np

@dataclass
class CacheAnalysisData:
    """缓存分析数据结构"""
    uid: str
    access_count: int
    last_access_time: float
    content_summary: str
    semantic_tags: List[str]
    related_units: List[str]
    space_memberships: List[str]
    importance_score: float = 0.0

class LLMCacheAdvisor:
    """基于大模型的缓存决策顾问"""
    
    def __init__(self, 
                 llm_client=None,  # 可以是OpenAI、DeepSeek等的客户端
                 model_name: str = "gpt-4",
                 max_context_units: int = 50):
        """
        初始化LLM缓存顾问
        
        参数:
            llm_client: 大模型客户端
            model_name: 模型名称
            max_context_units: 最大上下文单元数量
        """
        self.llm_client = llm_client
        self.model_name = model_name
        self.max_context_units = max_context_units
        
        # 缓存访问历史
        self.access_history: List[Dict] = []
        self.semantic_clusters: Dict[str, List[str]] = {}
        
    def analyze_cache_context(self, 
                            memory_units: Dict[str, Any],
                            access_counts: Dict[str, int],
                            last_accessed: Dict[str, float],
                            current_query_context: Optional[str] = None) -> List[CacheAnalysisData]:
        """
        分析当前缓存上下文，生成分析数据
        """
        analysis_data = []
        current_time = datetime.now().timestamp()
        
        for uid, unit in memory_units.items():
            # 提取内容摘要
            content_summary = self._extract_content_summary(unit)
            
            # 计算重要性分数（基于访问频率和时间衰减）
            access_count = access_counts.get(uid, 0)
            last_access = last_accessed.get(uid, 0)
            time_decay = self._calculate_time_decay(last_access, current_time)
            importance_score = access_count * time_decay
            
            # 获取语义标签
            semantic_tags = self._extract_semantic_tags(unit)
            
            # 查找相关单元
            related_units = self._find_related_units(uid, memory_units)
            
            # 获取空间归属
            space_memberships = getattr(unit, 'spaces', [])
            
            analysis_data.append(CacheAnalysisData(
                uid=uid,
                access_count=access_count,
                last_access_time=last_access,
                content_summary=content_summary,
                semantic_tags=semantic_tags,
                related_units=related_units,
                space_memberships=space_memberships,
                importance_score=importance_score
            ))
        
        return sorted(analysis_data, key=lambda x: x.importance_score, reverse=True)
    
    def recommend_eviction(self,
                          analysis_data: List[CacheAnalysisData],
                          eviction_count: int,
                          current_query_context: Optional[str] = None,
                          recent_queries: Optional[List[str]] = None) -> List[str]:
        """
        使用大模型推荐要换出的单元
        """
        if not self.llm_client:
            logging.warning("未配置LLM客户端，使用基础算法")
            return self._fallback_eviction(analysis_data, eviction_count)
        
        try:
            # 准备上下文数据
            context_data = self._prepare_llm_context(
                analysis_data, 
                eviction_count,
                current_query_context,
                recent_queries
            )
            
            # 构建提示词
            prompt = self._build_eviction_prompt(context_data, eviction_count)
            
            # 调用大模型
            response = self._call_llm(prompt)
            
            # 解析响应
            recommended_uids = self._parse_llm_response(response, analysis_data)
            
            # 验证推荐结果
            validated_uids = self._validate_recommendations(
                recommended_uids, analysis_data, eviction_count
            )
            
            logging.info(f"LLM推荐换出 {len(validated_uids)} 个单元: {validated_uids}")
            return validated_uids
            
        except Exception as e:
            logging.error(f"LLM缓存决策失败: {e}，使用备用算法")
            return self._fallback_eviction(analysis_data, eviction_count)
    
    def recommend_prefetch(self,
                          current_query: str,
                          available_units: List[str],
                          prefetch_count: int = 5) -> List[str]:
        """
        基于当前查询推荐要预取的单元
        """
        if not self.llm_client:
            return []
            
        try:
            prompt = self._build_prefetch_prompt(current_query, available_units, prefetch_count)
            response = self._call_llm(prompt)
            recommended_uids = self._parse_prefetch_response(response, available_units)
            
            return recommended_uids[:prefetch_count]
            
        except Exception as e:
            logging.error(f"LLM预取推荐失败: {e}")
            return []
    
    def _extract_content_summary(self, unit) -> str:
        """提取内容摘要"""
        try:
            raw_data = getattr(unit, 'raw_data', {})
            if 'text_content' in raw_data:
                content = raw_data['text_content']
                return content[:200] + "..." if len(content) > 200 else content
            elif 'description' in raw_data:
                return raw_data['description']
            else:
                # 提取其他有意义的字段
                meaningful_content = []
                for key, value in raw_data.items():
                    if isinstance(value, str) and len(value) > 10:
                        meaningful_content.append(f"{key}: {value[:50]}")
                return "; ".join(meaningful_content[:3])
        except:
            return f"Unit {getattr(unit, 'uid', 'unknown')}"
    
    def _extract_semantic_tags(self, unit) -> List[str]:
        """提取语义标签"""
        tags = []
        try:
            raw_data = getattr(unit, 'raw_data', {})
            if 'type' in raw_data:
                tags.append(raw_data['type'])
            if 'field' in raw_data:
                tags.append(raw_data['field'])
            if 'category' in raw_data:
                tags.append(raw_data['category'])
        except:
            pass
        return tags
    
    def _find_related_units(self, uid: str, memory_units: Dict) -> List[str]:
        """查找相关单元（这里简化实现）"""
        # 在实际实现中，可以使用语义相似度或图关系
        return []
    
    def _calculate_time_decay(self, last_access: float, current_time: float) -> float:
        """计算时间衰减因子"""
        if last_access == 0:
            return 0.1  # 从未访问的权重很低
        
        hours_since_access = (current_time - last_access) / 3600
        # 使用指数衰减
        return np.exp(-hours_since_access / 24)  # 24小时半衰期
    
    def _prepare_llm_context(self,
                           analysis_data: List[CacheAnalysisData],
                           eviction_count: int,
                           current_query_context: Optional[str],
                           recent_queries: Optional[List[str]]) -> Dict:
        """准备LLM上下文数据"""
        # 选择最相关的单元进行分析
        selected_units = analysis_data[:self.max_context_units]
        
        context = {
            "eviction_count": eviction_count,
            "total_units": len(analysis_data),
            "current_query": current_query_context,
            "recent_queries": recent_queries or [],
            "units": []
        }
        
        for unit_data in selected_units:
            context["units"].append({
                "uid": unit_data.uid,
                "access_count": unit_data.access_count,
                "last_access_hours_ago": (datetime.now().timestamp() - unit_data.last_access_time) / 3600,
                "content_summary": unit_data.content_summary,
                "semantic_tags": unit_data.semantic_tags,
                "spaces": unit_data.space_memberships,
                "importance_score": unit_data.importance_score
            })
        
        return context
    
    def _build_eviction_prompt(self, context_data: Dict, eviction_count: int) -> str:
        """构建换出决策提示词"""
        prompt = f"""
            你是一个智能缓存管理专家。现在需要从内存中换出 {eviction_count} 个数据单元来释放空间。

            ## 当前上下文
            - 当前查询: {context_data.get('current_query', '无')}
            - 最近查询: {context_data.get('recent_queries', [])}
            - 总单元数: {context_data['total_units']}

            ## 内存单元信息 (按重要性排序)
            """
        
        for i, unit in enumerate(context_data['units'][:20]):  # 只显示前20个
            prompt += f"""
            {i+1}. UID: {unit['uid']}
            - 访问次数: {unit['access_count']}
            - 上次访问: {unit['last_access_hours_ago']:.1f}小时前
            - 重要性分数: {unit['importance_score']:.3f}
            - 内容摘要: {unit['content_summary']}
            - 语义标签: {unit['semantic_tags']}
            - 所属空间: {unit['spaces']}
            """
                    
            prompt += f"""

            ## 决策原则
            1. 优先换出访问频率低、最近未访问的单元
            2. 考虑语义相关性，避免换出与当前查询相关的单元
            3. 保持语义聚类的完整性
            4. 考虑数据的长期价值

            请基于以上信息，推荐 {eviction_count} 个最适合换出的单元UID。

            ## 输出格式
            请以JSON格式返回，包含推荐换出的UID列表和理由：
            {{
                "recommended_uids": ["uid1", "uid2", ...],
                "reasoning": "换出这些单元的主要理由"
            }}
            """
        return prompt
    
    def _build_prefetch_prompt(self, current_query: str, available_units: List[str], prefetch_count: int) -> str:
        """构建预取推荐提示词"""
        prompt = f"""
        你是一个智能预取推荐专家。用户刚发起了查询，需要预测他们接下来可能需要的数据。

        ## 当前查询
        {current_query}

        ## 可预取的单元UID
        {available_units[:50]}  # 限制显示数量

        请基于当前查询内容，推荐 {prefetch_count} 个最可能在后续查询中用到的单元。

        ## 输出格式
        {{
            "recommended_uids": ["uid1", "uid2", ...],
            "reasoning": "推荐理由"
        }}
        """
        return prompt
    
    def _call_llm(self, prompt: str) -> str:
        """调用大模型"""
        if hasattr(self.llm_client, 'chat'):
            # OpenAI风格的调用
            response = self.llm_client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            return response.choices[0].message.content
        else:
            # 其他客户端的调用方式
            return self.llm_client.generate(prompt)
    
    def _parse_llm_response(self, response: str, analysis_data: List[CacheAnalysisData]) -> List[str]:
        """解析LLM响应"""
        try:
            # 尝试解析JSON
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return result.get('recommended_uids', [])
        except:
            pass
        
        # 如果JSON解析失败，尝试从文本中提取UID
        valid_uids = {data.uid for data in analysis_data}
        found_uids = []
        for uid in valid_uids:
            if uid in response:
                found_uids.append(uid)
        
        return found_uids
    
    def _parse_prefetch_response(self, response: str, available_units: List[str]) -> List[str]:
        """解析预取推荐响应"""
        return self._parse_llm_response(response, [CacheAnalysisData(uid, 0, 0, "", [], [], []) 
                                                 for uid in available_units])
    
    def _validate_recommendations(self, 
                                recommended_uids: List[str],
                                analysis_data: List[CacheAnalysisData],
                                eviction_count: int) -> List[str]:
        """验证推荐结果"""
        valid_uids = {data.uid for data in analysis_data}
        validated = [uid for uid in recommended_uids if uid in valid_uids]
        
        # 如果推荐数量不足，补充一些低重要性的单元
        if len(validated) < eviction_count:
            additional_needed = eviction_count - len(validated)
            remaining_units = [data for data in analysis_data if data.uid not in validated]
            # 按重要性升序排序，选择最不重要的
            remaining_units.sort(key=lambda x: x.importance_score)
            additional_uids = [data.uid for data in remaining_units[:additional_needed]]
            validated.extend(additional_uids)
        
        return validated[:eviction_count]
    
    def _fallback_eviction(self, analysis_data: List[CacheAnalysisData], eviction_count: int) -> List[str]:
        """备用换出算法"""
        # 使用改进的LRU算法
        sorted_data = sorted(analysis_data, key=lambda x: (x.last_access_time, x.access_count))
        return [data.uid for data in sorted_data[:eviction_count]]
    
    def record_access(self, uid: str, query_context: Optional[str] = None):
        """记录访问历史"""
        self.access_history.append({
            "uid": uid,
            "timestamp": datetime.now().timestamp(),
            "query_context": query_context
        })
        
        # 保持历史记录在合理范围内
        if len(self.access_history) > 1000:
            self.access_history = self.access_history[-500:]