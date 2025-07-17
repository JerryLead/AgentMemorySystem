# neo4j_operator.py
"""
Neo4j操作类 - 完整支持SemanticGraph关系管理功能
提供图数据库存储、查询、关系管理、图遍历等完整接口
"""

import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from neo4j import GraphDatabase, Driver, Session, Transaction
from neo4j.exceptions import ServiceUnavailable, TransientError
import numpy as np

from core.memory_unit import MemoryUnit
from core.memory_space import MemorySpace
from core.milvus_operator import MilvusOperator

class Neo4jOperator:
    """
    增强版Neo4j操作类，完整支持SemanticGraph关系管理功能
    提供节点存储、关系管理、图遍历、复杂查询等功能
    集成Milvus支持混合存储架构
    """
    
    def __init__(
        self,
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_user: str = "neo4j",
        neo4j_password: str = "password",
        neo4j_database: str = "neo4j",
        # Milvus集成参数
        milvus_host: str = "localhost",
        milvus_port: str = "19530",
        milvus_user: str = "",
        milvus_password: str = "",
        milvus_collection: str = "semantic_memory_units",
        embedding_dim: int = 512,
        # 连接配置
        max_connection_pool_size: int = 50,
        connection_timeout: int = 30,
        max_retry_attempts: int = 3
    ):
        """
        初始化Neo4j操作类
        
        Args:
            neo4j_uri: Neo4j数据库URI
            neo4j_user: Neo4j用户名
            neo4j_password: Neo4j密码
            neo4j_database: Neo4j数据库名
            milvus_*: Milvus连接参数
            embedding_dim: 向量维度
            max_connection_pool_size: 最大连接池大小
            connection_timeout: 连接超时时间
            max_retry_attempts: 最大重试次数
        """
        # Neo4j连接配置
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.neo4j_database = neo4j_database
        self.max_retry_attempts = max_retry_attempts
        
        # 连接状态
        self.driver: Optional[Driver] = None
        self.neo4j_connected = False
        
        # Milvus集成
        self.milvus_operator: Optional[MilvusOperator] = None
        self.milvus_enabled = False
        
        # 性能统计
        self.query_stats = {
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "total_query_time": 0.0
        }
        
        # 初始化连接
        self._connect_neo4j(max_connection_pool_size, connection_timeout)
        
        # 初始化Milvus（可选）
        if milvus_host and milvus_port:
            self._connect_milvus(
                milvus_host, milvus_port, milvus_user, milvus_password,
                milvus_collection, embedding_dim
            )
        
        # 初始化数据库结构
        if self.neo4j_connected:
            self._initialize_database()

    def _connect_neo4j(self, max_pool_size: int, timeout: int) -> bool:
        """建立Neo4j连接"""
        try:
            self.driver = GraphDatabase.driver(
                self.neo4j_uri,
                auth=(self.neo4j_user, self.neo4j_password),
                max_connection_pool_size=max_pool_size,
                connection_timeout=timeout,
                max_transaction_retry_time=30
            )
            
            # 测试连接
            with self.driver.session(database=self.neo4j_database) as session:
                result = session.run("RETURN 1 as test")
                test_value = result.single()["test"]
                
            if test_value == 1:
                self.neo4j_connected = True
                logging.info(f"成功连接到Neo4j数据库: {self.neo4j_uri}")
                return True
            else:
                raise Exception("连接测试失败")
                
        except Exception as e:
            logging.error(f"连接到Neo4j数据库失败: {e}")
            self.neo4j_connected = False
            return False

    def _connect_milvus(
        self, host: str, port: str, user: str, password: str, 
        collection: str, embedding_dim: int
    ) -> bool:
        """连接到Milvus向量数据库"""
        try:
            self.milvus_operator = MilvusOperator(
                host=host,
                port=port,
                user=user,
                password=password,
                collection_name=collection,
                embedding_dim=embedding_dim
            )
            
            if self.milvus_operator.is_connected:
                self.milvus_enabled = True
                logging.info(f"成功连接到Milvus数据库: {host}:{port}")
                return True
            else:
                self.milvus_operator = None
                return False
                
        except Exception as e:
            logging.error(f"连接到Milvus数据库失败: {e}")
            self.milvus_operator = None
            return False

    def _initialize_database(self):
        """初始化数据库结构（创建索引和约束）"""
        initialization_queries = [
            # 创建唯一性约束
            "CREATE CONSTRAINT memory_unit_uid IF NOT EXISTS FOR (n:MemoryUnit) REQUIRE n.uid IS UNIQUE",
            "CREATE CONSTRAINT memory_space_name IF NOT EXISTS FOR (n:MemorySpace) REQUIRE n.name IS UNIQUE",
            
            # 创建索引以优化查询性能
            "CREATE INDEX memory_unit_created IF NOT EXISTS FOR (n:MemoryUnit) ON (n.created_at)",
            "CREATE INDEX memory_unit_type IF NOT EXISTS FOR (n:MemoryUnit) ON (n.content_type)",
            "CREATE INDEX memory_space_created IF NOT EXISTS FOR (n:MemorySpace) ON (n.created_at)",
            "CREATE INDEX relationship_type IF NOT EXISTS FOR ()-[r]-() ON (type(r))",
            
            # 创建全文搜索索引
            "CREATE FULLTEXT INDEX memory_unit_fulltext IF NOT EXISTS FOR (n:MemoryUnit) ON EACH [n.content, n.title, n.description]"
        ]
        
        for query in initialization_queries:
            try:
                self._execute_write_query(query)
                logging.debug(f"数据库初始化查询执行成功: {query[:50]}...")
            except Exception as e:
                logging.warning(f"数据库初始化查询执行失败: {e}")

    # ==============================
    # 基础查询执行方法
    # ==============================

    def _execute_read_query(
        self, 
        query: str, 
        parameters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """执行读查询"""
        if not self.neo4j_connected:
            logging.error("Neo4j未连接，无法执行查询")
            return []
        
        start_time = time.time()
        retry_count = 0
        
        while retry_count < self.max_retry_attempts:
            try:
                with self.driver.session(database=self.neo4j_database) as session:
                    result = session.run(query, parameters or {})
                    records = [record.data() for record in result]
                
                # 更新统计信息
                self.query_stats["total_queries"] += 1
                self.query_stats["successful_queries"] += 1
                self.query_stats["total_query_time"] += time.time() - start_time
                
                return records
                
            except (ServiceUnavailable, TransientError) as e:
                retry_count += 1
                if retry_count >= self.max_retry_attempts:
                    logging.error(f"读查询重试{retry_count}次后仍失败: {e}")
                    break
                logging.warning(f"读查询失败，重试 {retry_count}/{self.max_retry_attempts}: {e}")
                time.sleep(2 ** retry_count)  # 指数退避
            except Exception as e:
                logging.error(f"读查询执行失败: {e}")
                break
        
        # 更新失败统计
        self.query_stats["total_queries"] += 1
        self.query_stats["failed_queries"] += 1
        return []

    def _execute_write_query(
        self, 
        query: str, 
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """执行写查询"""
        if not self.neo4j_connected:
            logging.error("Neo4j未连接，无法执行写查询")
            return {}
        
        start_time = time.time()
        retry_count = 0
        
        while retry_count < self.max_retry_attempts:
            try:
                with self.driver.session(database=self.neo4j_database) as session:
                    result = session.execute_write(
                        lambda tx: tx.run(query, parameters or {}).data()
                    )
                
                # 更新统计信息
                self.query_stats["total_queries"] += 1
                self.query_stats["successful_queries"] += 1
                self.query_stats["total_query_time"] += time.time() - start_time
                
                return {"success": True, "result": result}
                
            except (ServiceUnavailable, TransientError) as e:
                retry_count += 1
                if retry_count >= self.max_retry_attempts:
                    logging.error(f"写查询重试{retry_count}次后仍失败: {e}")
                    break
                logging.warning(f"写查询失败，重试 {retry_count}/{self.max_retry_attempts}: {e}")
                time.sleep(2 ** retry_count)
            except Exception as e:
                logging.error(f"写查询执行失败: {e}")
                break
        
        # 更新失败统计
        self.query_stats["total_queries"] += 1
        self.query_stats["failed_queries"] += 1
        return {"success": False}

    # ==============================
    # 记忆单元(节点)操作接口
    # ==============================

    def add_unit(
        self, 
        unit: MemoryUnit, 
        labels: Optional[List[str]] = None,
        properties: Optional[Dict[str, Any]] = None,
        sync_to_milvus: bool = True
    ) -> bool:
        """
        添加记忆单元作为节点到Neo4j
        
        Args:
            unit: MemoryUnit对象
            labels: 额外的节点标签
            properties: 额外的节点属性
            sync_to_milvus: 是否同步到Milvus
            
        Returns:
            bool: 操作是否成功
        """
        try:
            # 准备节点标签
            node_labels = ["MemoryUnit"]
            if labels:
                node_labels.extend(labels)
            labels_str = ":".join(node_labels)
            
            # 准备节点属性
            node_properties = {
                "uid": unit.uid,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "has_embedding": unit.embedding is not None,
                "content_type": self._infer_content_type(unit.raw_data)
            }
            
            # 添加raw_data中的可索引字段
            for key, value in unit.raw_data.items():
                if isinstance(value, (str, int, float, bool)):
                    node_properties[f"raw_{key}"] = value
                elif isinstance(value, (list, dict)):
                    # 复杂数据类型转为JSON字符串
                    node_properties[f"raw_{key}"] = json.dumps(value, ensure_ascii=False)
            
            # 添加metadata
            if unit.metadata:
                for key, value in unit.metadata.items():
                    if isinstance(value, (str, int, float, bool)):
                        node_properties[f"meta_{key}"] = value
            
            # 添加额外属性
            if properties:
                node_properties.update(properties)
            
            # 构建Cypher查询
            query = f"""
            MERGE (n:{labels_str} {{uid: $uid}})
            SET n += $properties
            RETURN n.uid as uid
            """
            
            result = self._execute_write_query(query, {
                "uid": unit.uid,
                "properties": node_properties
            })
            
            if result.get("success"):
                logging.debug(f"成功添加记忆单元节点: {unit.uid}")
                
                # 同步到Milvus
                if sync_to_milvus and self.milvus_enabled:
                    try:
                        self.milvus_operator.add_unit(unit)
                        logging.debug(f"记忆单元 {unit.uid} 已同步到Milvus")
                    except Exception as e:
                        logging.warning(f"同步记忆单元到Milvus失败: {e}")
                
                return True
            else:
                logging.error(f"添加记忆单元节点失败: {unit.uid}")
                return False
                
        except Exception as e:
            logging.error(f"添加记忆单元 {unit.uid} 时出错: {e}")
            return False

    def get_unit(self, unit_id: str) -> Optional[MemoryUnit]:
        """
        从Neo4j获取记忆单元
        
        Args:
            unit_id: 记忆单元ID
            
        Returns:
            Optional[MemoryUnit]: MemoryUnit对象或None
        """
        query = """
        MATCH (n:MemoryUnit {uid: $uid})
        RETURN n
        """
        
        results = self._execute_read_query(query, {"uid": unit_id})
        
        if not results:
            return None
        
        node_data = results[0]["n"]
        return self._parse_node_to_memory_unit(node_data)

    def get_units_batch(self, unit_ids: List[str]) -> List[MemoryUnit]:
        """
        批量获取记忆单元
        
        Args:
            unit_ids: 记忆单元ID列表
            
        Returns:
            List[MemoryUnit]: MemoryUnit对象列表
        """
        if not unit_ids:
            return []
        
        query = """
        MATCH (n:MemoryUnit)
        WHERE n.uid IN $uids
        RETURN n
        """
        
        results = self._execute_read_query(query, {"uids": unit_ids})
        
        units = []
        for record in results:
            try:
                unit = self._parse_node_to_memory_unit(record["n"])
                if unit:
                    units.append(unit)
            except Exception as e:
                logging.error(f"解析记忆单元节点失败: {e}")
                continue
        
        return units

    def delete_unit(self, unit_id: str, delete_relationships: bool = True) -> bool:
        """
        删除记忆单元节点
        
        Args:
            unit_id: 记忆单元ID
            delete_relationships: 是否同时删除相关关系
            
        Returns:
            bool: 删除是否成功
        """
        try:
            if delete_relationships:
                # 删除节点及其所有关系
                query = """
                MATCH (n:MemoryUnit {uid: $uid})
                DETACH DELETE n
                """
            else:
                # 只删除节点（如果有关系会失败）
                query = """
                MATCH (n:MemoryUnit {uid: $uid})
                DELETE n
                """
            
            result = self._execute_write_query(query, {"uid": unit_id})
            
            if result.get("success"):
                logging.debug(f"成功删除记忆单元节点: {unit_id}")
                
                # 从Milvus删除
                if self.milvus_enabled:
                    try:
                        self.milvus_operator.delete_unit(unit_id)
                        logging.debug(f"记忆单元 {unit_id} 已从Milvus删除")
                    except Exception as e:
                        logging.warning(f"从Milvus删除记忆单元失败: {e}")
                
                return True
            else:
                return False
                
        except Exception as e:
            logging.error(f"删除记忆单元 {unit_id} 时出错: {e}")
            return False

    def unit_exists(self, unit_id: str) -> bool:
        """
        检查记忆单元是否存在
        
        Args:
            unit_id: 记忆单元ID
            
        Returns:
            bool: 单元是否存在
        """
        query = """
        MATCH (n:MemoryUnit {uid: $uid})
        RETURN count(n) as count
        """
        
        results = self._execute_read_query(query, {"uid": unit_id})
        return results[0]["count"] > 0 if results else False

    def ensure_node_exists(self, node_id: str, node_type: str = "MemoryUnit") -> bool:
        """
        确保节点存在，如果不存在则创建占位符节点
        
        Args:
            node_id: 节点ID
            node_type: 节点类型
            
        Returns:
            bool: 操作是否成功
        """
        query = f"""
        MERGE (n:{node_type} {{uid: $node_id}})
        ON CREATE SET n.created_at = datetime(), n.placeholder = true
        RETURN n.uid as uid
        """
        
        result = self._execute_write_query(query, {"node_id": node_id})
        return result.get("success", False)

    # ==============================
    # 记忆空间操作接口
    # ==============================

    def add_memory_space(
        self, 
        space: MemorySpace,
        properties: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        添加记忆空间作为节点到Neo4j
        
        Args:
            space: MemorySpace对象
            properties: 额外的节点属性
            
        Returns:
            bool: 操作是否成功
        """
        try:
            # 准备节点属性
            node_properties = {
                "name": space.name,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "unit_count": len(space.get_unit_uids()),
                "child_space_count": len(space.get_child_space_names())
            }
            
            if properties:
                node_properties.update(properties)
            
            query = """
            MERGE (s:MemorySpace {name: $name})
            SET s += $properties
            RETURN s.name as name
            """
            
            result = self._execute_write_query(query, {
                "name": space.name,
                "properties": node_properties
            })
            
            return result.get("success", False)
            
        except Exception as e:
            logging.error(f"添加记忆空间 {space.name} 时出错: {e}")
            return False

    def get_memory_space_info(self, space_name: str) -> Optional[Dict[str, Any]]:
        """
        获取记忆空间信息
        
        Args:
            space_name: 空间名称
            
        Returns:
            Optional[Dict[str, Any]]: 空间信息或None
        """
        query = """
        MATCH (s:MemorySpace {name: $name})
        RETURN s
        """
        
        results = self._execute_read_query(query, {"name": space_name})
        return results[0]["s"] if results else None

    # ==============================
    # 关系操作接口
    # ==============================

    def add_relationship(
        self,
        source_id: str,
        target_id: str,
        relationship_type: str,
        properties: Optional[Dict[str, Any]] = None,
        source_labels: Optional[List[str]] = None,
        target_labels: Optional[List[str]] = None
    ) -> bool:
        """
        在两个节点之间添加关系
        
        Args:
            source_id: 源节点ID
            target_id: 目标节点ID
            relationship_type: 关系类型
            properties: 关系属性
            source_labels: 源节点标签列表
            target_labels: 目标节点标签列表
            
        Returns:
            bool: 操作是否成功
        """
        try:
            # 准备关系属性
            rel_properties = {
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            }
            
            if properties:
                # 过滤掉不能存储的复杂类型
                filtered_props = {}
                for k, v in properties.items():
                    if isinstance(v, (str, int, float, bool, list)):
                        if isinstance(v, list) and all(isinstance(x, (str, int, float, bool)) for x in v):
                            filtered_props[k] = v
                        elif not isinstance(v, list):
                            filtered_props[k] = v
                    elif isinstance(v, dict):
                        # 复杂对象转JSON字符串
                        filtered_props[k] = json.dumps(v, ensure_ascii=False)
                
                rel_properties.update(filtered_props)
            
            # 构建节点匹配模式
            source_pattern = self._build_node_pattern("source", source_id, source_labels)
            target_pattern = self._build_node_pattern("target", target_id, target_labels)
            
            # 清理关系类型名称
            clean_rel_type = self._clean_relationship_type(relationship_type)
            
            query = f"""
            MATCH {source_pattern}, {target_pattern}
            MERGE (source)-[r:{clean_rel_type}]->(target)
            SET r += $properties
            RETURN type(r) as rel_type
            """
            
            result = self._execute_write_query(query, {
                "source_id": source_id,
                "target_id": target_id,
                "properties": rel_properties
            })
            
            if result.get("success"):
                logging.debug(f"成功添加关系: {source_id} -[{relationship_type}]-> {target_id}")
                return True
            else:
                logging.error(f"添加关系失败: {source_id} -[{relationship_type}]-> {target_id}")
                return False
                
        except Exception as e:
            logging.error(f"添加关系时出错: {e}")
            return False

    def get_relationships(
        self,
        source_id: Optional[str] = None,
        target_id: Optional[str] = None,
        relationship_type: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        查询关系
        
        Args:
            source_id: 源节点ID（可选）
            target_id: 目标节点ID（可选）
            relationship_type: 关系类型（可选）
            limit: 返回结果限制
            
        Returns:
            List[Dict[str, Any]]: 关系信息列表
        """
        # 构建查询条件
        conditions = []
        parameters = {"limit": limit}
        
        if source_id:
            conditions.append("source.uid = $source_id")
            parameters["source_id"] = source_id
            
        if target_id:
            conditions.append("target.uid = $target_id")
            parameters["target_id"] = target_id
        
        where_clause = " AND ".join(conditions) if conditions else ""
        if where_clause:
            where_clause = "WHERE " + where_clause
        
        # 构建关系类型过滤
        rel_pattern = f"[r:{self._clean_relationship_type(relationship_type)}]" if relationship_type else "[r]"
        
        query = f"""
        MATCH (source)-{rel_pattern}->(target)
        {where_clause}
        RETURN source.uid as source_id, target.uid as target_id, 
               type(r) as relationship_type, properties(r) as properties
        LIMIT $limit
        """
        
        return self._execute_read_query(query, parameters)

    def delete_relationship(
        self,
        source_id: str,
        target_id: str,
        relationship_type: Optional[str] = None
    ) -> bool:
        """
        删除关系
        
        Args:
            source_id: 源节点ID
            target_id: 目标节点ID
            relationship_type: 关系类型（可选，如果不指定则删除所有关系）
            
        Returns:
            bool: 删除是否成功
        """
        try:
            if relationship_type:
                clean_rel_type = self._clean_relationship_type(relationship_type)
                query = f"""
                MATCH (source {{uid: $source_id}})-[r:{clean_rel_type}]->(target {{uid: $target_id}})
                DELETE r
                """
            else:
                query = """
                MATCH (source {uid: $source_id})-[r]->(target {uid: $target_id})
                DELETE r
                """
            
            result = self._execute_write_query(query, {
                "source_id": source_id,
                "target_id": target_id
            })
            
            return result.get("success", False)
            
        except Exception as e:
            logging.error(f"删除关系时出错: {e}")
            return False

    def update_relationship_properties(
        self,
        source_id: str,
        target_id: str,
        relationship_type: str,
        properties: Dict[str, Any],
        merge: bool = True
    ) -> bool:
        """
        更新关系属性
        
        Args:
            source_id: 源节点ID
            target_id: 目标节点ID
            relationship_type: 关系类型
            properties: 新的属性
            merge: 是否与现有属性合并
            
        Returns:
            bool: 更新是否成功
        """
        try:
            clean_rel_type = self._clean_relationship_type(relationship_type)
            
            # 过滤属性
            filtered_props = self._filter_properties(properties)
            filtered_props["updated_at"] = datetime.now().isoformat()
            
            if merge:
                query = f"""
                MATCH (source {{uid: $source_id}})-[r:{clean_rel_type}]->(target {{uid: $target_id}})
                SET r += $properties
                RETURN type(r) as rel_type
                """
            else:
                query = f"""
                MATCH (source {{uid: $source_id}})-[r:{clean_rel_type}]->(target {{uid: $target_id}})
                SET r = $properties
                RETURN type(r) as rel_type
                """
            
            result = self._execute_write_query(query, {
                "source_id": source_id,
                "target_id": target_id,
                "properties": filtered_props
            })
            
            return result.get("success", False)
            
        except Exception as e:
            logging.error(f"更新关系属性时出错: {e}")
            return False

    # ==============================
    # 图遍历和查询接口
    # ==============================

    def get_neighbors(
        self,
        node_id: str,
        direction: str = "both",
        relationship_types: Optional[List[str]] = None,
        depth: int = 1,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        获取节点的邻居
        
        Args:
            node_id: 节点ID
            direction: 方向 ("outgoing", "incoming", "both")
            relationship_types: 关系类型列表
            depth: 遍历深度
            limit: 结果限制
            
        Returns:
            List[Dict[str, Any]]: 邻居节点信息
        """
        # 构建关系模式
        if relationship_types:
            rel_types = "|".join([self._clean_relationship_type(rt) for rt in relationship_types])
            rel_pattern = f"[:{rel_types}]"
        else:
            rel_pattern = ""
        
        # 构建方向模式
        if direction == "outgoing":
            pattern = f"-{rel_pattern}->"
        elif direction == "incoming":
            pattern = f"<-{rel_pattern}-"
        else:  # both
            pattern = f"-{rel_pattern}-"
        
        # 构建深度模式
        depth_pattern = f"*1..{depth}" if depth > 1 else ""
        
        query = f"""
        MATCH (start {{uid: $node_id}}){pattern}{depth_pattern}(neighbor)
        WHERE start <> neighbor
        RETURN DISTINCT neighbor.uid as uid, labels(neighbor) as labels, 
               properties(neighbor) as properties
        LIMIT $limit
        """
        
        return self._execute_read_query(query, {
            "node_id": node_id,
            "limit": limit
        })

    def find_shortest_path(
        self,
        source_id: str,
        target_id: str,
        relationship_types: Optional[List[str]] = None,
        max_depth: int = 6
    ) -> Optional[Dict[str, Any]]:
        """
        查找两个节点之间的最短路径
        
        Args:
            source_id: 源节点ID
            target_id: 目标节点ID
            relationship_types: 关系类型列表
            max_depth: 最大搜索深度
            
        Returns:
            Optional[Dict[str, Any]]: 路径信息或None
        """
        # 构建关系模式
        if relationship_types:
            rel_types = "|".join([self._clean_relationship_type(rt) for rt in relationship_types])
            rel_pattern = f"[:{rel_types}]"
        else:
            rel_pattern = ""
        
        query = f"""
        MATCH path = shortestPath((source {{uid: $source_id}})-{rel_pattern}*1..{max_depth}-(target {{uid: $target_id}}))
        RETURN path, length(path) as path_length
        """
        
        results = self._execute_read_query(query, {
            "source_id": source_id,
            "target_id": target_id
        })
        
        return results[0] if results else None

    def find_paths_between_nodes(
        self,
        source_id: str,
        target_id: str,
        relationship_types: Optional[List[str]] = None,
        max_depth: int = 4,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        查找两个节点之间的所有路径
        
        Args:
            source_id: 源节点ID
            target_id: 目标节点ID
            relationship_types: 关系类型列表
            max_depth: 最大搜索深度
            limit: 路径数量限制
            
        Returns:
            List[Dict[str, Any]]: 路径信息列表
        """
        if relationship_types:
            rel_types = "|".join([self._clean_relationship_type(rt) for rt in relationship_types])
            rel_pattern = f"[:{rel_types}]"
        else:
            rel_pattern = ""
        
        query = f"""
        MATCH path = (source {{uid: $source_id}})-{rel_pattern}*1..{max_depth}-(target {{uid: $target_id}})
        RETURN path, length(path) as path_length
        ORDER BY path_length
        LIMIT $limit
        """
        
        return self._execute_read_query(query, {
            "source_id": source_id,
            "target_id": target_id,
            "limit": limit
        })

    def get_subgraph(
        self,
        center_nodes: List[str],
        depth: int = 2,
        relationship_types: Optional[List[str]] = None,
        node_limit: int = 500
    ) -> Dict[str, Any]:
        """
        获取以指定节点为中心的子图
        
        Args:
            center_nodes: 中心节点ID列表
            depth: 扩展深度
            relationship_types: 关系类型列表
            node_limit: 节点数量限制
            
        Returns:
            Dict[str, Any]: 子图信息（节点和关系）
        """
        if relationship_types:
            rel_types = "|".join([self._clean_relationship_type(rt) for rt in relationship_types])
            rel_pattern = f"[:{rel_types}]"
        else:
            rel_pattern = ""
        
        query = f"""
        MATCH (center)
        WHERE center.uid IN $center_nodes
        CALL {{
            WITH center
            MATCH path = (center)-{rel_pattern}*0..{depth}-(node)
            RETURN DISTINCT node
            LIMIT $node_limit
        }}
        WITH collect(DISTINCT node) as nodes
        UNWIND nodes as n1
        UNWIND nodes as n2
        MATCH (n1)-[r]->(n2)
        RETURN collect(DISTINCT {{uid: n1.uid, labels: labels(n1), properties: properties(n1)}}) as nodes,
               collect(DISTINCT {{source: n1.uid, target: n2.uid, type: type(r), properties: properties(r)}}) as relationships
        """
        
        results = self._execute_read_query(query, {
            "center_nodes": center_nodes,
            "node_limit": node_limit
        })
        
        if results:
            return {
                "nodes": results[0]["nodes"],
                "relationships": results[0]["relationships"]
            }
        else:
            return {"nodes": [], "relationships": []}

    # ==============================
    # 高级查询接口
    # ==============================

    def search_nodes_by_properties(
        self,
        properties: Dict[str, Any],
        node_labels: Optional[List[str]] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        根据属性搜索节点
        
        Args:
            properties: 搜索属性
            node_labels: 节点标签列表
            limit: 结果限制
            
        Returns:
            List[Dict[str, Any]]: 节点信息列表
        """
        # 构建标签模式
        if node_labels:
            label_pattern = ":".join(node_labels)
            node_pattern = f"(n:{label_pattern})"
        else:
            node_pattern = "(n)"
        
        # 构建属性条件
        conditions = []
        parameters = {"limit": limit}
        
        for key, value in properties.items():
            param_key = f"prop_{len(conditions)}"
            if isinstance(value, str) and "*" in value:
                # 支持通配符搜索
                conditions.append(f"n.{key} =~ $" + param_key)
                parameters[param_key] = value.replace("*", ".*")
            else:
                conditions.append(f"n.{key} = $" + param_key)
                parameters[param_key] = value
        
        where_clause = " AND ".join(conditions) if conditions else "true"
        
        query = f"""
        MATCH {node_pattern}
        WHERE {where_clause}
        RETURN n.uid as uid, labels(n) as labels, properties(n) as properties
        LIMIT $limit
        """
        
        return self._execute_read_query(query, parameters)

    def search_nodes_by_fulltext(
        self,
        search_text: str,
        node_labels: Optional[List[str]] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        使用全文搜索查找节点
        
        Args:
            search_text: 搜索文本
            node_labels: 节点标签列表
            limit: 结果限制
            
        Returns:
            List[Dict[str, Any]]: 节点信息列表
        """
        # 使用Neo4j的全文搜索索引
        query = """
        CALL db.index.fulltext.queryNodes("memory_unit_fulltext", $search_text)
        YIELD node, score
        RETURN node.uid as uid, labels(node) as labels, 
               properties(node) as properties, score
        ORDER BY score DESC
        LIMIT $limit
        """
        
        return self._execute_read_query(query, {
            "search_text": search_text,
            "limit": limit
        })

    def get_relationship_statistics(self) -> Dict[str, Any]:
        """
        获取关系统计信息
        
        Returns:
            Dict[str, Any]: 关系统计信息
        """
        query = """
        MATCH ()-[r]->()
        RETURN type(r) as relationship_type, count(r) as count
        ORDER BY count DESC
        """
        
        results = self._execute_read_query(query)
        
        stats = {
            "relationship_types": {},
            "total_relationships": 0
        }
        
        for record in results:
            rel_type = record["relationship_type"]
            count = record["count"]
            stats["relationship_types"][rel_type] = count
            stats["total_relationships"] += count
        
        return stats

    def get_node_statistics(self) -> Dict[str, Any]:
        """
        获取节点统计信息
        
        Returns:
            Dict[str, Any]: 节点统计信息
        """
        query = """
        MATCH (n)
        UNWIND labels(n) as label
        RETURN label, count(n) as count
        ORDER BY count DESC
        """
        
        results = self._execute_read_query(query)
        
        stats = {
            "node_labels": {},
            "total_nodes": 0
        }
        
        for record in results:
            label = record["label"]
            count = record["count"]
            stats["node_labels"][label] = count
        
        # 获取总节点数（去重）
        total_query = "MATCH (n) RETURN count(n) as total"
        total_results = self._execute_read_query(total_query)
        stats["total_nodes"] = total_results[0]["total"] if total_results else 0
        
        return stats

    # ==============================
    # 数据管理接口
    # ==============================

    def clear_database(self, confirm: bool = False) -> bool:
        """
        清空数据库（危险操作）
        
        Args:
            confirm: 确认清空
            
        Returns:
            bool: 清空是否成功
        """
        if not confirm:
            logging.warning("清空数据库需要确认参数 confirm=True")
            return False
        
        try:
            # 分批删除所有节点和关系
            query = """
            MATCH (n)
            WITH n LIMIT 1000
            DETACH DELETE n
            RETURN count(n) as deleted
            """
            
            total_deleted = 0
            while True:
                result = self._execute_write_query(query)
                if not result.get("success"):
                    break
                
                deleted = result.get("result", [{}])[0].get("deleted", 0)
                total_deleted += deleted
                
                if deleted == 0:
                    break
            
            logging.info(f"已清空数据库，删除了 {total_deleted} 个节点")
            return True
            
        except Exception as e:
            logging.error(f"清空数据库失败: {e}")
            return False

    def export_graph_data(
        self,
        output_file: str,
        format: str = "json",
        include_properties: bool = True
    ) -> bool:
        """
        导出图数据
        
        Args:
            output_file: 输出文件路径
            format: 导出格式 (json/cypher)
            include_properties: 是否包含属性
            
        Returns:
            bool: 导出是否成功
        """
        try:
            if format.lower() == "json":
                return self._export_as_json(output_file, include_properties)
            elif format.lower() == "cypher":
                return self._export_as_cypher(output_file, include_properties)
            else:
                logging.error(f"不支持的导出格式: {format}")
                return False
                
        except Exception as e:
            logging.error(f"导出图数据失败: {e}")
            return False

    def import_graph_data(
        self,
        input_file: str,
        format: str = "json",
        clear_existing: bool = False
    ) -> bool:
        """
        导入图数据
        
        Args:
            input_file: 输入文件路径
            format: 数据格式 (json/cypher)
            clear_existing: 是否清空现有数据
            
        Returns:
            bool: 导入是否成功
        """
        try:
            if clear_existing:
                self.clear_database(confirm=True)
            
            if format.lower() == "json":
                return self._import_from_json(input_file)
            elif format.lower() == "cypher":
                return self._import_from_cypher(input_file)
            else:
                logging.error(f"不支持的导入格式: {format}")
                return False
                
        except Exception as e:
            logging.error(f"导入图数据失败: {e}")
            return False

    # ==============================
    # 性能监控和维护接口
    # ==============================

    def get_performance_stats(self) -> Dict[str, Any]:
        """
        获取性能统计信息
        
        Returns:
            Dict[str, Any]: 性能统计信息
        """
        stats = self.query_stats.copy()
        
        # 计算平均查询时间
        if stats["successful_queries"] > 0:
            stats["avg_query_time"] = stats["total_query_time"] / stats["successful_queries"]
        else:
            stats["avg_query_time"] = 0.0
        
        # 计算成功率
        if stats["total_queries"] > 0:
            stats["success_rate"] = stats["successful_queries"] / stats["total_queries"]
        else:
            stats["success_rate"] = 0.0
        
        return stats

    def optimize_database(self) -> bool:
        """
        优化数据库性能
        
        Returns:
            bool: 优化是否成功
        """
        try:
            # 更新统计信息
            self._execute_write_query("CALL db.stats.retrieve('GRAPH COUNTS')")
            
            # 重建索引（如果需要）
            # 注意：具体的优化操作依赖于Neo4j版本和配置
            
            logging.info("数据库优化完成")
            return True
            
        except Exception as e:
            logging.error(f"数据库优化失败: {e}")
            return False

    def health_check(self) -> Dict[str, Any]:
        """
        健康检查
        
        Returns:
            Dict[str, Any]: 健康状态信息
        """
        health_status = {
            "neo4j_connected": self.neo4j_connected,
            "milvus_enabled": self.milvus_enabled,
            "milvus_connected": self.milvus_operator.is_connected if self.milvus_operator else False,
            "connection_info": {
                "neo4j_uri": self.neo4j_uri,
                "neo4j_database": self.neo4j_database,
                "milvus_host": self.milvus_operator.host if self.milvus_operator else None,
                "milvus_collection": self.milvus_operator.collection_name if self.milvus_operator else None
            }
        }
        
        # 测试基本操作
        try:
            test_query = "RETURN 1 as test"
            test_result = self._execute_read_query(test_query)
            health_status["neo4j_query_test"] = len(test_result) > 0
        except Exception as e:
            health_status["neo4j_query_test"] = False
            health_status["neo4j_error"] = str(e)
        
        return health_status

    # ==============================
    # 内部辅助方法
    # ==============================

    def _infer_content_type(self, raw_data: Dict[str, Any]) -> str:
        """推断内容类型"""
        if "image_path" in raw_data or "image" in raw_data:
            return "image"
        elif "text_content" in raw_data or "text" in raw_data or "content" in raw_data:
            return "text"
        else:
            return "mixed"

    def _parse_node_to_memory_unit(self, node_data: Dict[str, Any]) -> Optional[MemoryUnit]:
        """将Neo4j节点数据解析为MemoryUnit对象"""
        try:
            # 提取基本信息
            uid = node_data.get("uid")
            if not uid:
                return None
            
            # 重构raw_data
            raw_data = {}
            metadata = {}
            
            for key, value in node_data.items():
                if key.startswith("raw_"):
                    # 原始数据字段
                    original_key = key[4:]  # 移除"raw_"前缀
                    if isinstance(value, str) and (value.startswith("{") or value.startswith("[")):
                        try:
                            raw_data[original_key] = json.loads(value)
                        except json.JSONDecodeError:
                            raw_data[original_key] = value
                    else:
                        raw_data[original_key] = value
                elif key.startswith("meta_"):
                    # 元数据字段
                    original_key = key[5:]  # 移除"meta_"前缀
                    metadata[original_key] = value
                elif key not in ["uid", "created_at", "updated_at", "has_embedding", "content_type"]:
                    # 其他字段归入metadata
                    metadata[key] = value
            
            # 从Milvus获取embedding（如果可用）
            embedding = None
            if self.milvus_enabled and node_data.get("has_embedding"):
                try:
                    milvus_unit = self.milvus_operator.get_unit(uid)
                    if milvus_unit:
                        embedding = milvus_unit.embedding
                except Exception as e:
                    logging.debug(f"从Milvus获取embedding失败: {e}")
            
            return MemoryUnit(
                uid=uid,
                raw_data=raw_data,
                metadata=metadata,
                embedding=embedding
            )
            
        except Exception as e:
            logging.error(f"解析节点数据为MemoryUnit失败: {e}")
            return None

    def _build_node_pattern(
        self, 
        var_name: str, 
        node_id: str, 
        labels: Optional[List[str]] = None
    ) -> str:
        """构建节点匹配模式"""
        if labels:
            label_pattern = ":".join(labels)
            return f"({var_name}:{label_pattern} {{uid: ${var_name}_id}})"
        else:
            return f"({var_name} {{uid: ${var_name}_id}})"

    def _clean_relationship_type(self, rel_type: str) -> str:
        """清理关系类型名称，确保符合Neo4j命名规范"""
        # 移除特殊字符，替换为下划线
        import re
        cleaned = re.sub(r'[^a-zA-Z0-9_]', '_', rel_type)
        # 确保不以数字开头
        if cleaned and cleaned[0].isdigit():
            cleaned = f"REL_{cleaned}"
        return cleaned or "RELATED_TO"

    def _filter_properties(self, properties: Dict[str, Any]) -> Dict[str, Any]:
        """过滤属性，移除不能存储的类型"""
        filtered = {}
        for k, v in properties.items():
            if isinstance(v, (str, int, float, bool)):
                filtered[k] = v
            elif isinstance(v, list) and all(isinstance(x, (str, int, float, bool)) for x in v):
                filtered[k] = v
            elif isinstance(v, (dict, list)):
                # 复杂类型转为JSON字符串
                try:
                    filtered[k] = json.dumps(v, ensure_ascii=False)
                except Exception:
                    continue
        return filtered

    def _export_as_json(self, output_file: str, include_properties: bool) -> bool:
        """导出为JSON格式"""
        try:
            # 导出节点
            node_query = "MATCH (n) RETURN n"
            nodes = self._execute_read_query(node_query)
            
            # 导出关系
            rel_query = """
            MATCH (source)-[r]->(target)
            RETURN source.uid as source_id, target.uid as target_id,
                   type(r) as relationship_type, properties(r) as properties
            """
            relationships = self._execute_read_query(rel_query)
            
            # 准备导出数据
            export_data = {
                "metadata": {
                    "exported_at": datetime.now().isoformat(),
                    "total_nodes": len(nodes),
                    "total_relationships": len(relationships)
                },
                "nodes": [{"uid": node["n"]["uid"], 
                          "labels": node["n"].labels,
                          "properties": dict(node["n"]) if include_properties else {}}
                         for node in nodes],
                "relationships": relationships
            }
            
            # 写入文件
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            
            logging.info(f"成功导出图数据到 {output_file}")
            return True
            
        except Exception as e:
            logging.error(f"JSON导出失败: {e}")
            return False

    def _export_as_cypher(self, output_file: str, include_properties: bool) -> bool:
        """导出为Cypher脚本"""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                # 导出节点创建语句
                node_query = "MATCH (n) RETURN n"
                nodes = self._execute_read_query(node_query)
                
                f.write("// 节点创建语句\n")
                for record in nodes:
                    node = record["n"]
                    labels = ":".join(node.labels)
                    if include_properties:
                        props = json.dumps(dict(node), ensure_ascii=False)
                        f.write(f"CREATE (:{labels} {props});\n")
                    else:
                        f.write(f"CREATE (:{labels} {{uid: '{node['uid']}'}};\n")
                
                # 导出关系创建语句
                f.write("\n// 关系创建语句\n")
                rel_query = """
                MATCH (source)-[r]->(target)
                RETURN source.uid as source_id, target.uid as target_id,
                       type(r) as relationship_type, properties(r) as properties
                """
                relationships = self._execute_read_query(rel_query)
                
                for rel in relationships:
                    source_id = rel["source_id"]
                    target_id = rel["target_id"]
                    rel_type = rel["relationship_type"]
                    
                    if include_properties and rel["properties"]:
                        props = json.dumps(rel["properties"], ensure_ascii=False)
                        f.write(f"MATCH (a {{uid: '{source_id}'}}), (b {{uid: '{target_id}'}}) "
                               f"CREATE (a)-[:{rel_type} {props}]->(b);\n")
                    else:
                        f.write(f"MATCH (a {{uid: '{source_id}'}}), (b {{uid: '{target_id}'}}) "
                               f"CREATE (a)-[:{rel_type}]->(b);\n")
            
            logging.info(f"成功导出Cypher脚本到 {output_file}")
            return True
            
        except Exception as e:
            logging.error(f"Cypher导出失败: {e}")
            return False

    def _import_from_json(self, input_file: str) -> bool:
        """从JSON文件导入"""
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 导入节点
            nodes = data.get("nodes", [])
            for node in nodes:
                uid = node["uid"]
                labels = node.get("labels", ["MemoryUnit"])
                properties = node.get("properties", {})
                
                label_str = ":".join(labels)
                query = f"CREATE (:{label_str} $properties)"
                self._execute_write_query(query, {"properties": properties})
            
            # 导入关系
            relationships = data.get("relationships", [])
            for rel in relationships:
                source_id = rel["source_id"]
                target_id = rel["target_id"]
                rel_type = self._clean_relationship_type(rel["relationship_type"])
                properties = rel.get("properties", {})
                
                query = f"""
                MATCH (a {{uid: $source_id}}), (b {{uid: $target_id}})
                CREATE (a)-[:{rel_type} $properties]->(b)
                """
                self._execute_write_query(query, {
                    "source_id": source_id,
                    "target_id": target_id,
                    "properties": properties
                })
            
            logging.info(f"成功从 {input_file} 导入图数据")
            return True
            
        except Exception as e:
            logging.error(f"JSON导入失败: {e}")
            return False

    def _import_from_cypher(self, input_file: str) -> bool:
        """从Cypher文件导入"""
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                cypher_content = f.read()
            
            # 分割为单独的语句
            statements = [stmt.strip() for stmt in cypher_content.split(';') if stmt.strip()]
            
            for statement in statements:
                if statement.startswith('//') or not statement:
                    continue  # 跳过注释和空行
                
                self._execute_write_query(statement)
            
            logging.info(f"成功从 {input_file} 导入Cypher脚本")
            return True
            
        except Exception as e:
            logging.error(f"Cypher导入失败: {e}")
            return False

    def close(self):
        """关闭连接"""
        if self.driver:
            try:
                self.driver.close()
                self.neo4j_connected = False
                logging.info("Neo4j连接已关闭")
            except Exception as e:
                logging.error(f"关闭Neo4j连接失败: {e}")
        
        if self.milvus_operator:
            try:
                self.milvus_operator.close()
                self.milvus_enabled = False
                logging.info("Milvus连接已关闭")
            except Exception as e:
                logging.error(f"关闭Milvus连接失败: {e}")

    def __enter__(self):
        """上下文管理器入口"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.close()

    def __repr__(self):
        """字符串表示"""
        neo4j_status = "已连接" if self.neo4j_connected else "未连接"
        milvus_status = "已启用" if self.milvus_enabled else "未启用"
        return f"Neo4jOperator(neo4j={neo4j_status}, milvus={milvus_status})"


# ==============================
# 使用示例和测试代码
# ==============================

if __name__ == "__main__":
    # 测试Neo4j操作
    with Neo4jOperator(
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="password",
        milvus_host="localhost",
        milvus_port="19530"
    ) as neo4j_op:
        
        print("Neo4j操作类测试")
        print("=" * 50)
        
        # 健康检查
        health = neo4j_op.health_check()
        print(f"健康状态: {health}")
        
        if neo4j_op.neo4j_connected:
            # 创建测试节点
            test_unit = MemoryUnit(
                uid="test_neo4j_001",
                raw_data={
                    "title": "测试Neo4j集成",
                    "content": "这是一个测试记忆单元",
                    "category": "test"
                },
                metadata={
                    "source": "test_script",
                    "version": "1.0"
                },
                embedding=np.random.rand(512).astype(np.float32)
            )
            
            # 添加节点
            print(f"添加测试节点: {neo4j_op.add_unit(test_unit)}")
            
            # 获取节点
            retrieved_unit = neo4j_op.get_unit("test_neo4j_001")
            print(f"检索节点: {retrieved_unit.uid if retrieved_unit else 'None'}")
            
            # 添加第二个节点
            test_unit2 = MemoryUnit(
                uid="test_neo4j_002",
                raw_data={
                    "title": "第二个测试节点",
                    "content": "用于测试关系",
                    "category": "test"# neo4j_operator.py
                },
                metadata={
                    "source": "test_script",
                    "version": "1.0"
                },
                embedding=np.random.rand(512).astype(np.float32)
            )
            print(f"添加第二个测试节点: {neo4j_op.add_unit(test_unit2)}")