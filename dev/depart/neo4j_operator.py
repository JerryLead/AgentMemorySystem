import logging
import os
import numpy as np
import faiss
from typing import Dict, List, Optional, Set, Tuple, Any, Union
import json
from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError
from sentence_transformers import SentenceTransformer
from .memory_unit import MemoryUnit
from .milvus_operator import MilvusOperator

class Neo4jOperator:
    """增强版Neo4j操作类，集成Neo4j和Milvus，并维护本地FAISS索引"""
    
    def __init__(
        self, 
        # Neo4j配置
        neo4j_uri: str = "bolt://localhost:7687", 
        neo4j_user: str = "neo4j", 
        neo4j_password: str = "password",
        neo4j_database: str = "neo4j",
        # Milvus配置
        milvus_host: str = "localhost",
        milvus_port: str = "19530",
        milvus_user: str = "",
        milvus_password: str = "",
        milvus_collection: str = "hippo_memory_units",
        # FAISS配置
        embedding_dim: int = 512,
        use_local_faiss: bool = True,
        faiss_index_type: str = "Flat"
    ):
        """
        初始化增强版Neo4j操作类
        
        参数:
            neo4j_uri: Neo4j服务器URI
            neo4j_user: Neo4j用户名
            neo4j_password: Neo4j密码
            neo4j_database: Neo4j数据库名称
            milvus_host: Milvus服务器地址
            milvus_port: Milvus服务器端口
            milvus_user: Milvus用户名
            milvus_password: Milvus密码
            milvus_collection: Milvus集合名称
            embedding_dim: 向量维度
            use_local_faiss: 是否使用本地FAISS索引
            faiss_index_type: FAISS索引类型
        """
        self.embedding_dim = embedding_dim
        self.use_local_faiss = use_local_faiss
        
        # 初始化Neo4j连接
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.neo4j_database = neo4j_database
        self.neo4j_connected = False
        
        try:
            self.neo4j_driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
            # 测试连接
            with self.neo4j_driver.session(database=neo4j_database) as session:
                session.run("RETURN 1")
            self.neo4j_connected = True
            logging.info(f"成功连接到Neo4j服务器: {neo4j_uri}")
        except Exception as e:
            logging.error(f"连接到Neo4j服务器失败: {e}")
            self.neo4j_connected = False
        
        # 初始化Milvus连接
        self.milvus_operator = MilvusOperator(
            host=milvus_host,
            port=milvus_port,
            user=milvus_user,
            password=milvus_password,
            collection_name=milvus_collection,
            embedding_dim=embedding_dim
        )
        
        # 初始化本地FAISS索引
        self.faiss_index = None
        self.faiss_id_to_unit_id = {}  # FAISS内部ID到单元ID的映射
        self.unit_id_to_faiss_id = {}  # 单元ID到FAISS内部ID的映射
        self.faiss_id_counter = 0
        
        if use_local_faiss:
            self._init_faiss_index(faiss_index_type)
    
    # def _init_faiss_index(self, index_type: str = "Flat"):
    #     """初始化FAISS索引"""
    #     try:
    #         if index_type == "Flat":
    #             self.faiss_index = faiss.IndexFlatL2(self.embedding_dim)
    #         elif index_type == "HNSW":
    #             self.faiss_index = faiss.IndexHNSWFlat(self.embedding_dim, 32)
    #         else:
    #             # 默认使用Flat索引
    #             self.faiss_index = faiss.IndexFlatL2(self.embedding_dim)
            
    #         logging.info(f"已初始化本地FAISS索引 (类型: {index_type})")
    #     except Exception as e:
    #         logging.error(f"初始化FAISS索引失败: {e}")
    #         self.faiss_index = None
    
    def _init_faiss_index(self, index_type: str = "Flat"):
        """初始化FAISS索引"""
        try:
            if index_type == "Flat":
                # 创建基础索引
                base_index = faiss.IndexFlatL2(self.embedding_dim)
                # 用IndexIDMap包装基础索引以支持add_with_ids
                self.faiss_index = faiss.IndexIDMap(base_index)
            elif index_type == "HNSW":
                # 对于HNSW索引，同样使用IndexIDMap包装
                base_index = faiss.IndexHNSWFlat(self.embedding_dim, 32)
                self.faiss_index = faiss.IndexIDMap(base_index)
            else:
                # 默认使用Flat索引并包装
                base_index = faiss.IndexFlatL2(self.embedding_dim)
                self.faiss_index = faiss.IndexIDMap(base_index)
            
            logging.info(f"已初始化本地FAISS索引 (类型: IndexIDMap({index_type}))")
        except Exception as e:
            logging.error(f"初始化FAISS索引失败: {e}")
            self.faiss_index = None

    def ensure_node_exists(self, unit_id: str, type_labels: Optional[List[str]] = None) -> bool:
        """
        确保节点在Neo4j中存在（只存储ID，不存储具体数据）
        
        参数:
            unit_id: 节点ID
            type_labels: 节点的附加标签列表
        """
        if not self.neo4j_connected:
            logging.error("未连接到Neo4j服务器")
            return False
        
        try:
            labels = ["MemoryUnit"]
            if type_labels:
                labels.extend(type_labels)
            
            label_str = ":".join(labels)
            
            with self.neo4j_driver.session(database=self.neo4j_database) as session:
                query = f"MERGE (n:{label_str} {{uid: $id}}) RETURN n"
                result = session.run(query, id=unit_id)
                summary = result.consume()
                
                if summary.counters.nodes_created > 0:
                    logging.info(f"成功创建节点 '{unit_id}'")
                else:
                    logging.debug(f"节点 '{unit_id}' 已存在")
                
                return True
                
        except Exception as e:
            logging.error(f"确保节点 '{unit_id}' 存在时失败: {e}")
            return False
    
    def add_unit(self, unit: MemoryUnit, space_names: Optional[List[str]] = None, type_labels: Optional[List[str]] = None) -> bool:
        """
        添加MemoryUnit到Neo4j和Milvus
        
        参数:
            unit: 要添加的MemoryUnit对象
            space_names: 该单元所属的空间名称列表
            type_labels: 节点在Neo4j中的附加标签
            
        返回:
            添加是否成功
        """
        success_neo4j = self.ensure_node_exists(unit.uid, type_labels)
        success_milvus = self.milvus_operator.add_unit(unit, space_names)
        
        # 更新本地FAISS索引
        # 在add_unit方法中的FAISS索引部分
        if success_milvus and self.use_local_faiss and self.faiss_index and unit.embedding is not None:
            try:
                # 添加新向量
                faiss_id = self.faiss_id_counter
                self.faiss_id_counter += 1
                
                vector = unit.embedding.reshape(1, -1).astype(np.float32)
                
                # 直接使用add_with_ids，不需要检查是否有该方法
                self.faiss_index.add_with_ids(vector, np.array([faiss_id], dtype=np.int64))
                
                self.faiss_id_to_unit_id[faiss_id] = unit.uid
                self.unit_id_to_faiss_id[unit.uid] = faiss_id
                
                logging.info(f"单元 '{unit.uid}' 已添加到本地FAISS索引")
            except Exception as e:
                logging.error(f"添加单元 '{unit.uid}' 到本地FAISS索引失败: {e}")
    
    def add_relationship(
        self, 
        source_id: str, 
        target_id: str, 
        relationship_type: str, 
        properties: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        在两个节点之间添加关系
        
        参数:
            source_id: 源节点ID
            target_id: 目标节点ID
            relationship_type: 关系类型
            properties: 关系属性
        
        返回:
            添加是否成功
        """
        if not self.neo4j_connected:
            logging.error("未连接到Neo4j服务器")
            return False
        
        # 首先确保两个节点都存在
        self.ensure_node_exists(source_id)
        self.ensure_node_exists(target_id)
        
        try:
            # 处理属性
            props = {}
            if properties:
                for k, v in properties.items():
                    if isinstance(v, (dict, list)):
                        props[k] = json.dumps(v)
                    else:
                        props[k] = v
            
            with self.neo4j_driver.session(database=self.neo4j_database) as session:
                query = (
                    "MATCH (source:MemoryUnit {uid: $source_id}), "
                    "(target:MemoryUnit {uid: $target_id}) "
                    "MERGE (source)-[r:" + relationship_type + "]->(target) "
                    "SET r = $properties "
                    "RETURN r"
                )
                result = session.run(
                    query, 
                    source_id=source_id, 
                    target_id=target_id, 
                    properties=props
                )
                summary = result.consume()
                
                if summary.counters.relationships_created > 0:
                    logging.info(f"成功创建从 '{source_id}' 到 '{target_id}' 的 '{relationship_type}' 关系")
                else:
                    logging.info(f"成功更新从 '{source_id}' 到 '{target_id}' 的 '{relationship_type}' 关系")
                
                return True
                
        except Exception as e:
            logging.error(f"添加从 '{source_id}' 到 '{target_id}' 的 '{relationship_type}' 关系失败: {e}")
            return False
    
    def delete_unit(self, unit_id: str) -> bool:
        """
        从Neo4j和Milvus中删除MemoryUnit
        
        参数:
            unit_id: 要删除的单元ID
            
        返回:
            删除是否成功
        """
        success_neo4j = True
        success_milvus = True
        
        # 从Neo4j删除
        if self.neo4j_connected:
            try:
                with self.neo4j_driver.session(database=self.neo4j_database) as session:
                    query = "MATCH (n:MemoryUnit {uid: $id}) DETACH DELETE n"
                    result = session.run(query, id=unit_id)
                    summary = result.consume()
                    
                    if summary.counters.nodes_deleted > 0:
                        logging.info(f"成功从Neo4j删除节点 '{unit_id}' 及其关系")
                    else:
                        logging.info(f"节点 '{unit_id}' 不存在于Neo4j中")
            except Exception as e:
                logging.error(f"从Neo4j删除节点 '{unit_id}' 失败: {e}")
                success_neo4j = False
        
        # 从Milvus删除
        success_milvus = self.milvus_operator.delete_unit(unit_id)
        
        # 从本地FAISS索引删除
        if self.use_local_faiss and unit_id in self.unit_id_to_faiss_id:
            try:
                if hasattr(self.faiss_index, 'remove_ids'):
                    faiss_id = self.unit_id_to_faiss_id[unit_id]
                    self.faiss_index.remove_ids(np.array([faiss_id], dtype=np.int64))
                    del self.faiss_id_to_unit_id[faiss_id]
                    del self.unit_id_to_faiss_id[unit_id]
                    logging.info(f"单元 '{unit_id}' 已从本地FAISS索引删除")
                else:
                    logging.warning(f"当前FAISS索引不支持删除，单元 '{unit_id}' 将在下次重建索引时移除")
            except Exception as e:
                logging.error(f"从本地FAISS索引删除单元 '{unit_id}' 失败: {e}")
        
        return success_neo4j and success_milvus
    
    def get_related_node_ids(
        self, 
        unit_id: str, 
        relationship_type: Optional[str] = None, 
        direction: str = "outgoing"
    ) -> List[str]:
        """
        获取与指定节点相关的节点ID列表
        
        参数:
            unit_id: 起始节点ID
            relationship_type: 关系类型，如果为None则获取所有类型的关系
            direction: 关系方向，"outgoing"表示出边，"incoming"表示入边，"all"表示所有
            
        返回:
            List[str]: 相关节点ID列表
        """
        if not self.neo4j_connected:
            logging.error("未连接到Neo4j服务器")
            return []
        
        try:
            with self.neo4j_driver.session(database=self.neo4j_database) as session:
                if direction == "outgoing":
                    if relationship_type:
                        query = (
                            f"MATCH (source:MemoryUnit {{uid: $id}})"
                            f"-[r:{relationship_type}]->"
                            f"(target:MemoryUnit) "
                            f"RETURN target.uid"
                        )
                    else:
                        query = (
                            "MATCH (source:MemoryUnit {uid: $id})"
                            "-[r]->"
                            "(target:MemoryUnit) "
                            "RETURN target.uid"
                        )
                elif direction == "incoming":
                    if relationship_type:
                        query = (
                            f"MATCH (source:MemoryUnit)"
                            f"-[r:{relationship_type}]->"
                            f"(target:MemoryUnit {{uid: $id}}) "
                            f"RETURN source.uid"
                        )
                    else:
                        query = (
                            "MATCH (source:MemoryUnit)"
                            "-[r]->"
                            "(target:MemoryUnit {uid: $id}) "
                            "RETURN source.uid"
                        )
                else:  # "all"
                    if relationship_type:
                        query = (
                            f"MATCH (node:MemoryUnit {{uid: $id}})-[r:{relationship_type}]-(other:MemoryUnit) "
                            f"RETURN other.uid"
                        )
                    else:
                        query = (
                            "MATCH (node:MemoryUnit {uid: $id})-[r]-(other:MemoryUnit) "
                            "RETURN other.uid"
                        )
                
                result = session.run(query, id=unit_id)
                related_ids = [record[0] for record in result]
                return related_ids
                
        except Exception as e:
            logging.error(f"获取与 '{unit_id}' 相关的节点ID失败: {e}")
            return []
    
    def get_related_units(
        self, 
        unit_id: str, 
        relationship_type: Optional[str] = None, 
        direction: str = "outgoing"
    ) -> List[MemoryUnit]:
        """
        获取与指定节点相关的完整MemoryUnit对象列表
        
        参数:
            unit_id: 起始节点ID
            relationship_type: 关系类型，如果为None则获取所有类型的关系
            direction: 关系方向，"outgoing"表示出边，"incoming"表示入边，"all"表示所有
            
        返回:
            List[MemoryUnit]: 相关MemoryUnit对象列表
        """
        # 先获取相关节点ID
        related_ids = self.get_related_node_ids(unit_id, relationship_type, direction)
        
        # 然后从Milvus获取完整的MemoryUnit对象
        if related_ids:
            return self.milvus_operator.get_units_batch(related_ids)
        return []
    
    def search_by_text(
        self,
        query_text: str,
        k: int = 5,
        space_name: Optional[str] = None,
        filter_ids: Optional[List[str]] = None
    ) -> List[Tuple[MemoryUnit, float]]:
        """
        基于文本在Milvus中进行相似性搜索
        
        参数:
            query_text: 查询文本
            k: 返回的最相似结果数量
            space_name: 如果提供，则仅在特定空间内搜索
            filter_ids: 如果提供，则只从这些ID中搜索
            
        返回:
            List[Tuple[MemoryUnit, float]]: [(memory_unit, distance), ...]
        """
        # 这个方法需要先将文本转换为向量
        # 理想情况下，这应该使用与Milvus中相同的嵌入模型
        # 这里假设嵌入模型是外部提供的，我们只处理已经嵌入的向量
        # 实际实现中，应该导入适当的模型进行嵌入
        
        try:
            # 使用适当的嵌入模型，这里使用简单模型示例
            model = SentenceTransformer('clip-ViT-B-32-multilingual-v1')
            query_vector = model.encode(query_text, normalize_embeddings=True)
            
            return self.search_by_vector(
                query_vector=query_vector,
                k=k,
                space_name=space_name,
                filter_ids=filter_ids
            )
        except Exception as e:
            logging.error(f"文本搜索失败: {e}")
            return []
    
    def search_by_vector(
        self,
        query_vector: np.ndarray,
        k: int = 5,
        space_name: Optional[str] = None,
        filter_ids: Optional[List[str]] = None
    ) -> List[Tuple[MemoryUnit, float]]:
        """
        基于向量在Milvus中进行相似性搜索
        
        参数:
            query_vector: 查询向量
            k: 返回的最相似结果数量
            space_name: 如果提供，则仅在特定空间内搜索
            filter_ids: 如果提供，则只从这些ID中搜索
            
        返回:
            List[Tuple[MemoryUnit, float]]: [(memory_unit, distance), ...]
        """
        # 将参数名从query_vector改为query_embedding
        return self.milvus_operator.search_similarity(
            query_embedding=query_vector,  # 这里修改为query_embedding
            k=k,
            space_name=space_name,
            filter_ids=filter_ids
        )
    
    def search_in_graph_context(
        self,
        query_text: str,
        context_node_id: str,
        relationship_type: Optional[str] = None,
        direction: str = "all",
        k: int = 5
    ) -> List[Tuple[MemoryUnit, float]]:
        """
        在图上下文中搜索：先获取相关节点，再在这些节点中进行语义搜索
        
        参数:
            query_text: 查询文本
            context_node_id: 上下文节点ID
            relationship_type: 关系类型过滤
            direction: 关系方向
            k: 返回结果数量
            
        返回:
            List[Tuple[MemoryUnit, float]]: [(memory_unit, distance), ...]
        """
        # 1. 先在Neo4j中获取与上下文节点相关的节点ID
        related_ids = self.get_related_node_ids(
            context_node_id, 
            relationship_type, 
            direction
        )
        
        if not related_ids:
            logging.info(f"节点 '{context_node_id}' 没有相关节点，将进行全局搜索")
            return self.search_by_text(query_text, k)
        
        # 2. 然后在这些相关节点中进行语义搜索
        return self.search_by_text(
            query_text=query_text,
            k=k,
            filter_ids=related_ids
        )
    
    def search_by_path_and_similarity(
        self,
        query_text: str,
        start_node_id: str,
        max_path_length: int = 3,
        k: int = 5
    ) -> List[Tuple[MemoryUnit, float, List[str]]]:
        """
        结合路径分析和相似度搜索
        
        参数:
            query_text: 查询文本
            start_node_id: 起始节点ID
            max_path_length: 最大路径长度
            k: 返回结果数量
            
        返回:
            List[Tuple[MemoryUnit, float, List[str]]]: [(内存单元, 相似度, 路径), ...]
        """
        if not self.neo4j_connected:
            logging.error("未连接到Neo4j服务器")
            return []
        
        try:
            # 1. 查找与起始节点相距指定步数内的所有节点
            with self.neo4j_driver.session(database=self.neo4j_database) as session:
                query = (
                    f"MATCH path = (start:MemoryUnit {{uid: $start_id}})-[*1..{max_path_length}]-(target:MemoryUnit) "
                    f"RETURN target.uid, [node IN nodes(path) | node.uid] AS path_nodes"
                )
                result = session.run(query, start_id=start_node_id)
                
                # 收集所有可达节点及其路径
                reachable_nodes = {}  # {node_id: path}
                for record in result:
                    node_id = record[0]
                    path = record[1]
                    if node_id not in reachable_nodes:
                        reachable_nodes[node_id] = path
            
            if not reachable_nodes:
                logging.info(f"节点 '{start_node_id}' 没有可达节点，将进行全局搜索")
                results = self.search_by_text(query_text, k)
                return [(unit, score, []) for unit, score in results]
            
            # 2. 在可达节点中进行语义搜索
            results = self.search_by_text(
                query_text=query_text,
                k=k,
                filter_ids=list(reachable_nodes.keys())
            )
            
            # 3. 将路径信息添加到结果中
            enriched_results = []
            for unit, score in results:
                path = reachable_nodes.get(unit.uid, [])
                enriched_results.append((unit, score, path))
            
            return enriched_results
                
        except Exception as e:
            logging.error(f"路径与相似度结合搜索失败: {e}")
            return []
    
    def get_all_node_ids(self) -> List[str]:
        """获取数据库中所有节点的ID列表"""
        try:
            with self.neo4j_driver.session(database=self.neo4j_database) as session:
                result = session.run("MATCH (n) RETURN n.uid AS uid")
                node_ids = [record["uid"] for record in result if record["uid"] is not None]
                logging.info(f"从Neo4j获取了 {len(node_ids)} 个节点ID")
                return node_ids
        except Exception as e:
            logging.error(f"获取所有节点ID失败: {e}")
            return []

    def node_exists(self, uid: str) -> bool:
        """检查节点是否存在"""
        try:
            with self.neo4j_driver.session(database=self.neo4j_database) as session:
                result = session.run(
                    "MATCH (n {uid: $uid}) RETURN COUNT(n) AS count",
                    uid=uid
                )
                count = result.single()["count"]
                return count > 0
        except Exception as e:
            logging.error(f"检查节点存在性失败: {e}")
            return False

    def get_node_by_uid(self, uid: str) -> Optional[Dict[str, Any]]:
        """根据UID获取节点数据"""
        try:
            with self.neo4j_driver.session(database=self.neo4j_database) as session:
                result = session.run(
                    "MATCH (n {uid: $uid}) RETURN n",
                    uid=uid
                )
                record = result.single()
                if record:
                    node = dict(record["n"])
                    return node
                return None
        except Exception as e:
            logging.error(f"获取节点失败: {e}")
            return None

    def get_nodes_by_uids(self, uids: List[str]) -> List[Dict[str, Any]]:
        """批量获取节点数据"""
        try:
            with self.neo4j_driver.session(database=self.neo4j_database) as session:
                result = session.run(
                    "MATCH (n) WHERE n.uid IN $uids RETURN n",
                    uids=uids
                )
                nodes = [dict(record["n"]) for record in result]
                logging.info(f"批量获取了 {len(nodes)} 个节点")
                return nodes
        except Exception as e:
            logging.error(f"批量获取节点失败: {e}")
            return []

    def delete_node(self, uid: str) -> bool:
        """删除节点及其所有关系"""
        try:
            with self.neo4j_driver.session(database=self.neo4j_database) as session:
                result = session.run(
                    "MATCH (n {uid: $uid}) DETACH DELETE n RETURN COUNT(n) AS deleted",
                    uid=uid
                )
                deleted_count = result.single()["deleted"]
                success = deleted_count > 0
                if success:
                    logging.info(f"成功删除节点: {uid}")
                return success
        except Exception as e:
            logging.error(f"删除节点失败: {e}")
            return False

    def get_node_relationships(self, uid: str) -> List[Dict[str, Any]]:
        """获取节点的所有关系"""
        try:
            with self.neo4j_driver.session(database=self.neo4j_database) as session:
                result = session.run("""
                    MATCH (n {uid: $uid})-[r]-(m)
                    RETURN 
                        n.uid AS source_uid,
                        type(r) AS relationship_type,
                        m.uid AS target_uid,
                        r AS relationship_properties
                """, uid=uid)
                
                relationships = []
                for record in result:
                    rel_data = {
                        "source_uid": record["source_uid"],
                        "target_uid": record["target_uid"],
                        "relationship_type": record["relationship_type"],
                        "properties": dict(record["relationship_properties"]) if record["relationship_properties"] else {}
                    }
                    relationships.append(rel_data)
                
                return relationships
        except Exception as e:
            logging.error(f"获取节点关系失败: {e}")
            return []

    def clear_all_data(self) -> bool:
        """清除数据库中所有数据"""
        try:
            with self.neo4j_driver.session(database=self.neo4j_database) as session:
                # 删除所有节点和关系
                session.run("MATCH (n) DETACH DELETE n")
                logging.info("已清除Neo4j数据库中的所有数据")
                return True
        except Exception as e:
            logging.error(f"清除数据失败: {e}")
            return False

    def get_database_stats(self) -> Dict[str, int]:
        """获取数据库统计信息"""
        try:
            with self.neo4j_driver.session(database=self.neo4j_database) as session:
                # 获取节点数量
                node_result = session.run("MATCH (n) RETURN COUNT(n) AS node_count")
                node_count = node_result.single()["node_count"]
                
                # 获取关系数量
                rel_result = session.run("MATCH ()-[r]->() RETURN COUNT(r) AS rel_count")
                rel_count = rel_result.single()["rel_count"]
                
                return {
                    "nodes": node_count,
                    "relationships": rel_count
                }
        except Exception as e:
            logging.error(f"获取数据库统计失败: {e}")
            return {"nodes": 0, "relationships": 0}

    def rebuild_local_faiss_index(self):
        """
        重建本地FAISS索引
        """
        if not self.use_local_faiss:
            logging.warning("本地FAISS索引未启用")
            return
        
        try:
            # 从Milvus获取所有单元
            all_units = self.milvus_operator.get_units_batch([])
            
            if not all_units:
                logging.info("没有单元可用于重建FAISS索引")
                return
            
            # 重置FAISS索引
            self._init_faiss_index()
            self.faiss_id_to_unit_id = {}
            self.unit_id_to_faiss_id = {}
            self.faiss_id_counter = 0
            
            # 添加所有向量
            vectors = []
            ids = []
            for unit in all_units:
                if unit.embedding is not None:
                    vectors.append(unit.embedding.astype(np.float32))
                    faiss_id = self.faiss_id_counter
                    self.faiss_id_counter += 1
                    ids.append(faiss_id)
                    self.faiss_id_to_unit_id[faiss_id] = unit.uid
                    self.unit_id_to_faiss_id[unit.uid] = faiss_id
            
            if vectors:
                vectors_np = np.array(vectors)
                ids_np = np.array(ids, dtype=np.int64)
                
                if hasattr(self.faiss_index, 'add_with_ids'):
                    self.faiss_index.add_with_ids(vectors_np, ids_np)
                else:
                    self.faiss_index.add(vectors_np)
                
                logging.info(f"本地FAISS索引已重建，包含 {len(vectors)} 个向量")
            else:
                logging.info("没有有效向量可用于重建FAISS索引")
            
        except Exception as e:
            logging.error(f"重建本地FAISS索引失败: {e}")
    
    def search_local_faiss(self, query_vector: np.ndarray, k: int = 5) -> List[Tuple[str, float]]:
        """
        在本地FAISS索引中搜索
        
        参数:
            query_vector: 查询向量
            k: 返回结果数量
            
        返回:
            List[Tuple[str, float]]: [(单元ID, 距离), ...]
        """
        if not self.use_local_faiss or self.faiss_index is None:
            logging.warning("本地FAISS索引未启用或未初始化")
            return []
        
        try:
            # 确保查询向量格式正确
            query_vector = query_vector.reshape(1, -1).astype(np.float32)
            
            # 执行搜索
            distances, indices = self.faiss_index.search(query_vector, k)
            
            results = []
            for i in range(indices.shape[1]):
                idx = indices[0, i]
                if idx != -1 and idx in self.faiss_id_to_unit_id:
                    unit_id = self.faiss_id_to_unit_id[idx]
                    distance = distances[0, i]
                    results.append((unit_id, float(distance)))
            
            return results
            
        except Exception as e:
            logging.error(f"本地FAISS搜索失败: {e}")
            return []
    
    def close(self):
        """关闭所有连接"""
        # 关闭Neo4j连接
        if self.neo4j_connected:
            try:
                self.neo4j_driver.close()
                logging.info("已关闭Neo4j连接")
            except Exception as e:
                logging.error(f"关闭Neo4j连接失败: {e}")
        
        # 关闭Milvus连接
        self.milvus_operator.close()

if __name__ == "__main__":
    # 示例用法
    enhanced_op = Neo4jOperator(
        neo4j_uri ="bolt://localhost:7687",
        neo4j_user = "neo4j", 
        neo4j_password = "20031117",
        neo4j_database = "academicgraph",
        milvus_port="19530"
    )

    unit1 = MemoryUnit(
        uid="doc1", 
        raw_data={"title": "人工智能简介", "content": "AI是计算机科学的一个分支..."}, 
        embedding=np.random.rand(512),
        metadata={}  # 添加空字典作为metadata
    )
    unit2 = MemoryUnit(
        uid="doc2", 
        raw_data={"title": "机器学习基础", "content": "机器学习是AI的核心技术..."}, 
        embedding=np.random.rand(512),
        metadata={}  # 添加空字典作为metadata
    )
    unit3 = MemoryUnit(
        uid="doc3", 
        raw_data={"title": "深度学习进展", "content": "深度学习近年来取得了巨大突破..."}, 
        embedding=np.random.rand(512),
        metadata={}  # 添加空字典作为metadata
    )

    # 调试打印，检查原始数据是否正确
    print(f"Unit1 raw_data: {unit1.raw_data}")
    
    # 添加单元到数据库
    enhanced_op.add_unit(unit1, space_names=["AI文档"])
    enhanced_op.add_unit(unit2, space_names=["AI文档", "技术文档"])
    enhanced_op.add_unit(unit3, space_names=["AI文档", "研究进展"])

    # 添加关系 - 使用uid而不是id
    enhanced_op.add_relationship(unit1.uid, unit2.uid, "RELATED_TO", {"relevance": 0.9})
    enhanced_op.add_relationship(unit2.uid, unit3.uid, "PREREQUISITE_FOR")

    # 查询示例1: 基于关系查询
    related_units = enhanced_op.get_related_units(unit1.uid, relationship_type="RELATED_TO")
    print("与文档1相关的单元:", [u.uid for u in related_units])
    if related_units:
        print(f"检索到的相关单元raw_data: {related_units[0].raw_data}")
    
    # 查询示例2: 混合查询 - 先关系后相似度
    context_results = enhanced_op.search_in_graph_context(
        query_text="深度学习的最新进展是什么？",
        context_node_id=unit1.uid,
        k=2
    )
    print("\n在文档1的图上下文中搜索:")
    for unit, score in context_results:
        print(f"- {unit.uid}: {unit.raw_data} (分数: {score:.4f})")
        print(f"  标题: {unit.raw_data.get('title', '未找到标题')}")

    # 查询示例3: 路径与相似度结合
    path_results = enhanced_op.search_by_path_and_similarity(
        query_text="神经网络",
        start_node_id=unit1.uid,
        max_path_length=2,
        k=2
    )
    print("\n基于路径和相似度的搜索:")
    for unit, score, path in path_results:
        print(f"- {unit.uid}: 完整数据: {unit.raw_data} (分数: {score:.4f})")
        print(f"  标题: {unit.raw_data.get('title', '未找到标题')}")
        print(f"  路径: {' -> '.join(path)}")

    # 关闭连接
    enhanced_op.close()