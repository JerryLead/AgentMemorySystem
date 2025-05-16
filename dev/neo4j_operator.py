import logging
from typing import Dict, List, Optional, Set, Tuple, Any
import json
from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError

from Hippo import MemoryUnit

class Neo4jOperator:
    """Neo4j图数据库操作类，提供图数据的存储和检索功能"""
    
    def __init__(
        self, 
        uri: str = "bolt://localhost:7687", 
        user: str = "neo4j", 
        password: str = "password",
        database: str = "neo4j"
    ):
        """
        初始化Neo4j操作类
        
        参数:
            uri: Neo4j服务器URI
            user: 用户名
            password: 密码
            database: 数据库名称
        """
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self.is_connected = False
        
        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            # 测试连接
            with self.driver.session(database=database) as session:
                session.run("RETURN 1")
            self.is_connected = True
            logging.info(f"成功连接到Neo4j服务器: {uri}")
        except Exception as e:
            logging.error(f"连接到Neo4j服务器失败: {e}")
            self.is_connected = False
    
    def add_memory_unit(self, unit: MemoryUnit) -> bool:
        """
        添加MemoryUnit到Neo4j
        
        参数:
            unit: 要添加的MemoryUnit对象
        
        返回:
            添加是否成功
        """
        if not self.is_connected:
            logging.error("未连接到Neo4j服务器")
            return False
        
        try:
            # 将value字典转换为适合Neo4j的属性
            properties = {}
            for k, v in unit.value.items():
                # 处理复杂类型
                if isinstance(v, (dict, list)):
                    properties[k] = json.dumps(v)
                else:
                    properties[k] = v
            
            # 添加ID属性
            properties["id"] = unit.id
            
            with self.driver.session(database=self.database) as session:
                # 使用MERGE确保节点唯一性
                query = (
                    "MERGE (n:MemoryUnit {id: $id}) "
                    "SET n = $properties "
                    "RETURN n"
                )
                result = session.run(
                    query, 
                    id=unit.id, 
                    properties=properties
                )
                summary = result.consume()
                created = summary.counters.nodes_created
                set_props = summary.counters.properties_set
                
                if created > 0:
                    logging.info(f"成功创建内存单元节点 '{unit.id}'")
                else:
                    logging.info(f"成功更新内存单元节点 '{unit.id}' ({set_props} 个属性)")
                
                return True
                
        except Exception as e:
            logging.error(f"添加内存单元 '{unit.id}' 到Neo4j失败: {e}")
            return False
    
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
        if not self.is_connected:
            logging.error("未连接到Neo4j服务器")
            return False
        
        try:
            # 处理属性
            props = {}
            if properties:
                for k, v in properties.items():
                    if isinstance(v, (dict, list)):
                        props[k] = json.dumps(v)
                    else:
                        props[k] = v
            
            with self.driver.session(database=self.database) as session:
                # 首先确保两个节点都存在
                query = (
                    "MATCH (source:MemoryUnit {id: $source_id}), "
                    "(target:MemoryUnit {id: $target_id}) "
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
                created = summary.counters.relationships_created
                
                if created > 0:
                    logging.info(f"成功创建从 '{source_id}' 到 '{target_id}' 的 '{relationship_type}' 关系")
                else:
                    logging.info(f"成功更新从 '{source_id}' 到 '{target_id}' 的 '{relationship_type}' 关系")
                
                return True
                
        except Exception as e:
            logging.error(f"添加从 '{source_id}' 到 '{target_id}' 的 '{relationship_type}' 关系失败: {e}")
            return False
    
    def delete_memory_unit(self, unit_id: str) -> bool:
        """删除Neo4j中的MemoryUnit及其相关关系"""
        if not self.is_connected:
            logging.error("未连接到Neo4j服务器")
            return False
        
        try:
            with self.driver.session(database=self.database) as session:
                query = (
                    "MATCH (n:MemoryUnit {id: $id}) "
                    "DETACH DELETE n"
                )
                result = session.run(query, id=unit_id)
                summary = result.consume()
                deleted = summary.counters.nodes_deleted
                
                if deleted > 0:
                    logging.info(f"成功从Neo4j删除内存单元 '{unit_id}' 及其关系")
                else:
                    logging.info(f"内存单元 '{unit_id}' 不存在于Neo4j中")
                
                return True
                
        except Exception as e:
            logging.error(f"从Neo4j删除内存单元 '{unit_id}' 失败: {e}")
            return False
    
    def delete_relationship(self, source_id: str, target_id: str, relationship_type: Optional[str] = None) -> bool:
        """删除两个节点之间的关系"""
        if not self.is_connected:
            logging.error("未连接到Neo4j服务器")
            return False
        
        try:
            with self.driver.session(database=self.database) as session:
                if relationship_type:
                    # 删除特定类型的关系
                    query = (
                        "MATCH (source:MemoryUnit {id: $source_id})"
                        "-[r:" + relationship_type + "]->"
                        "(target:MemoryUnit {id: $target_id}) "
                        "DELETE r"
                    )
                else:
                    # 删除所有类型的关系
                    query = (
                        "MATCH (source:MemoryUnit {id: $source_id})"
                        "-[r]->"
                        "(target:MemoryUnit {id: $target_id}) "
                        "DELETE r"
                    )
                
                result = session.run(
                    query, 
                    source_id=source_id, 
                    target_id=target_id
                )
                summary = result.consume()
                deleted = summary.counters.relationships_deleted
                
                rel_desc = f"'{relationship_type}'" if relationship_type else "所有"
                if deleted > 0:
                    logging.info(f"成功删除从 '{source_id}' 到 '{target_id}' 的 {rel_desc} 关系")
                else:
                    logging.info(f"从 '{source_id}' 到 '{target_id}' 的 {rel_desc} 关系不存在")
                
                return True
                
        except Exception as e:
            rel_desc = f"'{relationship_type}'" if relationship_type else "所有"
            logging.error(f"删除从 '{source_id}' 到 '{target_id}' 的 {rel_desc} 关系失败: {e}")
            return False
    
    def get_memory_unit(self, unit_id: str) -> Optional[Dict[str, Any]]:
        """通过ID从Neo4j获取MemoryUnit"""
        if not self.is_connected:
            logging.error("未连接到Neo4j服务器")
            return None
        
        try:
            with self.driver.session(database=self.database) as session:
                query = (
                    "MATCH (n:MemoryUnit {id: $id}) "
                    "RETURN n"
                )
                result = session.run(query, id=unit_id)
                record = result.single()
                
                if not record:
                    return None
                
                # 将节点属性转换为字典
                properties = dict(record[0])
                
                # 解析可能的JSON字符串
                for k, v in properties.items():
                    if isinstance(v, str) and (v.startswith('{') or v.startswith('[')):
                        try:
                            properties[k] = json.loads(v)
                        except:
                            pass  # 如果不是有效的JSON，保持原样
                
                return properties
                
        except Exception as e:
            logging.error(f"从Neo4j获取内存单元 '{unit_id}' 失败: {e}")
            return None
    
    def get_related_units(
        self, 
        unit_id: str, 
        relationship_type: Optional[str] = None, 
        direction: str = "outgoing"
    ) -> List[Tuple[str, Dict[str, Any], Dict[str, Any]]]:
        """
        获取与指定节点相关的节点
        
        参数:
            unit_id: 起始节点ID
            relationship_type: 关系类型，如果为None则获取所有类型的关系
            direction: 关系方向，"outgoing"表示出边，"incoming"表示入边，"all"表示所有
            
        返回:
            List[Tuple[str, Dict[str, Any], Dict[str, Any]]]: 
            [(相关节点ID, 节点属性, 关系属性), ...]
        """
        if not self.is_connected:
            logging.error("未连接到Neo4j服务器")
            return []
        
        try:
            with self.driver.session(database=self.database) as session:
                if direction == "outgoing":
                    if relationship_type:
                        # 特定类型的出边
                        query = (
                            "MATCH (source:MemoryUnit {id: $id})"
                            "-[r:" + relationship_type + "]->"
                            "(target:MemoryUnit) "
                            "RETURN target.id, target, properties(r)"
                        )
                    else:
                        # 所有类型的出边
                        query = (
                            "MATCH (source:MemoryUnit {id: $id})"
                            "-[r]->"
                            "(target:MemoryUnit) "
                            "RETURN target.id, target, properties(r)"
                        )
                elif direction == "incoming":
                    if relationship_type:
                        # 特定类型的入边
                        query = (
                            "MATCH (source:MemoryUnit)"
                            "-[r:" + relationship_type + "]->"
                            "(target:MemoryUnit {id: $id}) "
                            "RETURN source.id, source, properties(r)"
                        )
                    else:
                        # 所有类型的入边
                        query = (
                            "MATCH (source:MemoryUnit)"
                            "-[r]->"
                            "(target:MemoryUnit {id: $id}) "
                            "RETURN source.id, source, properties(r)"
                        )
                else:  # "all"
                    if relationship_type:
                        # 特定类型的所有边
                        query = (
                            "MATCH (node:MemoryUnit {id: $id}), (other:MemoryUnit) "
                            "WHERE (node)-[r:" + relationship_type + "]->(other) "
                            "OR (other)-[r:" + relationship_type + "]->(node) "
                            "RETURN other.id, other, properties(r)"
                        )
                    else:
                        # 所有类型的所有边
                        query = (
                            "MATCH (node:MemoryUnit {id: $id}), (other:MemoryUnit) "
                            "WHERE (node)-[r]->(other) OR (other)-[r]->(node) "
                            "RETURN other.id, other, properties(r)"
                        )
                
                result = session.run(query, id=unit_id)
                
                related_units = []
                for record in result:
                    related_id = record[0]
                    node_properties = dict(record[1])
                    rel_properties = record[2]
                    
                    # 解析可能的JSON字符串
                    for k, v in node_properties.items():
                        if isinstance(v, str) and (v.startswith('{') or v.startswith('[')):
                            try:
                                node_properties[k] = json.loads(v)
                            except:
                                pass
                    
                    for k, v in rel_properties.items():
                        if isinstance(v, str) and (v.startswith('{') or v.startswith('[')):
                            try:
                                rel_properties[k] = json.loads(v)
                            except:
                                pass
                    
                    related_units.append((related_id, node_properties, rel_properties))
                
                return related_units
                
        except Exception as e:
            logging.error(f"获取与 '{unit_id}' 相关的节点失败: {e}")
            return []
    
    def close(self):
        """关闭Neo4j连接"""
        if self.is_connected:
            try:
                self.driver.close()
                logging.info("已关闭Neo4j连接")
            except Exception as e:
                logging.error(f"关闭Neo4j连接失败: {e}")

if __name__ == "__main__":
    # 测试Neo4jOperator类
    neo4j_operator = Neo4jOperator()
    
    # 创建一个MemoryUnit对象
    unit = MemoryUnit(id="test_unit", value={"name": "测试单元", "description": "这是一个测试单元"})
    
    # 添加MemoryUnit到Neo4j
    neo4j_operator.add_memory_unit(unit)
    
    # 添加关系
    neo4j_operator.add_relationship("test_unit", "another_unit", "RELATED_TO", {"weight": 1.0})
    
    # 获取MemoryUnit
    retrieved_unit = neo4j_operator.get_memory_unit("test_unit")
    print(retrieved_unit)
    
    # 获取相关节点
    related_units = neo4j_operator.get_related_units("test_unit")
    print(related_units)
    
    # 删除关系
    neo4j_operator.delete_relationship("test_unit", "another_unit")
    
    # 删除MemoryUnit
    neo4j_operator.delete_memory_unit("test_unit")
    
    # 关闭连接
    neo4j_operator.close()