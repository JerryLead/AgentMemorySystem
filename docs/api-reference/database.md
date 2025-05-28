# 数据库操作接口文档

本文档描述了 AgentMemorySystem 中与数据库操作相关的 API，包括换入换出、外部存储连接和同步等功能。

## SemanticMap 数据库操作

### 外部存储连接

#### `connect_external_storage(storage_type, host, port, user, password, collection_name)`
连接到外部存储系统
- **参数**:
  - `storage_type` (str): 存储类型，当前支持 "milvus"
  - `host` (str): 存储服务器地址，默认 "localhost"
  - `port` (str): 存储服务器端口，默认 "19530"
  - `user` (str): 用户名，默认 ""
  - `password` (str): 密码，默认 ""
  - `collection_name` (str): 集合名称，默认 "hippo_memory_units"
- **返回**: bool - 连接是否成功

### 数据同步

#### `sync_to_external(force_full_sync)`
将修改过的单元同步到外部存储
- **参数**:
  - `force_full_sync` (bool): 是否强制全量同步，默认 False
- **返回**: (success_count, fail_count) - 成功和失败的单元数量

#### `load_from_external(filter_space, limit, replace_existing)`
从外部存储加载单元到内存
- **参数**:
  - `filter_space` (Optional[str]): 空间过滤器，默认 None
  - `limit` (int): 加载限制数量，默认 1000
  - `replace_existing` (bool): 是否替换已存在的单元，默认 False
- **返回**: int - 加载的单元数量

### 内存管理

#### `_page_out_least_used_units(count)`
将最少使用的单元从内存移出到外部存储
- **参数**:
  - `count` (int): 要移出的单元数量，默认 100

#### `_load_unit_from_external(uid)`
从外部存储加载单元到内存
- **参数**:
  - `uid` (str): 单元唯一标识符
- **返回**: Optional[MemoryUnit] - 加载的内存单元对象

### 数据导出

#### `export_to_milvus(host, port, user, password, collection_name)`
将内存单元导出到 Milvus 数据库
- **参数**:
  - `host` (str): Milvus 服务器地址，默认 "localhost"
  - `port` (str): Milvus 服务器端口，默认 "19530"
  - `user` (str): 用户名，默认 ""
  - `password` (str): 密码，默认 ""
  - `collection_name` (str): 集合名称，默认 "hippo_memory_units"
- **返回**: bool - 导出是否成功

## SemanticGraph 数据库操作

### 外部存储连接

#### `connect_to_neo4j(uri, user, password, database, milvus_host, milvus_port, milvus_user, milvus_password, milvus_collection)`
连接到 Neo4j 数据库，同时设置 Milvus 连接
- **参数**:
  - `uri` (str): Neo4j 连接 URI，默认 "bolt://localhost:7687"
  - `user` (str): Neo4j 用户名，默认 "neo4j"
  - `password` (str): Neo4j 密码，默认 "password"
  - `database` (str): Neo4j 数据库名，默认 "neo4j"
  - `milvus_host` (str): Milvus 服务器地址，默认 "localhost"
  - `milvus_port` (str): Milvus 服务器端口，默认 "19530"
  - `milvus_user` (str): Milvus 用户名，默认 ""
  - `milvus_password` (str): Milvus 密码，默认 ""
  - `milvus_collection` (str): Milvus 集合名，默认 "hippo_memory_units"
- **返回**: bool - 连接是否成功

### 节点内存管理

#### `page_out_nodes(count, strategy)`
将不常用节点从内存移出到外部存储
- **参数**:
  - `count` (int): 要移出的节点数量，默认 100
  - `strategy` (str): 换出策略，支持 "LRU" 和 "LFU"，默认 "LRU"
- **返回**: int - 实际移出的节点数量

#### `page_in_nodes(node_ids)`
从外部存储加载节点到内存
- **参数**:
  - `node_ids` (List[str]): 要加载的节点 ID 列表
- **返回**: int - 成功加载的节点数量

### 关系管理

#### `get_relationship(source_uid, target_uid, relationship_name)`
获取关系属性，必要时从 Neo4j 加载
- **参数**:
  - `source_uid` (str): 源节点 ID
  - `target_uid` (str): 目标节点 ID
  - `relationship_name` (Optional[str]): 关系名称，默认 None
- **返回**: Dict - 关系属性字典

### 数据同步

#### `sync_to_external(force_full_sync)`
将修改同步到外部存储（Neo4j 和 Milvus）
- **参数**:
  - `force_full_sync` (bool): 是否强制全量同步，默认 False
- **返回**: Dict[str, int] - 同步统计信息

#### `incremental_export()`
增量导出修改过的节点和关系到外部存储
- **返回**: Dict[str, int] - 导出统计信息

#### `full_export()`
完整导出所有节点和关系到外部存储
- **返回**: Dict[str, int] - 导出统计信息

### 数据导出

#### `export_to_neo4j(uri, user, password, database)`
将图谱中的节点和关系导出到 Neo4j 数据库
- **参数**:
  - `uri` (str): Neo4j 连接 URI，默认 "bolt://localhost:7687"
  - `user` (str): Neo4j 用户名，默认 "neo4j"
  - `password` (str): Neo4j 密码，默认 "password"
  - `database` (str): Neo4j 数据库名，默认 "neo4j"
- **返回**: bool - 导出是否成功

## 注意事项

1. 在使用外部存储功能之前，需要先调用相应的连接方法
2. 换入换出操作会影响内存使用量，建议根据系统资源合理配置
3. 同步操作可能耗时较长，建议在适当时机执行
4. 导出操作会覆盖目标数据库中的现有数据，请谨慎使用
