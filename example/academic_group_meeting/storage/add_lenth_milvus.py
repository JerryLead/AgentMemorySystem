from pymilvus import connections, Collection, utility, FieldSchema, CollectionSchema, DataType
import traceback
import numpy as np

def update_collection_varchar_length(
    host="localhost", 
    port="19530",
    db_name="default",
    old_collection_name="academic_dialogues",
    new_collection_name=None,  # 如果为None，则替换原集合
    content_field_max_length=65535  # 新的长度限制
):
    """
    更新Milvus集合中VARCHAR字段的最大长度
    
    Args:
        host: Milvus服务器地址
        port: Milvus服务器端口
        db_name: 数据库名称
        old_collection_name: 原集合名称
        new_collection_name: 新集合名称，如为None则替换原集合
        content_field_max_length: 内容字段新的最大长度
    """
    try:
        # 连接到Milvus
        connections.connect(alias="default", host=host, port=port, db_name=db_name)
        print(f"已连接到Milvus服务器: {host}:{port}")
        
        # 检查原集合是否存在
        if not utility.has_collection(old_collection_name):
            print(f"错误: 集合 {old_collection_name} 不存在")
            return False
        
        # 获取原集合信息
        old_collection = Collection(old_collection_name)
        old_schema = old_collection.schema
        old_fields = old_schema.fields
        
        # 创建新集合名称(如果未指定)
        if new_collection_name is None:
            new_collection_name = f"{old_collection_name}_new"
            replace_old = True
        else:
            replace_old = False
        
        # 检查新集合名称是否已存在
        if utility.has_collection(new_collection_name):
            print(f"警告: 集合 {new_collection_name} 已存在，将删除")
            utility.drop_collection(new_collection_name)
        
        # 创建新的字段列表，修改VARCHAR字段的max_length
        new_fields = []
        for field in old_fields:
            if field.dtype == DataType.VARCHAR and field.name == "content":
                # 修改content字段的max_length
                print(f"修改字段 {field.name} 的max_length从 {field.params['max_length']} 到 {content_field_max_length}")
                new_field = FieldSchema(
                    name=field.name,
                    dtype=field.dtype,
                    is_primary=field.is_primary,
                    description=field.description,
                    max_length=content_field_max_length
                )
            else:
                # 其他字段保持不变
                new_field = field
            new_fields.append(new_field)
        
        # 创建新的集合schema
        new_schema = CollectionSchema(
            fields=new_fields,
            description=old_schema.description
        )
        
        # 创建新集合
        new_collection = Collection(new_collection_name, new_schema)
        print(f"已创建新集合 {new_collection_name}")
        
        # 为集合创建索引
        if old_collection.has_index():
            index_infos = old_collection.index()
            for index_info in index_infos:
                field_name = index_info.field_name
                index_params = index_info.params
                print(f"为字段 {field_name} 创建索引: {index_params}")
                new_collection.create_index(field_name, index_params)
        
        # 迁移数据
        print("开始迁移数据...")
        
        # 加载原集合
        old_collection.load()
        
        # 读取原集合数据并分批插入到新集合
        batch_size = 1000
        total = old_collection.num_entities
        
        for offset in range(0, total, batch_size):
            limit = min(batch_size, total - offset)
            query_results = old_collection.query(expr="id >= 0", limit=limit, offset=offset)
            
            if query_results:
                # 准备插入数据
                entities = []
                for field in new_fields:
                    field_values = []
                    for entity in query_results:
                        if field.name in entity:
                            value = entity[field.name]
                            # 特殊处理嵌入向量
                            if field.dtype == DataType.FLOAT_VECTOR and isinstance(value, list):
                                # 确保向量格式正确
                                value = np.array(value, dtype=np.float32).tolist()
                            field_values.append(value)
                        else:
                            # 如果实体中没有此字段，添加默认值
                            if field.dtype == DataType.VARCHAR:
                                field_values.append("")
                            elif field.dtype == DataType.INT64:
                                field_values.append(0)
                            elif field.dtype == DataType.FLOAT_VECTOR:
                                dim = field.params.get('dim', 768)
                                field_values.append(np.zeros(dim, dtype=np.float32).tolist())
                    entities.append(field_values)
                
                # 插入数据到新集合
                new_collection.insert(entities)
                print(f"已迁移 {offset + len(query_results)}/{total} 条记录")
        
        # 创建完毕后自动加载
        new_collection.load()
        print(f"数据迁移完成，共 {new_collection.num_entities} 条记录")
        
        # 如果选择替换原集合，则删除原集合并重命名新集合
        if replace_old:
            print(f"准备替换原集合 {old_collection_name}...")
            old_collection.release()  # 释放原集合
            utility.drop_collection(old_collection_name)  # 删除原集合
            utility.rename_collection(new_collection_name, old_collection_name)  # 重命名新集合
            print(f"成功替换集合 {old_collection_name}")
        
        print("集合VARCHAR字段长度更新完成")
        return True
    
    except Exception as e:
        print(f"更新Milvus集合时出错: {str(e)}")
        traceback.print_exc()
        return False
    finally:
        # 断开连接
        connections.disconnect("default")
        print("已断开Milvus连接")

# 执行更新
if __name__ == "__main__":
    # 更新academic_dialogues集合
    update_collection_varchar_length(
        old_collection_name="academic_dialogues",
        content_field_max_length=65535  # 增加到约64KB
    )
    
    # 更新summaries集合
    update_collection_varchar_length(
        old_collection_name="summaries",
        content_field_max_length=65535  # 增加到约64KB
    )