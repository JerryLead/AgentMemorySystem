import logging
import argparse
from neo4j import GraphDatabase
from pymilvus import connections, utility

# 配置日志记录器
logging.basicConfig(
    format="%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO
)

def test_neo4j_connection(uri="bolt://localhost:7687", user="neo4j", password="neo4j", database="academicgraph"):
    """
    测试与Neo4j数据库的连接
    
    参数:
        uri: Neo4j服务器URI
        user: 用户名
        password: 密码
        database: 数据库名称
    
    返回:
        bool: 连接是否成功
    """
    try:
        logging.info(f"正在连接Neo4j服务器: {uri}, 数据库: {database}")
        driver = GraphDatabase.driver(uri, auth=(user, password))
        
        # 测试连接
        with driver.session(database=database) as session:
            result = session.run("RETURN 'Neo4j连接成功!' AS message")
            message = result.single()[0]
            logging.info(f"Neo4j测试结果: {message}")
        
        # 获取数据库信息
        with driver.session(database=database) as session:
            result = session.run("CALL db.info()")
            db_info = result.single().data()
            logging.info(f"Neo4j数据库信息: {db_info}")
        
        driver.close()
        logging.info("Neo4j连接测试成功")
        return True
    
    except Exception as e:
        logging.error(f"Neo4j连接测试失败: {e}")
        return False

def test_milvus_connection(host="localhost", port="19530", user="", password=""):
    """
    测试与Milvus向量数据库的连接
    
    参数:
        host: Milvus服务器地址
        port: Milvus服务器端口
        user: 用户名（如果需要认证）
        password: 密码（如果需要认证）
    
    返回:
        bool: 连接是否成功
    """
    try:
        logging.info(f"正在连接Milvus服务器: {host}:{port}")
        connections.connect(
            alias="default", 
            host=host, 
            port=port,
            user=user,
            password=password
        )
        
        # 测试连接
        if connections.has_connection("default"):
            logging.info("Milvus连接建立成功")
            
            # 获取集合列表
            collections = utility.list_collections()
            logging.info(f"Milvus集合列表: {collections}")
            
            # 获取Milvus服务器状态
            status = utility.get_server_version()
            logging.info(f"Milvus服务器版本: {status}")
            
            connections.disconnect("default")
            logging.info("Milvus连接已关闭")
            return True
        else:
            logging.error("Milvus连接失败")
            return False
            
    except Exception as e:
        logging.error(f"Milvus连接测试失败: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='测试数据库连接')
    
    # Neo4j参数
    parser.add_argument('--neo4j_uri', default='bolt://localhost:7687', help='Neo4j服务器URI')
    parser.add_argument('--neo4j_user', default='neo4j', help='Neo4j用户名')
    # parser.add_argument('--neo4j_password', default='20031117', help='Neo4j密码')
    parser.add_argument('--neo4j_password', default='neo4j', help='Neo4j密码')
    parser.add_argument('--neo4j_database', default='academicgraph', help='Neo4j数据库名称')
    
    # Milvus参数
    parser.add_argument('--milvus_host', default='localhost', help='Milvus服务器地址')
    parser.add_argument('--milvus_port', default='19530', help='Milvus服务器端口')
    parser.add_argument('--milvus_user', default='', help='Milvus用户名')
    parser.add_argument('--milvus_password', default='', help='Milvus密码')
    
    # 选择要测试的数据库
    parser.add_argument('--test_neo4j', action='store_true', help='测试Neo4j连接')
    parser.add_argument('--test_milvus', action='store_true', help='测试Milvus连接')
    parser.add_argument('--test_all', action='store_true', help='测试所有数据库连接')
    
    args = parser.parse_args()
    
    # 默认测试所有数据库
    if not (args.test_neo4j or args.test_milvus):
        args.test_all = True
    
    logging.info("开始数据库连接测试...")
    
    # 测试Neo4j连接
    if args.test_neo4j or args.test_all:
        neo4j_success = test_neo4j_connection(
            uri=args.neo4j_uri,
            user=args.neo4j_user,
            password=args.neo4j_password,
            database=args.neo4j_database
        )
        print(f"\nNeo4j连接测试: {'✅ 成功' if neo4j_success else '❌ 失败'}")
    
    # 测试Milvus连接
    if args.test_milvus or args.test_all:
        milvus_success = test_milvus_connection(
            host=args.milvus_host,
            port=args.milvus_port,
            user=args.milvus_user,
            password=args.milvus_password
        )
        print(f"\nMilvus连接测试: {'✅ 成功' if milvus_success else '❌ 失败'}")
    
    logging.info("数据库连接测试完成")

if __name__ == "__main__":
    main()