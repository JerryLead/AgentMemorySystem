import sys
import os
import argparse
import traceback
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from academic_group_meeting_backend import AcademicMeetingSystem, MilvusEnabledAcademicMeetingSystem
from neo4j import GraphDatabase

def clear_neo4j_database(uri, username, password, database="academicgraph"):
    """
    清空Neo4j数据库中的所有节点和关系
    
    Args:
        uri: Neo4j服务器URI
        username: Neo4j用户名
        password: Neo4j密码
        database: Neo4j数据库名称
    """
    try:
        driver = GraphDatabase.driver(uri, auth=(username, password))
        with driver.session(database=database) as session:
            # 执行清空操作
            session.run("MATCH (n) DETACH DELETE n")
            print(f"已清空Neo4j数据库: {database}")
        driver.close()
        return True
    except Exception as e:
        print(f"清空Neo4j数据库失败: {str(e)}")
        return False

def run_academic_meeting(topic=None, use_remote=True, deep_search=False, rounds=3):
    """
    运行学术组会模拟系统
    
    Args:
        topic: 自定义会议主题，None则使用默认主题
        use_remote: 是否使用远程LLM
        deep_search: 是否使用深度网页内容搜索
        rounds: 讨论轮数
    """
    # 创建学术组会系统
    meeting = AcademicMeetingSystem(use_remote_llm=use_remote)
    
    # 创建学术角色
    meeting.create_academic_agent(
        agent_id="prof_xu",
        nickname="许教授",
        age=45,
        role_type="教授",
        specialty=["人工智能", "大语言模型", "多智能体系统"],
        personality="思维严谨，见解深刻，注重理论基础，喜欢引导学生思考问题本质"
    )
    
    meeting.create_academic_agent(
        agent_id="phd_li",
        nickname="李同学",
        age=28,
        role_type="博士生",
        specialty=["多智能体协作", "强化学习", "自然语言处理"],
        personality="研究能力强，善于分析问题，对新技术敏感，表达准确专业"
    )
    
    meeting.create_academic_agent(
        agent_id="msc_guo",
        nickname="郭同学",
        age=24,
        role_type="硕士生",
        specialty=["自然语言处理", "知识图谱"],
        personality="勤奋好学，思路开阔，擅长提问，代码能力强，渴望获取新知识"
    )
    
    meeting.create_academic_agent(
        agent_id="msc_wu",
        nickname="吴同学",
        age=25,
        role_type="硕士生",
        specialty=["计算机视觉", "多模态学习"],
        personality="实践能力强，关注应用场景，有创新思维，善于提出新观点"
    )
    
    # 确定讨论主题
    meeting_topic = topic if topic else "大型语言模型在多智能体协作中的应用"
    
    # 创建组会场景
    scene_id = meeting.create_scene(
        name="人工智能实验室组会",
        description=f"讨论主题：{meeting_topic}"
    )
    
    # 如果是自定义主题，使用自定义会议模式
    if topic:
        print(f"\n=== 开始自定义学术组会：{meeting_topic} ===\n")
        meeting.run_custom_meeting(
            topic=meeting_topic,
            professor_id="prof_xu",
            rounds=rounds,
            deep_search=deep_search
        )
    else:
        # 使用标准会议模式
        print(f"\n=== 开始标准学术组会：{meeting_topic} ===\n")
        meeting.start_academic_meeting(
            scene_id=scene_id,
            topic=meeting_topic,
            moderator_id="prof_xu",
            rounds=rounds,
            deep_search=deep_search
        )
    
    # 生成组会可视化图
    meeting.visualize_conversation_graph(scene_id)
    
    print("\n组会讨论已结束并保存为图表")
    return scene_id

# def run_academic_meeting_milvus(topic=None, use_remote=True, deep_search=False, use_milvus=False, 
#                          milvus_host="localhost", milvus_port="19530", rounds=3):
#     """
#     运行学术组会模拟系统
    
#     Args:
#         topic: 自定义会议主题，None则使用默认主题
#         use_remote: 是否使用远程LLM
#         deep_search: 是否使用深度网页内容搜索
#         use_milvus: 是否启用Milvus存储对话信息
#         milvus_host: Milvus服务器地址
#         milvus_port: Milvus服务器端口
#         rounds: 讨论轮数，控制会议讨论的轮次数量
#     """
#     try:
#         # 根据参数决定使用哪个系统类
#         if use_milvus:
#             # 使用支持Milvus的系统
#             print("启用Milvus对话存储...")
#             meeting = MilvusEnabledAcademicMeetingSystem(
#                 use_remote_llm=use_remote,
#                 milvus_host=milvus_host,
#                 milvus_port=milvus_port,
#                 milvus_collection="academic_dialogues"
#             )
#         else:
#             # 使用标准系统
#             meeting = AcademicMeetingSystem(use_remote_llm=use_remote)

def run_academic_meeting_database(topic=None, use_remote=True, deep_search=False, use_milvus=False, 
                         milvus_host="localhost", milvus_port="19530", 
                         use_neo4j=False, neo4j_database="academicgraph",
                         neo4j_uri="bolt://localhost:7687", 
                         neo4j_user="neo4j", neo4j_password="20031117",
                         rounds=3, clear_neo4j=True, fontpath="/home/zyh/anaconda3/envs/SDS/lib/python3.10/site-packages/matplotlib/mpl-data/fonts/ttf/SourceHanSans.ttc",
                         show_namespace=True):
    """
    运行学术组会模拟系统
    
    Args:
        topic: 自定义会议主题，None则使用默认主题
        use_remote: 是否使用远程LLM
        deep_search: 是否使用深度网页内容搜索
        use_milvus: 是否启用Milvus存储对话信息
        milvus_host: Milvus服务器地址
        milvus_port: Milvus服务器端口
        use_neo4j: 是否导出到Neo4j图数据库
        neo4j_uri: Neo4j服务器地址
        neo4j_user: Neo4j用户名
        neo4j_password: Neo4j密码
        neo4j_database: Neo4j数据库名称
        rounds: 讨论轮数，控制会议讨论的轮次数量
        clear_neo4j: 是否在运行前清空Neo4j数据库
        fontpath: 字体路径，用于解决中文显示问题
        show_namespace: 是否显示命名空间图
    """
    try:
        # 如果启用Neo4j并设置了清空，先清空数据库
        if use_neo4j and clear_neo4j:
            clear_neo4j_database(neo4j_uri, neo4j_user, neo4j_password, neo4j_database)

        # 根据参数决定使用哪个系统类
        if use_milvus:
            # 使用支持Milvus的系统
            print("启用Milvus对话存储...")
            meeting = MilvusEnabledAcademicMeetingSystem(
                use_remote_llm=use_remote,
                use_local_embeddings=True,
                milvus_host=milvus_host,
                milvus_port=milvus_port,
                milvus_collection="academic_dialogues",
                neo4j_database=neo4j_database,
                neo4j_uri=neo4j_uri,
                neo4j_user=neo4j_user,
                neo4j_password=neo4j_password
            )
        else:
            # 使用标准系统
            meeting = AcademicMeetingSystem(use_remote_llm=use_remote)

        # 创建学术角色
        meeting.create_academic_agent(
            agent_id="prof_xu",
            nickname="许教授",
            age=45,
            role_type="教授",
            specialty=["人工智能", "大语言模型", "多智能体系统"],
            personality="思维严谨，见解深刻，注重理论基础，喜欢引导学生思考问题本质"
        )
        
        meeting.create_academic_agent(
            agent_id="phd_li",
            nickname="李同学",
            age=28,
            role_type="博士生",
            specialty=["多智能体协作", "强化学习", "自然语言处理"],
            personality="研究能力强，善于分析问题，对新技术敏感，表达准确专业"
        )
        
        meeting.create_academic_agent(
            agent_id="msc_guo",
            nickname="郭同学",
            age=24,
            role_type="硕士生",
            specialty=["自然语言处理", "知识图谱"],
            personality="勤奋好学，思路开阔，擅长提问，代码能力强，渴望获取新知识"
        )
        
        meeting.create_academic_agent(
            agent_id="msc_wu",
            nickname="吴同学",
            age=25,
            role_type="硕士生",
            specialty=["计算机视觉", "多模态学习"],
            personality="实践能力强，关注应用场景，有创新思维，善于提出新观点"
        )
        
        # 确定讨论主题
        meeting_topic = topic if topic else "大型语言模型在多智能体协作中的应用"
        
        # 创建组会场景
        scene_id = meeting.create_scene(
            name="人工智能实验室组会",
            description=f"讨论主题：{meeting_topic}"
        )
        
        # 如果是自定义主题，使用自定义会议模式
        if topic:
            print(f"\n=== 开始自定义学术组会：{meeting_topic} ===\n")
            meeting.run_custom_meeting(
                topic=meeting_topic,
                professor_id="prof_xu",
                rounds=rounds,
                deep_search=deep_search
            )
        else:
            # 使用标准会议模式
            print(f"\n=== 开始标准学术组会：{meeting_topic} ===\n")
            meeting.start_academic_meeting(
                scene_id=scene_id,
                topic=meeting_topic,
                moderator_id="prof_xu",
                rounds=rounds,
                deep_search=deep_search
            )
        
        # 生成组会可视化图，使用设置的字体
        meeting.visualize_conversation_graph(scene_id, fontpath=fontpath)
        
        # 如果启用了命名空间显示，生成命名空间图
        if show_namespace:
            try:
                print("\n=== 生成命名空间图 ===\n")
                meeting.backend.semantic_graph.visualize_namespaces(
                    title=f"学术组会命名空间关系图 - {meeting_topic}",
                    fontpath=fontpath,
                    output_filename="academic_meeting_namespaces.png"
                )
            except Exception as e:
                print(f"生成命名空间图时出错: {str(e)}")
                import traceback
                traceback.print_exc()
        
        # 如果启用了Neo4j，导出语义图到Neo4j
        if use_neo4j:
            print("\n=== 导出语义图到Neo4j ===\n")
            try:
                stats = meeting.export_to_neo4j(
                    neo4j_uri=neo4j_uri,
                    neo4j_user=neo4j_user,
                    neo4j_password=neo4j_password,
                    neo4j_database=neo4j_database
                )
                print(f"Neo4j导出完成，共导出 {stats.get('nodes_total', 0)} 个节点和 {stats.get('relations_total', 0)} 个关系")
            except Exception as e:
                print(f"导出到Neo4j时出错: {str(e)}")
                import traceback
                traceback.print_exc()
        
        print("\n组会讨论已结束并保存为图表")
        
        # 关闭所有连接
        if use_milvus:
            meeting.close_milvus()
            
        return scene_id
    
    except Exception as e:
        print(f"运行学术组会时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
    
# def run_academic_meeting_database(topic=None, use_remote=True, deep_search=False, use_milvus=False, 
#                          milvus_host="localhost", milvus_port="19530", 
#                          use_neo4j=False, neo4j_database="academicgraph",
#                          neo4j_uri="bolt://localhost:7687", 
#                          neo4j_user="neo4j", neo4j_password="20031117",
#                          rounds=3, clear_neo4j=True):
#     """
#     运行学术组会模拟系统
    
#     Args:
#         topic: 自定义会议主题，None则使用默认主题
#         use_remote: 是否使用远程LLM
#         deep_search: 是否使用深度网页内容搜索
#         use_milvus: 是否启用Milvus存储对话信息
#         milvus_host: Milvus服务器地址
#         milvus_port: Milvus服务器端口
#         use_neo4j: 是否导出到Neo4j图数据库
#         neo4j_uri: Neo4j服务器地址
#         neo4j_user: Neo4j用户名
#         neo4j_password: Neo4j密码
#         rounds: 讨论轮数，控制会议讨论的轮次数量
#         clear_neo4j: 是否在运行前清空Neo4j数据库
#     """
#     try:
#         # 如果启用Neo4j并设置了清空，先清空数据库
#         if use_neo4j and clear_neo4j:
#             clear_neo4j_database(neo4j_uri, neo4j_user, neo4j_password)

#         # 根据参数决定使用哪个系统类
#         if use_milvus:
#             # 使用支持Milvus的系统
#             print("启用Milvus对话存储...")
#             meeting = MilvusEnabledAcademicMeetingSystem(
#                 use_remote_llm=use_remote,
#                 use_local_embeddings=True,
#                 milvus_host=milvus_host,
#                 milvus_port=milvus_port,
#                 milvus_collection="academic_dialogues",
#                 neo4j_database=neo4j_database,
#                 neo4j_uri=neo4j_uri,
#                 neo4j_user=neo4j_user,
#                 neo4j_password=neo4j_password
#             )
#         else:
#             # 使用标准系统
#             meeting = AcademicMeetingSystem(use_remote_llm=use_remote)

#         # 创建学术角色
#         meeting.create_academic_agent(
#             agent_id="prof_xu",
#             nickname="许教授",
#             age=45,
#             role_type="教授",
#             specialty=["人工智能", "大语言模型", "多智能体系统"],
#             personality="思维严谨，见解深刻，注重理论基础，喜欢引导学生思考问题本质"
#         )
        
#         meeting.create_academic_agent(
#             agent_id="phd_li",
#             nickname="李同学",
#             age=28,
#             role_type="博士生",
#             specialty=["多智能体协作", "强化学习", "自然语言处理"],
#             personality="研究能力强，善于分析问题，对新技术敏感，表达准确专业"
#         )
        
#         meeting.create_academic_agent(
#             agent_id="msc_guo",
#             nickname="郭同学",
#             age=24,
#             role_type="硕士生",
#             specialty=["自然语言处理", "知识图谱"],
#             personality="勤奋好学，思路开阔，擅长提问，代码能力强，渴望获取新知识"
#         )
        
#         meeting.create_academic_agent(
#             agent_id="msc_wu",
#             nickname="吴同学",
#             age=25,
#             role_type="硕士生",
#             specialty=["计算机视觉", "多模态学习"],
#             personality="实践能力强，关注应用场景，有创新思维，善于提出新观点"
#         )
        
#         # 确定讨论主题
#         meeting_topic = topic if topic else "大型语言模型在多智能体协作中的应用"
        
#         # 创建组会场景
#         scene_id = meeting.create_scene(
#             name="人工智能实验室组会",
#             description=f"讨论主题：{meeting_topic}"
#         )
        
#         # 如果是自定义主题，使用自定义会议模式
#         if topic:
#             print(f"\n=== 开始自定义学术组会：{meeting_topic} ===\n")
#             meeting.run_custom_meeting(
#                 topic=meeting_topic,
#                 professor_id="prof_xu",
#                 rounds=rounds,
#                 deep_search=deep_search
#             )
#         else:
#             # 使用标准会议模式
#             print(f"\n=== 开始标准学术组会：{meeting_topic} ===\n")
#             meeting.start_academic_meeting(
#                 scene_id=scene_id,
#                 topic=meeting_topic,
#                 moderator_id="prof_xu",
#                 rounds=rounds,
#                 deep_search=deep_search
#             )
        
#         # 生成组会可视化图
#         meeting.visualize_conversation_graph(scene_id)
        
#         # 如果启用了Neo4j，导出语义图到Neo4j
#         if use_neo4j:
#             print("\n=== 导出语义图到Neo4j ===\n")
#             try:
#                 stats = meeting.export_to_neo4j()
#                     # neo4j_uri=neo4j_uri,
#                     # neo4j_user=neo4j_user,
#                     # neo4j_password=neo4j_password
#                 print(f"Neo4j导出完成，共导出 {stats.get('nodes_total', 0)} 个节点和 {stats.get('relations_total', 0)} 个关系")
#             except Exception as e:
#                 print(f"导出到Neo4j时出错: {str(e)}")
#                 traceback.print_exc()
        
#         print("\n组会讨论已结束并保存为图表")
        
#         # 关闭所有连接
#         if use_milvus:
#             meeting.close_milvus()
            
#         return scene_id
    
#     except Exception as e:
#         print(f"运行学术组会时出错: {str(e)}")
#         traceback.print_exc()
#         return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='学术组会模拟系统')
    parser.add_argument('--topic', type=str, help='自定义会议主题', default=None)
    parser.add_argument('--local', action='store_true', help='使用本地LLM模型')
    parser.add_argument('--deep-search', action='store_true', 
                       help='使用深度网页内容搜索(提取网页的全部内容而非摘要)')
    parser.add_argument('--use-milvus', action='store_true', 
                       help='启用Milvus向量数据库存储对话信息')
    parser.add_argument('--milvus-host', type=str, default='localhost', 
                       help='Milvus服务器地址')
    parser.add_argument('--milvus-port', type=str, default='19530', 
                       help='Milvus服务器端口')
    parser.add_argument('--rounds', '-r', type=int, default=3,
                       help='讨论轮数，控制会议讨论的轮次数量')
    
    # 添加Neo4j相关参数
    parser.add_argument('--use-neo4j', action='store_true',
                       help='启用Neo4j图数据库存储图结构')
    parser.add_argument('--neo4j-uri', type=str, default='bolt://localhost:7687',
                       help='Neo4j服务器URI')
    parser.add_argument('--neo4j-user', type=str, default='neo4j',
                       help='Neo4j用户名')
    parser.add_argument('--neo4j-database', type=str, default='academicgraph',)
    parser.add_argument('--neo4j-password', type=str, default='20031117',
                       help='Neo4j密码')
    # 添加是否清空Neo4j数据库的选项
    parser.add_argument('--no-clear-neo4j', action='store_true',
                       help='不清空Neo4j数据库中的内容')
    
    args = parser.parse_args()
    
    # 运行学术组会
    run_academic_meeting_database(
        topic=args.topic,
        use_remote=not args.local,
        deep_search=args.deep_search,
        use_milvus=args.use_milvus,
        milvus_host=args.milvus_host,
        milvus_port=args.milvus_port,
        use_neo4j=args.use_neo4j,
        neo4j_uri=args.neo4j_uri,
        neo4j_user=args.neo4j_user,
        neo4j_password=args.neo4j_password,
        neo4j_database=args.neo4j_database,
        rounds=args.rounds,
        clear_neo4j=not args.no_clear_neo4j
    )
