import sys
import os
import argparse
import traceback
from typing import Sequence
import uuid 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from academic_group_meeting_backend import AcademicMeetingSystem
from milvus_enabled_academic_meeting_system import MilvusEnabledAcademicMeetingSystem
from neo4j import GraphDatabase
import matplotlib.pyplot as plt

# 定义输出目录常量
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
PICTURE_DIR = os.path.join(OUTPUT_DIR, "picture")
JSON_DIR = os.path.join(OUTPUT_DIR, "json")
ARTICLE_DIR = os.path.join(OUTPUT_DIR, "article")
ATTACHMENT_DIR = os.path.join(BASE_DIR, "attachment")

# 确保输出目录存在
os.makedirs(PICTURE_DIR, exist_ok=True)
os.makedirs(JSON_DIR, exist_ok=True)
os.makedirs(ARTICLE_DIR, exist_ok=True)
os.makedirs(ATTACHMENT_DIR, exist_ok=True)

try:
    plt.rcParams['font.family'] = ['WenQuanYi Micro Hei']
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
except:
    print("警告：可能无法正确显示中文，请安装相应字体")

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

def create_academic_agents(meeting: AcademicMeetingSystem):
    """
    创建标准的学术角色
    
    Args:
        meeting: 学术组会系统实例
    """
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

def run_academic_meeting(topic=None, use_remote=True, deep_search=False, rounds=3):
    """
     运行基本学术组会模拟系统
    
    Args:
        topic: 自定义会议主题，None则使用默认主题
        use_remote: 是否使用远程LLM
        deep_search: 是否使用深度网页内容搜索
        rounds: 讨论轮数
    """
    try:
        # 创建学术组会系统
        meeting = AcademicMeetingSystem(use_remote_llm=use_remote)
        
        # 创建学术角色
        create_academic_agents(meeting)
        
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
        output_filename = os.path.join(PICTURE_DIR, f"academic_meeting_{scene_id}.png")
        meeting.visualize_conversation_graph(scene_id, output_filename=output_filename)
        
        print(f"\n组会讨论已结束并保存为图表: {output_filename}")
        return scene_id
    except Exception as e:
        print(f"运行学术组会时出错: {str(e)}")
        traceback.print_exc()
        return None

def run_academic_meeting_database(topic=None, use_remote=True, deep_search=False, use_milvus=True, 
                         milvus_host="localhost", milvus_port="19530", 
                         use_neo4j=True, neo4j_database="academicgraph",
                         neo4j_uri="bolt://localhost:7687", 
                         neo4j_user="neo4j", neo4j_password="20031117",
                         rounds=3, clear_neo4j=True, fontpath=None,
                         show_namespace=True, generate_review=False, review_subtopics=None, 
                         export_detail_props=False, attachments=None,
                         local_model="deepseek-r1:1.5b"):
    # fontpath="/home/zyh/anaconda3/envs/SDS/lib/python3.10/site-packages/matplotlib/mpl-data/fonts/ttf/SourceHanSans.ttc"
    """
    运行学术组会模拟系统，支持多种数据库和综述模式
    
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
        generate_review: 是否生成综述报告而非常规会议
        review_subtopics: 综述报告的子话题列表，仅在generate_review=True时有效
        export_detail_props: 是否导出详细的节点属性到Neo4j
        attachments: 附件文件路径列表
        
    Returns:
        str: 场景ID或综述报告ID
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
                local_model=local_model,  # 添加模型参数
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
            meeting = AcademicMeetingSystem(
                use_remote_llm=use_remote,
                local_model=local_model,  # 添加模型参数
            )

        # 创建学术角色
        create_academic_agents(meeting)
        
        # 如果提供了附件，处理附件
        if attachments and isinstance(attachments, list) and len(attachments) > 0:
            print(f"发现 {len(attachments)} 个附件，正在处理...")
            for attachment_path in attachments:
                if os.path.exists(attachment_path):
                    filename = os.path.basename(attachment_path)
                    print(f"添加附件: {filename}")
                    attachment_id = uuid.uuid4()
                    attachment_info = {
                        "filename": filename,
                        "path": attachment_path
                    }
                    meeting.add_attachment(scene_id=None,attachment_id=attachment_id,attachment_info=attachment_info)
        
        # 设置导出详细属性标志
        # if hasattr(meeting, 'set_export_detail_props') and export_detail_props:
        #     meeting.set_export_detail_props(export_detail_props)
        
        # 如果是生成综述报告模式
        if generate_review:
            print(f"\n=== 开始生成综述报告：{topic or '面向智能体的记忆管理系统'} ===\n")
            # 打印初始轮次信息
            print(f"\n--- 准备开始讨论，总共 {rounds} 轮 ---\n")
            
            # 如果未指定主题，使用默认的记忆管理系统主题
            if not topic:
                topic = "面向智能体的记忆管理系统"
                print(f"使用默认综述主题：{topic}")
            
            # 使用通用综述方法
            review_result = meeting.generate_comprehensive_review(
                topic=topic,
                professor_id="prof_xu",
                subtopics=review_subtopics,
                # rounds=max(rounds, 3),  # 综述至少需要3轮讨论
                rounds=rounds,
                include_literature=True,
                include_open_source=True,
                output_dir=ARTICLE_DIR  # 传递输出目录给报告生成方法
            )

            print(f"\n综述报告已生成: {review_result.get('filename', '未知文件')}")
            
            
            # 如果启用了Neo4j，导出语义图到Neo4j
            if use_neo4j:
                print("\n=== 导出综述报告语义图到Neo4j ===\n")
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
                    traceback.print_exc()
            
            # 关闭所有连接
            if use_milvus:
                meeting.close_milvus()
                
            return review_result.get("report_id", "review_report")
            
        else:
            # 常规学术组会模式
            # 确定讨论主题
            meeting_topic = topic if topic else "大型语言模型在多智能体协作中的应用"
            
            # 创建组会场景
            scene_id = meeting.create_scene(
                name="人工智能实验室组会",
                description=f"讨论主题：{meeting_topic}"
            )
            
            # 打印初始轮次信息
            print("\n--- 准备开始讨论，总共 {} 轮 ---\n".format(rounds))
            
            # 如果是自定义主题，使用自定义会议模式
            if topic:
                print(f"\n=== 开始自定义学术组会：{meeting_topic} ===\n")
                # 确保在每轮讨论开始时调用increment_discussion_round方法
                if hasattr(meeting, '_increment_discussion_round'):
                    # 初始轮次
                    meeting._increment_discussion_round()
                    meeting.run_custom_meeting(
                        topic=meeting_topic,
                        professor_id="prof_xu",
                        rounds=rounds,
                        deep_search=deep_search
                    )
            else:
                # 使用标准会议模式
                print(f"\n=== 开始标准学术组会：{meeting_topic} ===\n")
                # 确保在每轮讨论开始时调用increment_discussion_round方法
                if hasattr(meeting, '_increment_discussion_round'):
                    # 初始轮次
                    meeting._increment_discussion_round()
                meeting.start_academic_meeting(
                    scene_id=scene_id,
                    topic=meeting_topic,
                    moderator_id="prof_xu",
                    rounds=rounds,
                    deep_search=deep_search
                )
            
            # 生成组会可视化图，使用设置的字体
            output_filename = os.path.join(PICTURE_DIR, f"academic_meeting_{scene_id}.png")
            meeting.visualize_conversation_graph(scene_id, fontpath=fontpath, output_filename=output_filename)
            
            # 如果启用了命名空间显示，生成命名空间图
            if show_namespace:
                try:
                    print("\n=== 生成命名空间图 ===\n")
                    namespace_output = os.path.join(PICTURE_DIR, "academic_meeting_namespaces.png")
                    meeting.backend.semantic_graph.visualize_namespaces(
                        title=f"学术组会命名空间关系图 - {meeting_topic}",
                        fontpath=fontpath,
                        output_filename=namespace_output
                    )
                except Exception as e:
                    print(f"生成命名空间图时出错: {str(e)}")
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
                    traceback.print_exc()
            
            print(f"\n组会讨论已结束并保存为图表: {output_filename}")
            
            # 关闭所有连接
            if use_milvus:
                meeting.close_milvus()
                
            return scene_id
    
    except Exception as e:
        print(f"运行学术组会时出错: {str(e)}")
        traceback.print_exc()
        return None

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
    
    #!!!
    #!!!round轮次控制有问题，bug点，需要进一步审查
    #!!!
    parser.add_argument('--rounds', '-r', type=int, default=3,
                       help='讨论轮数，控制会议讨论的轮次数量')
    
    # 添加Neo4j相关参数
    parser.add_argument('--use-neo4j', action='store_true',
                       help='启用Neo4j图数据库存储图结构')
    parser.add_argument('--neo4j-uri', type=str, default='bolt://localhost:7687',
                       help='Neo4j服务器URI')
    parser.add_argument('--neo4j-user', type=str, default='neo4j',
                       help='Neo4j用户名')
    parser.add_argument('--neo4j-database', type=str, default='academicgraph',
                       help='Neo4j数据库名称')
    parser.add_argument('--neo4j-password', type=str, default='20031117',
                       help='Neo4j密码')
    # 添加是否清空Neo4j数据库的选项
    parser.add_argument('--no-clear-neo4j', action='store_true',
                       help='不清空Neo4j数据库中的内容')
    
    # 添加综述报告相关参数
    parser.add_argument('--generate-review', action='store_true',
                       help='生成综述报告而非常规会议')
    # parser.add_argument('--memory-review', action='store_true',
    #                    help='生成面向智能体的记忆管理系统综述报告（快捷方式）')
    
    # 添加详细属性导出选项
    parser.add_argument('--export-detail-props', action='store_true',
                       help='导出详细节点属性到Neo4j')
                        
    # 添加附件支持
    parser.add_argument('--attachments', type=str, nargs='+', default=None,
                       help='要包含的附件文件路径列表')
    
    # 在现有参数下方添加
    parser.add_argument('--local-model', type=str, default="deepseek-r1:1.5b",
                    help='指定本地模型名称，例如 "deepseek-r1:14b"，仅在使用 --local 参数时有效')
    
    args:Sequence[str] = parser.parse_args()
    

    # 运行学术组会或普通综述报告
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
        clear_neo4j=not args.no_clear_neo4j,
        generate_review=args.generate_review,
        export_detail_props=args.export_detail_props,
        attachments=args.attachments,
        local_model=args.local_model  # 添加这一行传递模型选项
    )

    # run_academic_meeting_database(
    #     topic=args.topic,
    #     use_remote=not args.local,
    #     deep_search=args.deep_search,
    #     use_milvus=args.use_milvus,
    #     milvus_host=args.milvus_host,
    #     milvus_port=args.milvus_port,
    #     use_neo4j=args.use_neo4j,
    #     neo4j_uri=args.neo4j_uri,
    #     neo4j_user=args.neo4j_user,
    #     neo4j_password=args.neo4j_password,
    #     neo4j_database=args.neo4j_database,
    #     rounds=args.rounds,
    #     clear_neo4j=not args.no_clear_neo4j,
    #     generate_review=args.generate_review,
    #     export_detail_props=args.export_detail_props,
    #     attachments=args.attachments
    # )
