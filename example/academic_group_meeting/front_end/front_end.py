"""
version: 3.0
author: 张宇涵
feature: 1. 优化了对话消息的处理逻辑，避免重复处理相同内容
2.实现了附件上传和管理功能，支持上传多种文件类型，并提供预览功能
"""

# 完整实现 front_end_forth.py
import json
import logging
import streamlit as st
import sys
import os
import time
import re
from datetime import datetime
import base64
import pandas as pd
from io import StringIO

# 配置日志
logging.basicConfig(
    level=logging.WARNING,  # 设置为WARNING级别以上才会输出
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("academic_meeting.log"), logging.StreamHandler()]
)

# 避免torch._classes导致的streamlit错误
for module_name in list(sys.modules.keys()):
    if (module_name.startswith('torch._classes')):
        del sys.modules[module_name]

# 修改导入语句，使用相对导入
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 直接导入接口函数
from academic_group_meeting_interface import run_academic_meeting_database

# CSS样式保持不变
# # CSS样式
st.markdown("""
<style>
    /* 全局样式 */
    .main {
        background-color: #f9f9f9;
    }
    .header-text {
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 20px;
    }
    .stButton>button {
        width: 100%;
    }
    .small-text {
        font-size: 12px;
    }
    .config-section {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    
    /* 命名空间标签 */
    .namespace-tag {
        display: inline-block;
        margin-right: 10px;
        padding: 3px 8px;
        border-radius: 15px;
        font-size: 12px;
        font-weight: bold;
    }
    .namespace-user {
        background-color: #90CAF9;
        color: black;
    }
    .namespace-dialogue {
        background-color: #A5D6A7;
        color: black;
    }
    .namespace-attachment {
        background-color: #FFCC80;
        color: black;
    }
    .namespace-task {
        background-color: #FFF59D;
        color: black;
    }
    
    /* 对话窗口 */
    .chat-container {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        height: 500px;
        overflow-y: auto;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        display: flex;
        flex-direction: column;
        margin-bottom: 20px;
    }
    
    /* 聊天消息 */
    .chat-message {
        display: flex;
        margin-bottom: 15px;
    }
    .chat-message.right {
        flex-direction: row-reverse;
    }
    .message-avatar {
        width: 42px;
        height: 42px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 20px;
        flex-shrink: 0;
        margin: 0 10px;
    }
    .message-content {
        padding: 10px 15px;
        border-radius: 18px;
        max-width: 80%;
        position: relative;
    }
    .message-name {
        font-size: 12px;
        font-weight: bold;
        margin-bottom: 4px;
    }
    
    /* 特定角色的消息样式 */
    .system-avatar {
        background-color: #E0E0E0;
        color: #333;
    }
    .system-message {
        background-color: #F5F5F5;
        border-left: 3px solid #9E9E9E;
        color: #333;
    }
    
    .professor-avatar {
        background-color: #BBDEFB;
        color: #0D47A1;
    }
    .professor-message {
        background-color: #E3F2FD;
        border-left: 3px solid #2196F3;
        color: #0D47A1;
    }
    
    .phd-avatar {
        background-color: #C8E6C9;
        color: #1B5E20;
    }
    .phd-message {
        background-color: #E8F5E9;
        border-left: 3px solid #4CAF50;
        color: #1B5E20;
    }
    
    .msc-guo-avatar {
        background-color: #FFECB3;
        color: #E65100;
    }
    .msc-guo-message {
        background-color: #FFF8E1;
        border-left: 3px solid #FFC107;
        color: #E65100;
    }
    
    .msc-wu-avatar {
        background-color: #E1BEE7;
        color: #4A148C;
    }
    .msc-wu-message {
        background-color: #F3E5F5;
        border-left: 3px solid #9C27B0;
        color: #4A148C;
    }
    
    /* 附件样式 */
    .file-list {
        background-color: #f8f9fa;
        padding: 10px;
        border-radius: 8px;
        margin-top: 10px;
        margin-bottom: 15px;
        border: 1px solid #e9ecef;
    }
    .file-item {
        padding: 8px;
        border-bottom: 1px solid #e9ecef;
        display: flex;
        align-items: center;
    }
    .file-item:last-child {
        border-bottom: none;
    }
    .file-icon {
        margin-right: 10px;
        font-size: 18px;
    }
    .file-name {
        flex-grow: 1;
    }
    .file-info {
        font-size: 12px;
        color: #6c757d;
    }
    
    /* 按钮样式 */
    .custom-button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 8px 16px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 14px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 4px;
    }
    .custom-button.delete {
        background-color: #f44336;
    }
    
    /* 进度条样式 */
    .stProgress > div > div > div > div {
        background-color: #4CAF50;
    }
    
    /* 轮次信息样式 */
    .round-info {
        background-color: #4a6baf;
        color: white;
        padding: 8px 15px;
        border-radius: 5px;
        margin: 10px 0;
        font-weight: bold;
        text-align: center;
    }
    
    /* 系统消息样式增强 */
    .system-message strong {
        color: #1565c0;
    }
</style>
""", unsafe_allow_html=True)

# 确保附件目录存在
ATTACHMENT_DIR = "./semantic_map_case/academic_group_meeting/attachment"
os.makedirs(ATTACHMENT_DIR, exist_ok=True)

def get_attachment_dir():
    """获取附件目录"""
    if not os.path.exists(ATTACHMENT_DIR):
        os.makedirs(ATTACHMENT_DIR, exist_ok=True)
    return ATTACHMENT_DIR

def list_attachments():
    """列出所有附件"""
    attachment_dir = get_attachment_dir()
    if os.path.exists(attachment_dir):
        return [f for f in os.listdir(attachment_dir) if os.path.isfile(os.path.join(attachment_dir, f))]
    return []

def upload_attachment(uploaded_file):
    """保存上传的附件"""
    attachment_dir = get_attachment_dir()
    
    # 创建带时间戳的文件名防止重名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{uploaded_file.name}"
    filepath = os.path.join(attachment_dir, filename)
    
    # 保存文件
    with open(filepath, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return filepath

class ChatProcessor:
    def __init__(self):
        # 初始化消息列表
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # 添加消息渲染跟踪，避免重复渲染
        if "rendered_messages" not in st.session_state:
            st.session_state.rendered_messages = set()
            
        self.buffer = ""
        self.current_speaker = "system"
        # 添加已处理消息的跟踪集合
        self.processed_messages = set()
    
    def _process_system_message(self, message, message_id):
        """处理系统消息"""
        content = message['content']
        
        st.session_state.messages.append({
            "role": "system",
            "content": content,
            "time": message.get('timestamp', datetime.now().strftime("%H:%M:%S")),
            "id": message_id
        })

    def _process_round_info(self, message, message_id):
        """处理轮次信息"""
        content = message['content']
        
        st.session_state.messages.append({
            "role": "system",
            "content": content,
            "time": message.get('timestamp', datetime.now().strftime("%H:%M:%S")),
            "id": message_id,
            "is_round_info": True  # 标记为轮次信息，用于特殊显示
        })

    def _process_agent_message(self, message, message_id):
        """处理代理消息"""
        agent_name = message['agent_name']
        content = message['content']
        message_type = message.get('message_type', 'normal')
        
        # 确定角色类型
        role = "professor" if "教授" in agent_name else \
            "phd" if "李同学" in agent_name else \
            "msc_guo" if "郭同学" in agent_name else \
            "msc_wu" if "吴同学" in agent_name else "unknown"
        
        st.session_state.messages.append({
            "role": role,
            "content": content,
            "time": message.get('timestamp', datetime.now().strftime("%H:%M:%S")),
            "id": message_id,
            "message_type": message_type,
            "agent_name": agent_name
        })

    def _process_summary_message(self, message, message_id):
        """处理总结消息"""
        content = message['content']
        summary_type = message.get('summary_type', 'normal')
        
        # 确定角色类型
        role = "professor" if "教授" in message['agent_name'] else \
            "phd" if "李同学" in message['agent_name'] else \
            "msc_guo" if "郭同学" in message['agent_name'] else \
            "msc_wu" if "吴同学" in message['agent_name'] else "unknown"
        
        st.session_state.messages.append({
            "role": role,
            "content": content,
            "time": message.get('timestamp', datetime.now().strftime("%H:%M:%S")),
            "id": message_id,
            "summary_type": summary_type
        })

    def write(self, text):
        """处理输出文本并转换为聊天消息"""
        # 将新文本添加到缓冲区
        self.buffer += text
        
        # 检测不同角色的输出
        self.process_buffer()
        
        # 显示消息
        self.display_messages()
        
        return len(text)
    
    def process_buffer(self):
        """处理缓冲区文本，提取结构化消息"""
        # 在方法开头初始化变量，确保它们在所有代码路径上都已定义
        detected_role = None
        text = ""
        
        # 使用正则表达式匹配完整的消息格式 <MSG_START>.*?<MSG_END>
        msg_pattern = re.compile(r'<MSG_START>(.*?)<MSG_END>', re.DOTALL)
        matches = list(msg_pattern.finditer(self.buffer))
        
        if matches:
            logging.debug(f"找到 {len(matches)} 个完整消息")  # 使用logging代替print
            
            # 处理每个找到的消息
            for match in matches:
                # 获取完整的匹配和JSON部分
                full_match = match.group(0)  # 完整匹配，包括标记
                json_str = match.group(1)    # 仅JSON部分
                
                logging.debug(f"提取的JSON字符串: {json_str}")  # 使用logging代替print
                
                try:
                    # message = json.loads(json_str.replace("'", '"'))
                    message = json.loads(json_str)  # 替换单引号以确保JSON有效
                    logging.debug(f"解析后的消息: {message}")
                    message_id = f"{message['type']}:{message.get('timestamp', '')}"
                    
                    # 如果消息已处理则跳过
                    if message_id in self.processed_messages:
                        logging.debug(f"消息已处理，跳过: {message_id}")
                        continue
                    
                    # 根据消息类型处理
                    if message['type'] == 'system':
                        logging.debug("处理系统消息")  # 使用logging代替print
                        self._process_system_message(message, message_id)
                    elif message['type'] == 'round_info':
                        logging.debug("处理轮次信息")  # 使用logging代替print
                        self._process_round_info(message, message_id)
                    elif message['type'] == 'agent':
                        logging.debug("处理代理消息")  # 使用logging代替print
                        self._process_agent_message(message, message_id)
                    elif message['type'] == 'summary':
                        logging.debug("处理总结消息")  # 使用logging代替print
                        self._process_summary_message(message, message_id)
                    
                    # 标记为已处理
                    self.processed_messages.add(message_id)
                
                except json.JSONDecodeError as e:
                    logging.error(f"JSON解析错误: {str(e)}, JSON: '{json_str}'")  # 使用logging代替print
            
            # 移除所有已处理的消息
            self.buffer = msg_pattern.sub('', self.buffer)
        
        self.buffer = ""

    def display_messages(self):
        """显示消息，避免重复渲染"""
        for message in st.session_state.messages:
            message_id = message.get("id")
            
            # 如果消息已渲染则跳过
            if message_id in st.session_state.rendered_messages:
                continue
                
            # 标记为已渲染
            st.session_state.rendered_messages.add(message_id)
            
            role = message["role"]
            content = message["content"]
            
            # 特殊处理轮次信息
            if role == "system" and message.get("is_round_info", True):
                st.markdown(f"<div class='round-info'>{content}</div>", unsafe_allow_html=True)
                continue
            
            # 根据角色和消息类型显示
            if role == "system":
                with st.chat_message("system", avatar="🛠️"):
                    st.markdown(f"**系统**")
                    st.markdown(body=content,unsafe_allow_html=True)
            elif role == "professor":
                with st.chat_message("user", avatar="👨‍🏫"):
                    # 根据消息类型添加标记
                    message_type = message.get("message_type", "normal")
                    display_name = "**许教授**"
                    
                    if message_type == "subtopic_summary":
                        display_name = "**许教授 (子话题总结)**"
                    elif message_type == "round_summary":
                        display_name = f"**许教授 (第{message.get('round', '')}轮总结)**"
                    
                    st.markdown(display_name)
                    st.markdown(body=content,unsafe_allow_html=True)
            # 其他角色类似处理...
            elif role == "phd":
                with st.chat_message("assistant", avatar="👨‍🎓"):
                    st.markdown(f"**李同学 (博士)**")
                    st.markdown(body=content,unsafe_allow_html=True)
            elif role == "msc_guo":
                with st.chat_message("human", avatar="👩‍🎓"):
                    st.markdown(f"**郭同学 (硕士)**")
                    st.markdown(body=content,unsafe_allow_html=True)
            elif role == "msc_wu":
                with st.chat_message("ai", avatar="👨‍💻"):
                    st.markdown(f"**吴同学 (硕士)**")
                    st.markdown(body=content,unsafe_allow_html=True)

    def flush(self):
        """刷新缓冲区，确保所有内容都被处理"""
        # 处理缓冲区中的任何剩余内容
        self.process_buffer()
        # 清空缓冲区
        self.buffer = ""
        return


class StreamlitChatOutput:
    def __init__(self):
        self.processor = ChatProcessor()
        
    def write(self, text):
        """处理输出文本"""
        return self.processor.write(text)
            
    def flush(self):
        """刷新缓冲区"""
        self.processor.flush()


def format_file_size(size_bytes):
    """格式化文件大小为人类可读形式"""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes/1024:.1f} KB"
    else:
        return f"{size_bytes/(1024*1024):.1f} MB"

def display_file_preview(file_path):
    """显示文件预览"""
    file_ext = os.path.splitext(file_path)[1].lower()
    
    # 图片文件
    if file_ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
        # st.image(file_path, caption=os.path.basename(file_path), width=300)
        st.image(file_path, caption=os.path.basename(file_path), width=500,use_container_width=True)
    
    # PDF文件
    elif file_ext == '.pdf':
        st.markdown(f"[查看PDF文件]({file_path})")
    
    # 文本文件
    elif file_ext in ['.txt', '.md', '.py', '.java', '.cpp', '.html', '.css', '.js', '.json', '.csv']:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            if len(content) > 1000:
                content = content[:1000] + "...(内容已截断)"
            st.code(content, language='python' if file_ext == '.py' else None)
        except Exception as e:
            st.warning(f"无法预览文件内容: {str(e)}")
    
    # 表格数据
    elif file_ext in ['.csv', '.xlsx', '.xls']:
        try:
            if file_ext == '.csv':
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path)
            st.dataframe(df.head(10), use_container_width=True)
            st.caption(f"仅显示前10行，共{len(df)}行")
        except Exception as e:
            st.warning(f"无法预览表格数据: {str(e)}")
    else:
        st.info(f"不支持预览此类型的文件: {file_ext}")

def manage_attachments():
    """附件管理界面"""
    st.subheader("📎 附件管理")
    
    # 上传新附件
    uploaded_file = st.file_uploader("选择要上传的文件", type=['pdf', 'txt', 'png', 'jpg', 'jpeg', 'py', 'ipynb', 'csv', 'xlsx'])
    
    if uploaded_file is not None:
        # 上传按钮
        if st.button("上传文件"):
            filepath = upload_attachment(uploaded_file)
            st.success(f"文件 '{uploaded_file.name}' 已上传成功！")
            st.rerun()
    
    # 显示现有附件
    attachments = list_attachments()
    
    if not attachments:
        st.info("暂无上传的附件")
        return
    
    st.subheader("已上传的附件")
    
    # 创建一个表格显示所有附件
    file_data = []
    for filename in attachments:
        file_path = os.path.join(ATTACHMENT_DIR, filename)
        file_size = os.path.getsize(file_path)
        file_date = datetime.fromtimestamp(os.path.getmtime(file_path)).strftime("%Y-%m-%d %H:%M:%S")
        file_data.append({
            "文件名": filename,
            "大小": format_file_size(file_size),
            "修改时间": file_date
        })
    
    df = pd.DataFrame(file_data)
    st.dataframe(df, use_container_width=True)
    
    # 文件预览和操作
    selected_file = st.selectbox("选择文件进行预览或操作", [""] + attachments)
    
    if selected_file:
        file_path = os.path.join(ATTACHMENT_DIR, selected_file)
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            if st.button("预览文件"):
                with st.expander("文件预览", expanded=True):
                    display_file_preview(file_path)
        
        with col2:
            # 下载按钮
            with open(file_path, "rb") as file:
                st.download_button(
                    label="下载文件",
                    data=file,
                    file_name=selected_file,
                    use_container_width=True
                )
        
        with col3:
            # 删除按钮
            if st.button("删除文件"):
                try:
                    os.remove(file_path)
                    st.success(f"文件 '{selected_file}' 已删除")
                    st.rerun()
                except Exception as e:
                    st.error(f"删除文件时出错: {str(e)}")
    
    # 批量操作
    st.subheader("批量操作")
    if st.button("删除所有附件"):
        confirm = st.checkbox("确认删除所有附件？此操作不可撤销。")
        if confirm:
            try:
                for filename in attachments:
                    os.remove(os.path.join(ATTACHMENT_DIR, filename))
                st.success("已删除所有附件")
                st.rerun()
            except Exception as e:
                st.error(f"批量删除文件时出错: {str(e)}")

# 重定向stdout输出到Streamlit的函数
def redirected_stdout_to_streamlit():
    # 创建StringIO对象用于捕获stdout输出
    temp_stdout = StringIO()
    
    # 保存原始stdout
    original_stdout = sys.stdout
    
    # 重定向stdout到StringIO对象
    sys.stdout = temp_stdout
    
    yield
    
    # 将捕获的内容发送到Streamlit界面
    output = temp_stdout.getvalue()
    if output:
        st.text_area("终端输出", output, height=300)
    
    # 恢复原始stdout
    sys.stdout = original_stdout

def main():
    # 初始化session_state
    if 'running' not in st.session_state:
        st.session_state.running = False
    if 'completed' not in st.session_state:
        st.session_state.completed = False
    if 'show_advanced' not in st.session_state:
        st.session_state.show_advanced = False
    if 'local_model' not in st.session_state:
        st.session_state.local_model = "deepseek-r1:1.5b"
    
    st.title("🎓 学术组会模拟系统 3.0")
    
    # 顶部选项卡
    tab1, tab2 = st.tabs(["组会讨论", "附件管理"])
    
    with tab1:
        # 组会讨论标签页的内容
        # 侧边栏配置
        with st.sidebar:
            st.header("基本配置")
            
            # API密钥输入
            api_key = st.text_input(
                "DeepSeek API密钥 (sk-...)", 
                value="sk-fbab400c86184b0daf9bd59467d35772", 
                type="password", 
                help="输入DeepSeek API密钥以运行组会"
            )
            
            # 主题输入
            topic = st.text_input(
                "讨论主题", 
                value="面向智能体的记忆管理系统", 
                help="输入学术组会的讨论主题"
            )
            
            # 本地模型选项
            use_local = st.checkbox(
                "使用本地模型", 
                value=False, 
                help="勾选使用本地模型而非远程API"
            )
            
            # 添加生成综述报告选项
            generate_review = st.checkbox(
                "生成综述报告", 
                value=False, 
                help="生成综述报告而非进行正常学术讨论"
            )
            
            # 深度搜索选项
            use_deep_search = st.checkbox(
                "使用深度网页搜索", 
                value=False, 
                help="勾选使用深度网页内容搜索"
            )
            
            # 讨论轮数
            if generate_review:
                # 当选择生成综述报告时，限制最小轮次为3，但允许调整3-5轮
                rounds = st.slider(
                    "讨论轮数",
                    min_value=1,
                    max_value=5,
                    value=3,
                    help="综述报告模式下需要至少3轮讨论"
                )
            else:
                # 普通讨论模式下允许调整轮次
                rounds = st.slider(
                    "讨论轮数",
                    min_value=1,
                    max_value=5,
                    value=3,
                    help="设置学术组会的讨论轮数"
                )
                
            # 显示终端输出选项
            show_terminal_output = st.checkbox(
                "显示终端输出", 
                value=True, 
                help="在界面上显示终端输出，便于调试"
            )

            if use_local:
                st.session_state.local_model = st.selectbox(
                    "选择本地模型", 
                    ["deepseek-r1:1.5b", "deepseek-r1:14b"],
                    index=0,
                    help="选择要使用的本地模型，数字越大性能越好但资源占用也越多"
                )
            else:
                st.session_state.local_model = "deepseek-r1:1.5b"  # 使用远程API时此参数无效

            # 高级设置展开/折叠
            if st.button("显示/隐藏高级设置"):
                st.session_state.show_advanced = not st.session_state.show_advanced
            
            # 高级设置部分
            if st.session_state.show_advanced:
                st.markdown("---")
                st.subheader("数据库设置")
                
                # Milvus配置
                with st.expander("Milvus向量数据库配置"):
                    use_milvus = st.checkbox(
                        "启用Milvus存储", 
                        value=True,
                        help="使用Milvus向量数据库存储对话信息"
                    )
                    
                    milvus_host = st.text_input(
                        "Milvus主机地址",
                        value="localhost",
                        disabled=not use_milvus
                    )
                    
                    milvus_port = st.text_input(
                        "Milvus端口",
                        value="19530",
                        disabled=not use_milvus
                    )
                
                # Neo4j配置
                with st.expander("Neo4j图数据库配置"):
                    use_neo4j = st.checkbox(
                        "启用Neo4j存储", 
                        value=True,
                        help="将学术组会知识图谱导出到Neo4j数据库"
                    )
                    
                    neo4j_uri = st.text_input(
                        "Neo4j URI",
                        value="bolt://localhost:7687",
                        disabled=not use_neo4j
                    )
                    
                    neo4j_user = st.text_input(
                        "Neo4j用户名",
                        value="neo4j",
                        disabled=not use_neo4j
                    )
                    
                    neo4j_password = st.text_input(
                        "Neo4j密码",
                        value="20031117",
                        type="password",
                        disabled=not use_neo4j
                    )
                    
                    neo4j_database = st.text_input(
                        "Neo4j数据库名",
                        value="academicgraph",
                        disabled=not use_neo4j
                    )
                    
                    clear_neo4j = st.checkbox(
                        "清空Neo4j数据库", 
                        value=True,
                        disabled=not use_neo4j,
                        help="运行前清空Neo4j数据库中的内容"
                    )

                with st.expander("会话记忆设置"):
                    # 使用radio按钮提供三种记忆模式选择
                    memory_type = st.radio(
                        "记忆模式",
                        options=["buffer", "window_buffer", "summary"],
                        index=0,
                        help="选择AI如何管理对话历史：\n- buffer: 完整保留所有对话历史\n- window_buffer: 仅保留最近几轮对话\n- summary: 使用摘要压缩历史对话"
                    )
                    
                    # 如果选择window_buffer，提供窗口大小控制
                    window_size = 10  # 默认值
                    if memory_type == "window_buffer":
                        window_size = st.slider(
                            "窗口大小",
                            min_value=3,
                            max_value=20,
                            value=10,
                            help="保留最近多少轮对话记录"
                        )
                    
                    # 如果选择summary，提供最大令牌限制
                    max_token_limit = 2000  # 默认值
                    if memory_type == "summary":
                        max_token_limit = st.slider(
                            "最大令牌限制",
                            min_value=1000,
                            max_value=5000,
                            value=2000,
                            step=500,
                            help="摘要记忆的最大令牌数量"
                        )
                    
                    # 添加清除记忆的按钮
                    if st.button("清除对话记忆", help="清除当前会话的所有对话历史"):
                        st.session_state.clear_memory = True
                        st.success("对话记忆已清除！")
                    else:
                        st.session_state.clear_memory = False

                # 可视化设置
                with st.expander("可视化设置"):
                    # 使用系统字体而非YouYuan
                    available_fonts = [
                        "None (系统默认)",
                        "/home/zyh/anaconda3/envs/SDS/lib/python3.10/site-packages/matplotlib/mpl-data/fonts/ttf/SourceHanSans.ttc",
                        "/home/zyh/anaconda3/envs/SDS/lib/python3.10/site-packages/matplotlib/mpl-data/fonts/ttf/SimHei.ttf",
                        "/home/zyh/anaconda3/envs/SDS/lib/python3.10/site-packages/matplotlib/mpl-data/fontsttf/DejaVuSans.ttf"
                    ]
                    font_choice = st.selectbox(
                        "选择显示字体",
                        available_fonts,
                        index=1,
                        help="选择图表显示字体，解决中文显示问题"
                    )
                    
                    show_namespace = st.checkbox(
                        "显示命名空间图", 
                        value=True,
                        help="在结果中显示按命名空间划分的图"
                    )
            else:
                # 默认不启用高级功能
                use_milvus = True
                milvus_host = "localhost"
                milvus_port = "19530"
                use_neo4j = True
                neo4j_uri = "bolt://localhost:7687"
                neo4j_user = "neo4j"
                neo4j_password = "20031117"
                neo4j_database = "academicgraph"
                clear_neo4j = True
                font_path = "/home/zyh/anaconda3/envs/SDS/lib/python3.10/site-packages/matplotlib/mpl-data/fonts/ttf/SourceHanSans.ttc"
                font_choice = "/home/zyh/anaconda3/envs/SDS/lib/python3.10/site-packages/matplotlib/mpl-data/fonts/ttf/SourceHanSans.ttc"
                show_namespace = True

            # 开始按钮
            if not st.session_state.running:
                start_meeting = st.button("开始学术组会", type="primary")
                if start_meeting:
                    # 重置聊天消息
                    if "messages" in st.session_state:
                        st.session_state.messages = []
                    if "rendered_messages" in st.session_state:
                        st.session_state.rendered_messages = set()
                    st.session_state.running = True
                    st.session_state.completed = False
                    st.rerun()
            else:
                if st.button("停止并重新开始", type="secondary"):
                    st.session_state.running = False
                    st.session_state.completed = False
                    if "messages" in st.session_state:
                        st.session_state.messages = []
                    if "rendered_messages" in st.session_state:
                        st.session_state.rendered_messages = set()
                    st.rerun()
            
            st.markdown("---")
            st.markdown("""
            **参与人员**
            
            - 👨‍🏫 **许教授** - 人工智能领域资深专家，大语言模型研究方向带头人
            - 👨‍🎓 **李同学** - 博士研究生，研究多智能体协作系统
            - 👩‍🎓 **郭同学** - 硕士研究生，专注NLP与知识图谱
            - 👨‍💻 **吴同学** - 硕士研究生，计算机视觉与多模态融合方向
            
            **命名空间分类**
            
            <div class="namespace-tag namespace-user">用户</div> 
            <div class="namespace-tag namespace-dialogue">对话</div>
            <div class="namespace-tag namespace-task">任务</div>
            <div class="namespace-tag namespace-attachment">附件</div>
            """, unsafe_allow_html=True)
        
        # 显示附件提示
        attachments = list_attachments()
        if attachments:
            st.info(f"已加载 {len(attachments)} 个附件作为参考资料")
        
        # 主界面内容
        if st.session_state.running:
            st.markdown(f"## 当前讨论主题: {topic}")
            
            # 创建进度条
            progress_bar = st.progress(0)
            
            # 保存原始stdout
            original_stdout = sys.stdout
            
            try:
                # 创建聊天输出处理器
                chat_output = StreamlitChatOutput()
                
                # 重定向标准输出
                sys.stdout = chat_output
                
                # 验证API密钥
                if not use_local and not api_key.startswith("sk-"):
                    print("错误：请输入有效的API密钥或选择使用本地模型")
                    st.error("请输入有效的DeepSeek API密钥或勾选使用本地模型")
                    return

                # 设置环境变量
                os.environ["DEEPSEEK_API_KEY"] = api_key
                
                print("初始化学术组会系统...")
                st.info("初始化学术组会系统...")
                progress_bar.progress(0.1)
                
                # 处理字体选择
                if font_choice == "None (系统默认)":
                    font_path = None
                else:
                    font_path = font_choice
                
                # 收集附件路径
                attachment_paths = []
                if os.path.exists(ATTACHMENT_DIR):
                    attachment_paths = [os.path.join(ATTACHMENT_DIR, f) for f in os.listdir(ATTACHMENT_DIR)
                                      if os.path.isfile(os.path.join(ATTACHMENT_DIR, f))]
                
                # 更新进度条
                progress_bar.progress(0.2)
                
                # 调用接口函数
                scene_id = run_academic_meeting_database(
                    topic=topic,
                    use_remote=(not use_local),
                    deep_search=use_deep_search,
                    rounds=rounds,
                    use_milvus=use_milvus,
                    milvus_host=milvus_host,
                    milvus_port=milvus_port,
                    use_neo4j=use_neo4j,
                    neo4j_uri=neo4j_uri,
                    neo4j_user=neo4j_user,
                    neo4j_password=neo4j_password,
                    neo4j_database=neo4j_database,
                    clear_neo4j=clear_neo4j,
                    fontpath=font_path,
                    show_namespace=show_namespace,
                    generate_review=generate_review,
                    attachments=attachment_paths,  # 添加附件路径
                    local_model=st.session_state.local_model  # 添加本地模型参数
                )
                
                # 标记为完成
                st.session_state.completed = True
                progress_bar.progress(1.0)
                
                # 等待一会，确保图片已生成
                time.sleep(1)
                
                # 显示知识图谱
                st.subheader("🔍 生成的知识图谱")
                
                # graph_file = f"academic_meeting_{scene_id}.png"
                graph_file = "/home/zyh/code/SDS/semantic_map_case/academic_group_meeting/output/picture/academic_meeting_graph.png"
                if os.path.exists(graph_file):
                    st.image(graph_file, caption="学术组会知识图谱", use_container_width=True)
                    
                    # 下载按钮
                    with open(graph_file, "rb") as file:
                        st.download_button(
                            label="下载知识图谱",
                            data=file,
                            file_name=graph_file,
                            mime="image/png",
                            key="download_graph"
                        )
                else:
                    st.warning(f"未找到知识图谱文件: {graph_file}")
                
                # 如果启用了命名空间可视化
                if show_namespace:
                    namespace_file = "academic_meeting_namespaces.png"
                    if os.path.exists(namespace_file):
                        st.subheader("🔄 命名空间知识图谱")
                        st.image(namespace_file, caption="按命名空间划分的知识图谱", use_container_width=True)
                        
                        # 命名空间图下载按钮
                        with open(namespace_file, "rb") as file:
                            st.download_button(
                                label="下载命名空间图谱",
                                data=file,
                                file_name=namespace_file,
                                mime="image/png",
                                key="download_namespace"
                            )
                
                # Neo4j提示
                if use_neo4j:
                    st.info(f"""
                    知识图谱已成功导出到Neo4j数据库！
                    - URL: http://localhost:7474
                    - 用户名: {neo4j_user}
                    - 数据库: {neo4j_database}
                    """)
                
            except Exception as e:
                st.error(f"运行组会时出错: {str(e)}")
                import traceback
                st.error(traceback.format_exc())
            
            finally:
                # 恢复标准输出
                sys.stdout = original_stdout
                
                # 如果开启了终端输出显示
                if show_terminal_output and hasattr(chat_output, 'processor') and hasattr(chat_output.processor, 'buffer'):
                    remaining_output = chat_output.processor.buffer
                    if remaining_output:
                        st.text_area("剩余终端输出", remaining_output, height=300)
        
        else:
            # 显示欢迎信息和操作指南
            st.info("请在左侧配置组会参数，然后点击「开始学术组会」按钮")
            
            with st.expander("系统功能介绍", expanded=True):
                st.markdown("""
                ### 学术组会模拟系统 3.0
                
                本系统通过多智能体对话模拟学术组会讨论过程，支持实时显示和知识图谱生成。
                
                **主要功能:**
                
                1. **多智能体对话** - 模拟教授和学生之间的学术讨论
                2. **知识图谱构建** - 自动提取组会内容生成知识图谱
                3. **附件管理** - 上传论文、代码作为讨论材料
                4. **图数据库导出** - 将知识图谱存储到Neo4j数据库
                5. **向量数据库存储** - 使用Milvus保存对话向量
                
                **使用建议:**
                
                - 选择合适的讨论轮数，3-5轮可获得较完整的讨论
                - 上传相关学术论文作为参考材料可提升讨论质量
                - 对专业性强的主题，建议开启网页搜索以获取最新信息
                """)
            
            # 显示示例图片
            col1, col2 = st.columns(2)
            with col1:
                st.image("/home/zyh/code/SDS/semantic_map_case/academic_group_meeting/front_end_pictures/2.png", 
                         caption="系统生成的知识图谱示例", use_container_width=True)
            with col2:
                st.image("/home/zyh/code/SDS/semantic_map_case/academic_group_meeting/front_end_pictures/3.png", 
                         caption="按命名空间划分的图谱示例", use_container_width=True)
    
    # 附件管理标签
    with tab2:
        manage_attachments()

if __name__ == "__main__":
    main()
