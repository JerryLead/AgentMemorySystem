import json
from openai import OpenAI
import requests

messages = [{"role": "system", "content": "You are a helpful assistant."}]

class deepseek_remote:
    def __init__(self, api_key="sk-fbab400c86184b0daf9bd59467d35772", base_url="https://api.deepseek.com"):
        self.client = OpenAI(
            api_key = api_key,
            base_url = base_url
        )
        # self.messages = None

    def get_response(self, messages, temperature=1.0, max_tokens=4096)->str:
        """
        获取助手回复

        Args:
            messages: 对话历史
            temperature: 控制输出多样性，默认1.0
            max_tokens: 最大生成长度，默认4096
        
        Returns:
            str: 助手回复
        """
        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                max_tokens=max_tokens,  #默认4096
                # response_format={
                #     'type': 'text'
                # },
                temperature=temperature,  # 控制输出多样性，默认1.0
                stream=False  # 根据传入的参数控制流式输出，默认false
            )
            return response.choices[0].message.content
            # return response
        except Exception as e:
            print(f"API 请求失败: {str(e)}")
            return None
        
    def get_stream_response(self, messages, temperature=1.0, max_tokens=4096):
        """
        获取流式助手回复

        Args:
            messages: 对话历史
            temperature: 控制输出多样性，默认1.0
            max_tokens: 最大生成长度，默认4096
        
        Returns:
            ChatCompletionChunks: 流式输出结果
        """
        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                max_tokens=max_tokens,  #默认4096
                # response_format={
                #     'type': 'text'
                # },
                temperature=temperature,  # 控制输出多样性，默认1.0
                stream=True  # 根据传入的参数控制流式输出，默认false
            )
            return response
        except Exception as e:
            print(f"API 请求失败: {str(e)}")
            return None

 
    # 流式拼接（保存对话历史）
    # def get_stream_response(self, response):
    #     """
    #     提取并拼接流式返回的内容,用于保存对话历史
    #     Args:
    #         response: API 返回的流式输出结果
    #     Returns:
    #         str: 拼接后的对话历史
    #     """
    #     return ''.join(
    #         chunk['choices'][0]['delta']['content']
    #         for chunk in response
    #         if chunk.get('choices') and chunk['choices'][0].get('delta') and chunk['choices'][0]['delta'].get('content')
    #     )

    def collect_stream_response(self, response):
        """
        将流式响应中返回的每个 chunk 整合成完整文本，
        注意：由于 response 是生成器，这里先将其转为列表，避免后续重复消费。
        Returns:
            str: 拼接后的对话历史文本
        """
        collected_chunks = []
        response_list = list(response)
        for chunk in response_list:
            try:
                content = chunk.choices[0].delta.content
            except AttributeError:
                content = chunk.get('choices', [{}])[0].get('delta', {}).get('content', '')
            collected_chunks.append(content)
        return ''.join(collected_chunks)
    

class deepseek_local:
    # def __init__(self):
    #     self.client = OpenAI(
    #         api_key="sk-fbab400c86184b0daf9bd59467d35772",
    #         base_url="http://"
    #     )

    # def get_response(self, messages, model="deepseek-r1:1.5b")->str:# , temperature=1.0, max_tokens=4096):
    #     """获取助手回复，返回字符串
    #     Args:
    #         messages: prompt
    #         model: 使用的模型名称
    #     Returns:
    #         str: 助手回复
    #     """
    #     try:
    #         response = requests.post(
    #             url="http://localhost:11434/api/generate",
    #             json={"model": model, "prompt": messages, "stream": False},
    #             stream=False # 启用流式请求
    #         )
    #         # if stream:
    #         #     def generate():
    #         #         print("助手: ", end='', flush=True)
    #         #         for line in response.iter_lines():
    #         #             if line:
    #         #                     # 打印原始数据流内容
    #         #                     # print(line.decode('utf-8'))
    #         #                     # 尝试解析JSON，如果不需要可以注释掉下面两行
    #         #                     try:
    #         #                         json_obj = json.loads(line)
    #         #                         # print("解析后的 JSON 对象：", json_obj)
    #         #                         if json_obj.get("done", "")==False:
    #         #                             print(json_obj.get("response", ""), end='', flush=True)
    #         #                     except Exception as e:
    #         #                         print(f"解析流数据失败: {e}")    
    #         #             # if line:
    #         #             #     try:
    #         #             #         chunk = json.loads(line)
    #         #             #         yield chunk
    #         #             #         if chunk.get("done", False):
    #         #             #             break
    #         #             #     except Exception as e:
    #         #             #         print(f"解析流数据失败: {e}")
    #         #     return generate()
    #         # else:
    #         response_json = response.json()
    #         return response_json.get("response", "")
    #     except Exception as e:
    #         print(f"API 请求失败: {str(e)}")
    #         return None
    
    def get_response(self, messages, model="deepseek-r1:1.5b")->str:
        """获取助手回复，返回字符串
        Args:
            messages: prompt或消息列表
            model: 使用的模型名称
        Returns:
            str: 助手回复
        """
        try:
            # 检查是否为列表类型，如果是则提取提示内容
            if isinstance(messages, list):
                # 将消息列表转换为单个字符串
                prompt_text = ""
                for msg in messages:
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    if role == "system":
                        prompt_text += f"系统: {content}\n"
                    elif role == "user":
                        prompt_text += f"用户: {content}\n"
                    elif role == "assistant":
                        prompt_text += f"助手: {content}\n"
                prompt = prompt_text.strip()
            else:
                # 已经是字符串，直接使用
                prompt = messages
                
            response = requests.post(
                url="http://localhost:11434/api/generate",
                json={"model": model, "prompt": prompt, "stream": False},
                stream=False
            )
            response_json = response.json()
            return response_json.get("response", "")
        except Exception as e:
            print(f"API 请求失败: {str(e)}")
            return None
    
    
    def get_stream_response(self, messages, model="deepseek-r1:1.5b")->str:
        """获取助手回复,流式的回复直接打印，返回全部结果

        Args:
            messages: 对话历史

        Returns:
            str: 助手回复的全部结果
        """
        total_response = ""
        try:
            # response = requests.post(
            #     url="http://localhost:11434/api/generate",
            #     json={"model": model, "prompt": messages, "stream": True},
            #     stream=True  # 启用流式请求
            # )
             # 检查是否为列表类型，如果是则提取提示内容
            if isinstance(messages, list):
                # 将消息列表转换为单个字符串
                prompt_text = ""
                for msg in messages:
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    if role == "system":
                        prompt_text += f"系统: {content}\n"
                    elif role == "user":
                        prompt_text += f"用户: {content}\n"
                    elif role == "assistant":
                        prompt_text += f"助手: {content}\n"
                prompt = prompt_text.strip()
            else:
                # 已经是字符串，直接使用
                prompt = messages
            response = requests.post(
                url="http://localhost:11434/api/generate",
                json={"model": model, "prompt": prompt, "stream": True},
                stream=True
            )
            
        # 剩余代码保持不变...
            print("助手: ", end='', flush=True)
            for line in response.iter_lines():
                if line:
                        # 打印原始数据流内容
                        # print(line.decode('utf-8'))
                        # 尝试解析JSON，如果不需要可以注释掉下面两行
                        try:
                            json_obj = json.loads(line)
                            # print("解析后的 JSON 对象：", json_obj)
                            if json_obj.get("done", "")==False:
                                print(json_obj.get("response", ""), end='', flush=True)
                                total_response += json_obj.get("response", "")
                            if json_obj.get("done", "")==True:
                                print("\n")
                        except Exception as e:
                            print(f"解析流数据失败: {e}") 
            return total_response
        except Exception as e:
            print(f"API 请求失败: {str(e)}")
            return None

        # if not stream:
        #     try:
        #         response = requests.post(
        #             url="http://localhost:11434/api/generate",
        #             json={"model": "deepseek-r1:1.5b", "prompt": messages, "stream": stream}#, "temperature": temperature, "max_tokens": max_tokens},
        #         )
        #         response_json = response.json()
        #         return response_json.get("response", "")
        #     except Exception as e:
        #         print(f"API 请求失败: {str(e)}")
        #         return None
        # else:
        #     try:
        #         response = requests.post(
        #             url="http://localhost:11434/api/generate",
        #             json={"model": "deepseek-r1:1.5b", "prompt": messages, "stream": stream}#, "temperature": temperature, "max_tokens": max_tokens},
        #         )
        #         response_json = response.json()
        #         # return response_json.get("response", "")
        #         return response_json.get("done",""),response_json.get("response", "")
        #     except Exception as e:
        #         print(f"API 请求失败: {str(e)}")
        #         return None
        
    def collect_stream_response(self, response):
        """
        将流式响应中返回的每个 chunk 整合成完整文本，
        注意：由于 response 是生成器，这里先将其转为列表，避免后续重复消费。
        Returns:
            str: 拼接后的对话历史文本
        """
        collected_chunks = []
        response_list = list(response)
        for chunk in response_list:
            try:
                content = chunk
            except AttributeError:
                content = chunk.get('choices', [{}])[0].get('delta', {}).get('content', '')
            collected_chunks.append(content)
        return ''.join(collected_chunks)

def chat(client: deepseek_remote, stream_mode=False):
    print("欢迎使用 DeepSeek 聊天助手！输入 '退出' 来结束对话。")

    while True:
        user_input = input("你: ")
        if user_input.lower() in ["退出", "exit", "quit"]:
            print("聊天结束。再见！")
            break

        # 添加用户消息到 deepseek_client 内的对话历史
        messages.append({"role": "user", "content": user_input})
        # 使用 client.messages 进行 get_response 调用
        # 对于API类型的客户端，需要传入对话历史，也即json字符串
        if stream_mode:
            response = client.get_stream_response(messages)
        else:
            response = client.get_response(messages)

        if response:
            if stream_mode:
                print("助手: ", end='', flush=True)
                # 逐步输出流式响应，并保持历史记录
                collected_chunks = []
                for chunk in response:
                    chunk_message = chunk.choices[0].delta.content
                    collected_chunks.append(chunk_message)
                    print(chunk_message, end='', flush=True)
                print()
                assistant_response_history = ''.join(collected_chunks)
            else:
                assistant_response_history = response
                print(f"助手: {assistant_response_history}")
    
            messages.append({"role": "assistant", "content": assistant_response_history})
        else:
            print("未能获取回复，请查找原因。")

def chat_local(client: deepseek_local, stream_mode=False):
    print("欢迎使用 DeepSeek 聊天助手！输入 '退出' 来结束对话。")

    while True:
        user_input = input("你: ")
        if user_input.lower() in ["退出", "exit", "quit"]:
            print("聊天结束。再见！")
            break

        # 添加用户消息到 deepseek_client 内的对话历史
        messages.append({"role": "user", "content": user_input})
        # 使用 client.messages 进行 get_response 调用
        if stream_mode:
            response = client.get_stream_response(user_input)
        else:
            response = client.get_response(user_input)
            print(f"助手: {response}")

        # # if response:
        # if stream_mode:
        #     print("助手: ", end='', flush=True)
        #     # 逐步输出流式响应，并保持历史记录
        #     collected_chunks = []
        #     for chunk in response:
        #         chunk_message = chunk
        #         collected_chunks.append(chunk_message)
        #         print(chunk_message, end='', flush=True)
        #     print()
        #     assistant_response_history = ''.join(collected_chunks)
        # else:
        #     assistant_response_history = response
        #     print(f"助手: {assistant_response_history}")

        messages.append({"role": "assistant", "content": response})
    # else:
        #     print("未能获取回复，请查找原因。")

if __name__ == "__main__":
    # 选择客户端类型
    client_choice = input("请选择客户端类型（输入 1 使用 deepseek_remote, 输入 2 使用 deepseek_local）：").strip()
    while client_choice not in ["1", "2"]:
        client_choice = input("输入无效，请重新输入（1 或 2）：").strip()
    client = deepseek_remote() if client_choice == "1" else deepseek_local()
    
    # 选择是否流式输出
    stream_choice = input("请选择是否需要流式输出 (输入 True 或 False)：").lower().strip()
    while stream_choice not in ["true", "false"]:
        stream_choice = input("输入无效，请重新输入（True 或 False）：").lower().strip()
    stream_mode = True if stream_choice == "true" else False
    
    # 开始对话，依据选择调用对应的 chat 函数
    if client_choice == "1":
        chat(client, stream_mode=stream_mode)
    else:
        chat_local(client, stream_mode=stream_mode)
    
    print("对话历史：", messages)

    # deepseek_remote_test = deepseek_remote()
    # messages = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "你是谁"}]
    # response=deepseek_remote_test.get_response(messages)
    # print(response)

    # deepseek_local_test = deepseek_local()
    # response=deepseek_local_test.get_stream_response("你是谁")
    # print(response)