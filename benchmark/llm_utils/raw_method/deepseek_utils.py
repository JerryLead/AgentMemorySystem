import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from embedding_tools import get_embeddings
import pickle
import random
import os, json
from tqdm import tqdm
import time
# from global_methods import run_chatgpt
# from task_eval.rag_utils import get_embeddings
import tiktoken
import numpy as np
from global_methods import run_deepseek

MAX_LENGTH={'DeepSeek-R1-0528': 65536,
            'DeepSeek-V3-0324':65536}
PER_QA_TOKEN_BUDGET = 50

# 中文提示词
QA_PROMPT = """
请根据以上上下文，用简短的短语回答下列问题。尽量直接引用上下文中的原文作答。

问题：{} 简短回答：
"""

QA_PROMPT_CAT_5 = """
请根据以上上下文，回答下列问题。

问题：{} 简短回答：
"""

# QA_PROMPT_BATCH = """
# 请根据以上对话，分别用简短的短语回答下列每个问题。将答案以字符串列表的json格式输出，以方括号开始和结束。

# """

QA_PROMPT_BATCH = """
请根据以上对话，为下列每个问题写出简短的答案。
将答案以json字典的形式输出，每个条目的key为问题编号，value为简短答案。
对于专有名词请使用单引号，json元素请用双引号包裹。尽量直接引用对话中的原文作答。

"""

# 如果无法回答问题，请写“无可用信息”。

CONV_START_PROMPT = "以下是两个人的对话：{} 和 {}。对话发生在多天，每次对话的日期在开头已注明。\n\n"


def process_ouput(text):
    # 处理模型输出，兼容单双引号
    single_quote_count = text.count("'")
    double_quote_count = text.count('"')
    if single_quote_count > double_quote_count:
        text = text.replace('"', "")
        text = text.replace("'", '"')
        # print(text)
        return json.loads(text)
    else:
        return json.loads(text)


def prepare_for_rag(args, data):
    # 为RAG检索准备数据和向量
    dataset_prefix = os.path.splitext(os.path.split(args.data_file)[-1])[0]

    if args.rag_mode == "summary":
        # 检查摘要向量是否存在
        assert os.path.exists(os.path.join(args.emb_dir, '%s_session_summary_%s.pkl' % (dataset_prefix, data['sample_id']))), "摘要及向量不存在：%s" % data['sample_id']
        database = pickle.load(open(os.path.join(args.emb_dir, '%s_session_summary_%s.pkl' % (dataset_prefix, data['sample_id'])), 'rb'))

    elif args.rag_mode == 'dialog':
        # 检查对话向量是否存在
        if not os.path.exists(os.path.join(args.emb_dir, '%s_dialog_%s.pkl' % (dataset_prefix, data['sample_id']))):

            dialogs = []
            date_times = []
            context_ids = []
            session_nums = [int(k.split('_')[-1]) for k in data['conversation'].keys() if 'session' in k and 'date_time' not in k]
            for i in range(min(session_nums), max(session_nums) + 1):
            
                date_time = data['conversation']['session_%s_date_time' % i]
                for dialog in data['conversation']['session_%s' % i]:
                    context_ids.append(dialog['dia_id'])
                    date_times.append(date_time)
                    if 'blip_caption' in dialog:
                        dialogs.append(dialog['speaker'] + '说：“' + dialog['text'] + '”并分享了' + dialog['blip_caption'])
                    else:
                        dialogs.append(dialog['speaker'] + '说：“' + dialog['text'] + '”')

            print("为%d条对话获取向量" % len(dialogs))
            embeddings = get_embeddings(args.retriever, dialogs, 'context')
            assert embeddings.shape[0] == len(dialogs), "向量和对话数量不一致"
            database = {'embeddings': embeddings,
                             'date_time': date_times,
                             'dia_id': context_ids,
                             'context': dialogs}

            with open(os.path.join(args.emb_dir, '%s_dialog_%s.pkl' % (dataset_prefix, data['sample_id'])), 'wb') as f:
                pickle.dump(database, f)

        else:
            database = pickle.load(open(os.path.join(args.emb_dir, '%s_dialog_%s.pkl' % (dataset_prefix, data['sample_id'])), 'rb'))

    elif args.rag_mode == 'observation':
        # 检查observation向量是否存在
        assert os.path.exists(os.path.join(args.emb_dir, '%s_observation_%s.pkl' % (dataset_prefix, data['sample_id']))), "观察及向量不存在：%s" % data['sample_id']
        database = pickle.load(open(os.path.join(args.emb_dir, '%s_observation_%s.pkl' % (dataset_prefix, data['sample_id'])), 'rb'))

    else:
        raise ValueError
    
    print("为%d个问题获取向量" % len(data['qa']))
    question_embeddings = get_embeddings(args.retriever, [q['question'] for q in data['qa']], 'query')

    return database, question_embeddings


def get_cat_5_answer(model_prediction, answer_key):
    # 处理第5类问题的答案
    model_prediction = model_prediction.strip().lower()
    if len(model_prediction) == 1:
        if 'a' in model_prediction:
            return answer_key['a']
        else:
            return answer_key['b']
    elif len(model_prediction) == 3:
        if '(a)' in model_prediction:
            return answer_key['a']
        else:
            return answer_key['b']
    else:
        return model_prediction


def get_rag_context(context_database, query_vector, args):
    # 检索RAG上下文
    output = np.dot(query_vector, context_database['embeddings'].T)
    sorted_outputs = np.argsort(output)[::-1]
    sorted_context = [context_database['context'][idx] for idx in sorted_outputs[:args.top_k]]
    
    sorted_context_ids = []
    for idx in sorted_outputs[:args.top_k]:
        context_id = context_database['dia_id'][idx]
        if type(context_id) == str:
            if ',' in context_id:
                context_id = [s.strip() for s in context_id.split(',')]
        if type(context_id) == list:
            sorted_context_ids.extend(context_id)
        else:
            sorted_context_ids.append(context_id)

    sorted_date_times = [context_database['date_time'][idx] for idx in sorted_outputs[:args.top_k]]
    if args.rag_mode in ['dialog', 'observation']:
        query_context = '\n'.join([date_time + ': ' + context for date_time, context in zip(sorted_date_times, sorted_context)])
    else:
        query_context = '\n\n'.join([date_time + ': ' + context for date_time, context in zip(sorted_date_times, sorted_context)])

    return query_context, sorted_context_ids


def get_input_context(data, num_question_tokens, encoding, args):
    # 获取输入上下文
    query_conv = ''
    min_session = -1
    stop = False
    session_nums = [int(k.split('_')[-1]) for k in data.keys() if 'session' in k and 'date_time' not in k]
    for i in range(min(session_nums), max(session_nums) + 1):
        if 'session_%s' % i in data:
            query_conv += "\n\n"
            for dialog in data['session_%s' % i][::-1]:
                turn = ''
                turn = dialog['speaker'] + '说：“' + dialog['text'] + '”' + '\n'
                if "blip_caption" in dialog:
                    turn += '并分享了%s。' % dialog["blip_caption"]
                turn += '\n'
        
                num_tokens = len(encoding.encode('日期：' + data['session_%s_date_time' % i] + '\n' + '对话：\n' + turn))
                if (num_tokens + len(encoding.encode(query_conv)) + num_question_tokens) < (MAX_LENGTH[args.model]-(PER_QA_TOKEN_BUDGET*(args.batch_size))): # 20 tokens assigned for answers
                    query_conv = turn + query_conv
                else:
                    min_session = i
                    stop = True
                    break
            query_conv = '日期：' + data['session_%s_date_time' % i] + '\n' + '对话：\n' + query_conv
        if stop:
            break

    return query_conv

# 直接用字符长度近似控制上下文长度
def estimate_token_length(text):
    # 简单估算：1 token ≈ 2个汉字或1.3个英文单词
    return int(len(text) / 2)

def get_deepseek_answers(in_data, out_data, prediction_key, args):
    """
    适配DeepSeek接口的批量问答主流程，不依赖tiktoken，直接用字符长度近似控制输入长度。
    """

    assert len(in_data['qa']) == len(out_data['qa']), (len(in_data['qa']), len(out_data['qa']))

    # 起始提示词
    speakers_names = list(set([d['speaker'] for d in in_data['conversation']['session_1']]))
    start_prompt = CONV_START_PROMPT.format(speakers_names[0], speakers_names[1])
    start_tokens = estimate_token_length(start_prompt)

    if args.use_rag:
        assert args.batch_size == 1, "RAG模式下batch size必须为1。"
        context_database, query_vectors = prepare_for_rag(args, in_data)
    elif args.context_datebase is not None:
        context_database = args.context_datebase
    else:
        context_database, query_vectors = None, None

    for batch_start_idx in tqdm(range(0, len(in_data['qa']), args.batch_size), desc='生成答案'):
        questions = []
        include_idxs = []
        cat_5_idxs = []
        cat_5_answers = []
        for i in range(batch_start_idx, batch_start_idx + args.batch_size):

            if i >= len(in_data['qa']):
                break

            qa = in_data['qa'][i]

            if prediction_key not in out_data['qa'][i] or args.overwrite:
                include_idxs.append(i)
            else:
                continue

            if qa['category'] == 2:
                questions.append(qa['question'] + ' 请结合对话日期，给出一个大致的日期作为答案。')
            elif qa['category'] == 5:
                question = qa['question'] + " 请选择正确答案：(a) {} (b) {}。"
                if random.random() < 0.5:
                    question = question.format('对话中未提及', qa['answer'])
                    answer = {'a': '对话中未提及', 'b': qa['answer']}
                else:
                    question = question.format(qa['answer'], '对话中未提及')
                    answer = {'b': '对话中未提及', 'a': qa['answer']}

                cat_5_idxs.append(len(questions))
                questions.append(question)
                cat_5_answers.append(answer)
            else:
                questions.append(qa['question'])

        if questions == []:
            continue

        if args.use_rag:
            query_conv, context_ids = get_rag_context(context_database, query_vectors[include_idxs][0], args)
        else:
            question_prompt = QA_PROMPT_BATCH + "\n".join(["%s: %s" % (k, q) for k, q in enumerate(questions)])
            num_question_tokens = estimate_token_length(question_prompt)
            # 用字符长度近似控制上下文长度
            query_conv = ''
            min_session = -1
            stop = False
            data = in_data['conversation']
            session_nums = [int(k.split('_')[-1]) for k in data.keys() if 'session' in k and 'date_time' not in k]
            for i in range(min(session_nums), max(session_nums) + 1):
                if 'session_%s' % i in data:
                    query_conv += "\n\n"
                    for dialog in data['session_%s' % i][::-1]:
                        turn = dialog['speaker'] + '说：“' + dialog['text'] + '”\n'
                        if "blip_caption" in dialog:
                            turn += '并分享了%s。' % dialog["blip_caption"]
                        turn += '\n'
                        num_tokens = estimate_token_length('日期：' + data['session_%s_date_time' % i] + '\n' + '对话：\n' + turn)
                        if (num_tokens + estimate_token_length(query_conv) + num_question_tokens) < (MAX_LENGTH[args.model] - (PER_QA_TOKEN_BUDGET * (args.batch_size))):
                            query_conv = turn + query_conv
                        else:
                            min_session = i
                            stop = True
                            break
                    query_conv = '日期：' + data['session_%s_date_time' % i] + '\n' + '对话：\n' + query_conv
                if stop:
                    break
            query_conv = start_prompt + query_conv

        if args.batch_size == 1:
            query = query_conv + '\n\n' + QA_PROMPT.format(questions[0]) if len(cat_5_idxs) == 0 else query_conv + '\n\n' + QA_PROMPT_CAT_5.format(questions[0])
            answer = run_deepseek(query, num_gen=1, num_tokens_request=32,
                                  model=args.model,
                                  temperature=0, wait_time=2)

            if len(cat_5_idxs) > 0:
                answer = get_cat_5_answer(answer, cat_5_answers[0])

            out_data['qa'][include_idxs[0]][prediction_key] = answer.strip()
            if args.use_rag:
                out_data['qa'][include_idxs[0]][prediction_key + '_context'] = context_ids

        else:
            query = query_conv + '\n' + question_prompt

            trials = 0
            while trials < 3:
                try:
                    trials += 1
                    print("第%s/3次尝试" % trials)
                    answer = run_deepseek(query, num_gen=1, num_tokens_request=args.batch_size * PER_QA_TOKEN_BUDGET,
                                         model=args.model,
                                         temperature=0, wait_time=2)
                    answer = answer.replace('\\"', "'").replace('json', '').replace('`', '').strip().replace("\\'", "")
                    answers = process_ouput(answer.strip())
                    break

                except Exception as e:
                    print('第%s/3次出错' % trials, e)
                    raise ValueError

            for k, idx in enumerate(include_idxs):
                try:
                    answers = process_ouput(answer.strip())
                    if k in cat_5_idxs:
                        predicted_answer = get_cat_5_answer(answers[str(k)], cat_5_answers[cat_5_idxs.index(k)])
                        out_data['qa'][idx][prediction_key] = predicted_answer
                    else:
                        try:
                            out_data['qa'][idx][prediction_key] = str(answers[str(k)]).replace('(a)', '').replace('(b)', '').strip()
                        except:
                            out_data['qa'][idx][prediction_key] = ', '.join([str(n) for n in list(answers[str(k)].values())])
                except:
                    try:
                        answers = json.loads(answer.strip())
                        if k in cat_5_idxs:
                            predicted_answer = get_cat_5_answer(answers[k], cat_5_answers[cat_5_idxs.index(k)])
                            out_data['qa'][idx][prediction_key] = predicted_answer
                        else:
                            out_data['qa'][idx][prediction_key] = answers[k].replace('(a)', '').replace('(b)', '').strip()
                    except:
                        if k in cat_5_idxs:
                            predicted_answer = get_cat_5_answer(answer.strip(), cat_5_answers[cat_5_idxs.index(k)])
                            out_data['qa'][idx][prediction_key] = predicted_answer
                        else:
                            out_data['qa'][idx][prediction_key] = json.loads(answer.strip().replace('(a)', '').replace('(b)', '').split('\n')[k])[0]

    return out_data


# def get_gpt_answers(in_data, out_data, prediction_key, args):
#     # 获取GPT模型答案

#     encoding = tiktoken.encoding_for_model('gpt-3.5-turbo-16k' if any([k in args.model for k in ['16k', '12k', '8k', '4k']]) else args.model)
#     assert len(in_data['qa']) == len(out_data['qa']), (len(in_data['qa']), len(out_data['qa']))

#     # 起始提示词
#     speakers_names = list(set([d['speaker'] for d in in_data['conversation']['session_1']]))
#     start_prompt = CONV_START_PROMPT.format(speakers_names[0], speakers_names[1])
#     start_tokens = len(encoding.encode(start_prompt))

#     if args.use_rag:
#         assert args.batch_size == 1, "RAG模式下batch size必须为1。"
#         context_database, query_vectors = prepare_for_rag(args, in_data)
#     else:
#         context_database, query_vectors = None, None

#     for batch_start_idx in tqdm(range(0, len(in_data['qa']), args.batch_size), desc='生成答案'):
#         questions = []
#         include_idxs = []
#         cat_5_idxs = []
#         cat_5_answers = []
#         for i in range(batch_start_idx, batch_start_idx + args.batch_size):

#             if i>=len(in_data['qa']):
#                 break

#             qa = in_data['qa'][i]
            
#             if prediction_key not in out_data['qa'][i] or args.overwrite:
#                 include_idxs.append(i)
#             else:
#                 continue

#             if qa['category'] == 2:
#                 questions.append(qa['question'] + ' 请结合对话日期，给出一个大致的日期作为答案。')
#             elif qa['category'] == 5:
#                 question = qa['question'] + " 请选择正确答案：(a) {} (b) {}。"
#                 if random.random() < 0.5:
#                     question = question.format('对话中未提及', qa['answer'])
#                     answer = {'a': '对话中未提及', 'b': qa['answer']}
#                 else:
#                     question = question.format(qa['answer'], '对话中未提及')
#                     answer = {'b': '对话中未提及', 'a': qa['answer']}

#                 cat_5_idxs.append(len(questions))
#                 questions.append(question)
#                 cat_5_answers.append(answer)
#             else:
#                 questions.append(qa['question'])

#         if questions == []:
#             continue

#         if args.use_rag:
#             query_conv, context_ids = get_rag_context(context_database, query_vectors[include_idxs][0], args) # rag模式下batch size为1
#         else:
#             question_prompt =  QA_PROMPT_BATCH + "\n".join(["%s: %s" % (k, q) for k, q in enumerate(questions)])
#             num_question_tokens = len(encoding.encode(question_prompt))
#             query_conv = get_input_context(in_data['conversation'], num_question_tokens + start_tokens, encoding, args)
#             query_conv = start_prompt + query_conv

#         # print("%s tokens in query" % len(encoding.encode(query_conv)))

#         if 'gpt-4' in args.model:
#             time.sleep(5)
#         elif 'gpt-4' in args.model:
#             time.sleep(1)

#         if args.batch_size == 1:
#             query = query_conv + '\n\n' + QA_PROMPT.format(questions[0]) if len(cat_5_idxs) == 0 else query_conv + '\n\n' + QA_PROMPT_CAT_5.format(questions[0])
#             answer = run_deepseek(query, num_gen=1, num_tokens_request=32, 
#                     model='chatgpt' if 'gpt-3.5' in args.model else args.model, 
#                     use_16k=True if any([k in args.model for k in ['16k', '12k', '8k', '4k']]) else False, 
#                     temperature=0, wait_time=2)
            
#             if len(cat_5_idxs) > 0:
#                 answer = get_cat_5_answer(answer, cat_5_answers[0])

#             out_data['qa'][include_idxs[0]][prediction_key] = answer.strip()
#             if args.use_rag:
#                 out_data['qa'][include_idxs[0]][prediction_key + '_context'] = context_ids

#         else:
#             query = query_conv + '\n' + question_prompt
            
#             trials = 0
#             while trials < 3:
#                 try:
#                     trials += 1
#                     print("第%s/3次尝试" % trials)
#                     answer = run_deepseek(query, num_gen=1, num_tokens_request=args.batch_size*PER_QA_TOKEN_BUDGET, 
#                             model='chatgpt' if 'gpt-3.5' in args.model else args.model, 
#                             use_16k=True if any([k in args.model for k in ['16k', '12k', '8k', '4k']]) else False, 
#                             temperature=0, wait_time=2)
#                     answer = answer.replace('\\"', "'").replace('json','').replace('`','').strip().replace("\\'", "")
#                     answers = process_ouput(answer.strip())
#                     break

#                 except Exception as e:
#                     print('第%s/3次出错' % trials, e)
#                     raise ValueError
            
#             for k, idx in enumerate(include_idxs):
#                 try:
#                     answers = process_ouput(answer.strip())
#                     if k in cat_5_idxs:
#                         predicted_answer = get_cat_5_answer(answers[str(k)], cat_5_answers[cat_5_idxs.index(k)])
#                         out_data['qa'][idx][prediction_key] = predicted_answer
#                     else:
#                         try:
#                             out_data['qa'][idx][prediction_key] = str(answers[str(k)]).replace('(a)', '').replace('(b)', '').strip()
#                         except:
#                             out_data['qa'][idx][prediction_key] = ', '.join([str(n) for n in list(answers[str(k)].values())])
#                 except:
#                     try:
#                         answers = json.loads(answer.strip())
#                         if k in cat_5_idxs:
#                             predicted_answer = get_cat_5_answer(answers[k], cat_5_answers[cat_5_idxs.index(k)])
#                             out_data['qa'][idx][prediction_key] = predicted_answer
#                         else:
#                             out_data['qa'][idx][prediction_key] = answers[k].replace('(a)', '').replace('(b)', '').strip()
#                     except:
#                         if k in cat_5_idxs:
#                             predicted_answer = get_cat_5_answer(answer.strip(), cat_5_answers[cat_5_idxs.index(k)])
#                             out_data['qa'][idx][prediction_key] = predicted_answer
#                         else:
#                             out_data['qa'][idx][prediction_key] = json.loads(answer.strip().replace('(a)', '').replace('(b)', '').split('\n')[k])[0]

#     return out_data