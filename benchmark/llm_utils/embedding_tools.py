from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from tqdm import tqdm

def get_embeddings(model_name, texts, batch_size=32, max_length=512, device=None):
    """
    通用文本向量化函数，支持 HuggingFace 主流 embedding 模型。
    :param model_name: 模型名（如 'BAAI/bge-large-zh', 'moka-ai/m3e-base', 'sentence-transformers/all-MiniLM-L6-v2' 等）
    :param texts: 文本列表
    :param batch_size: 批处理大小
    :param max_length: 最大token长度
    :param device: 'cuda' 或 'cpu'，默认自动选择
    :return: (N, D) 的 numpy 数组
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    all_embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
            batch_texts = texts[i:i+batch_size]
            encoded = tokenizer(batch_texts, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
            input_ids = encoded['input_ids'].to(device)
            attention_mask = encoded['attention_mask'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            # 兼容不同模型的输出
            if hasattr(outputs, 'last_hidden_state'):
                # 句向量池化（取[CLS]或平均池化）
                if hasattr(model.config, 'pooler_type') and model.config.pooler_type == 'cls':
                    embeddings = outputs.last_hidden_state[:, 0, :]
                else:
                    # 平均池化
                    mask = attention_mask.unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
                    sum_embeddings = torch.sum(outputs.last_hidden_state * mask, 1)
                    sum_mask = torch.clamp(mask.sum(1), min=1e-9)
                    embeddings = sum_embeddings / sum_mask
            elif hasattr(outputs, 'pooler_output'):
                embeddings = outputs.pooler_output
            else:
                raise ValueError("未知的模型输出结构")
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings.cpu().numpy())
    return np.vstack(all_embeddings)

if __name__ == "__main__":
    # 示例用法
    model_name = "BAAI/bge-large-zh"
    texts = ["你好，世界！", "这是一个测试。", "深度学习很有趣。"]
    embeddings = get_embeddings(model_name, texts, batch_size=2)
    print(embeddings.shape)  # 输出 (3, D) 的形状

    # 例子：英文模型
    embs = get_embeddings('sentence-transformers/all-MiniLM-L6-v2', ["Hello world", "AI is great"], batch_size=16)
    print(embs.shape)  # 输出 (2, D) 的形状