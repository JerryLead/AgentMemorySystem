# filepath: /home/zyh/code/SDS/semantic_map/semantic_map.py
import abc
import os
import pickle
import logging
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from PIL import Image
from data_type import BaseDataType

logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO
)

class BaseSemanticMap(abc.ABC):
    def __init__(self,
                 image_embedding_model="clip-ViT-B-32",
                 text_embedding_model="sentence-transformers/clip-ViT-B-32-multilingual-v1",
                 embedding_dim=512,
                 index_type="flat"):
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        # 数据格式：[(key, value, DataType, embedding), ...]
        self.data = []
        self.index = None
        self.text_model = SentenceTransformer(text_embedding_model)
        self.image_model = SentenceTransformer(image_embedding_model)
        self._init_index()

    def _init_index(self):
        if self.index_type == "flat":
            self.index = faiss.IndexFlatL2(self.embedding_dim)
        else:
            raise ValueError(f"Unsupported index_type: {self.index_type}")

    def _get_text_embedding(self, text: str):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Text must be a non-empty string.")
        emb = self.text_model.encode(text)
        return emb.astype(np.float32)

    def _get_image_embedding(self, image_path: str):
        if not isinstance(image_path, str):
            raise ValueError("Image path must be a string.")
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        emb = self.image_model.encode(Image.open(image_path))
        return emb.astype(np.float32)
    
    # 对于不同的类型，通过继承 BaseDataType 类，实现不同的数据类型处理逻辑
    @abc.abstractmethod
    def insert(self, key: str, value: dict, datatype):
        """
        抽象方法，由用户自行实现数据插入及嵌入计算逻辑
        """
        pass
    # def insert(self, key: str, value: dict, datatype: DataType):
    #     # 根据不同类型生成嵌入文本（具体实现可参考原代码）
    #     if datatype == DataType.Character:
    #         name = value.get("Name", "")
    #         metadata = value.get("Metadata", [])
    #         if not name:
    #             raise ValueError("Character must have a Name.")
    #         text = name + (" " + " ".join(metadata) if metadata else "")
    #         emb = self._get_text_embedding(text)
    #     elif datatype == DataType.Location:
    #         name = value.get("Name", "")
    #         description = value.get("Description", "")
    #         if not name:
    #             raise ValueError("Location must have a Name.")
    #         text = name + " " + description
    #         emb = self._get_text_embedding(text)
    #     elif datatype == DataType.Plot:
    #         timestamp = value.get("Timestamp", "")
    #         location = value.get("Location", "")
    #         text_content = value.get("Text", "")
    #         related = value.get("Related Characters", [])
    #         if not text_content:
    #             raise ValueError("Plot event must have Text content.")
    #         text = f"{timestamp} {location} {text_content} " + " ".join(related)
    #         emb = self._get_text_embedding(text)
    #     else:
    #         raise ValueError(f"Unsupported data type: {datatype}")
        
    #     self.data.append((key, value, datatype, emb))

    def delete(self, key: str):
        new_data = [(k, v, dt, emb) for k, v, dt, emb in self.data if k != key]
        if len(new_data) == len(self.data):
            logging.warning(f"Key '{key}' not found.")
        else:
            self.data = new_data
            self.build_index()

    def build_index(self):
        self._init_index()
        if not self.data:
            return
        embeddings = [item[3] for item in self.data if item[3] is not None]
        all_emb = np.array(embeddings, dtype=np.float32)
        self.index.add(all_emb)

    def retrieve_similar(self, query_text, k=5):
        if not self.data:
            return []
        query_emb = self._get_text_embedding(query_text).reshape(1, -1)
        distances, indices = self.index.search(query_emb, k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.data):
                key, value, dt, _ = self.data[idx]
                results.append({
                    "key": key,
                    "value": value,
                    "distance": float(dist),
                    "data_type": dt,
                })
        return results

    def get(self, key: str):
        for k, value, dt, _ in self.data:
            if k == key:
                return {"key": k, "value": value, "data_type": dt}
        return None

    def save_data(self, data_path: str):
        with open(data_path, "wb") as f:
            pickle.dump(self.data, f)
        logging.info(f"Map data saved to {data_path}")

    def load_data(self, data_path: str):
        with open(data_path, "rb") as f:
            self.data = pickle.load(f)
        logging.info(f"Map data loaded from {data_path}")
