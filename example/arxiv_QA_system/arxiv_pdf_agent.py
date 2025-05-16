# 元数据信息: title,authors,abstract,keywords,chapter,paragraph,photo,reference
# 将title&authors作为根节点,其他元数据各自为一个节点插入,chapter,paragraph,photo,reference有多个,各自为一个节点
# paragraph和photo是chapter的子节点,reference是其他paper的根节点,与该paper的根节点是cite关系,其余元数据是根节点的子节点
# 实现query(根据query_text查询相关的5篇论文),delete(根据query_text删除最相关的一篇Paper),list_all(列出所有根节点)
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import requests
import datetime
from semantic_data_structure.semantic_simple_graph import SemanticSimpleGraph  
from semantic_data_structure.semantic_map import SemanticMap
import time



class ArxivSemanticGraph:
    def __init__(self, text_embedding_model="all-MiniLM-L6-v2", embedding_dim=384):
        self.semantic_graph = SemanticSimpleGraph(SemanticMap(text_embedding_model, embedding_dim))
        self.preferences = {"liked": [], "disliked": []}

    def _get_text_embedding(self, text):
        return self.semantic_graph.semantic_map.text_encoder.encode([text], convert_to_numpy=True)

    def insert(self, paper_id, title, authors, abstract, keywords, chapters, paragraphs, photos, references):
        # 将title和authors作为一个根节点插入
        root_key = f"{paper_id}_title_authors"
        root_value = {
            "title": title,
            "authors": authors
        }
        self.semantic_graph.add_node(root_key, root_value, text_for_embedding=f"{title} {' '.join(authors)}")

        # 插入其他元数据节点
        abstract_key = f"{paper_id}_abstract"
        self.semantic_graph.add_node(abstract_key, abstract, parent_keys=[root_key], parent_relation="has_abstract", text_for_embedding=abstract)

        keywords_key = f"{paper_id}_keywords"
        self.semantic_graph.add_node(keywords_key, keywords, parent_keys=[root_key], parent_relation="has_keywords", text_for_embedding=keywords)

        chapter_key_map = {}
        for idx, chapter in enumerate(chapters):
            chapter_key = f"{paper_id}_chapter_{idx}"
            self.semantic_graph.add_node(chapter_key, chapter, parent_keys=[root_key], parent_relation="has_chapter", text_for_embedding=chapter)
            chapter_key_map[idx] = chapter_key

        for para in paragraphs:
            chapter_idx = para['chapter_idx']
            para_key = f"{paper_id}_chapter_{chapter_idx}_paragraph_{para['para_idx']}"
            self.semantic_graph.add_node(para_key, para['content'], parent_keys=[chapter_key_map[chapter_idx]], parent_relation="has_paragraph", text_for_embedding=para['content'])

        for photo in photos:
            chapter_idx = photo['chapter_idx']
            photo_key = f"{paper_id}_chapter_{chapter_idx}_photo_{photo['photo_idx']}"
            self.semantic_graph.add_node(photo_key, photo['content'], parent_keys=[chapter_key_map[chapter_idx]], parent_relation="has_photo", text_for_embedding=photo['content'])

        for ref_idx, ref_paper_id in enumerate(references):
            ref_root_key = f"{ref_paper_id}_title_authors"
            self.semantic_graph.add_node(ref_root_key, {"title": "", "authors": []}, text_for_embedding="")
            self.semantic_graph.insert_edge(root_key, ref_root_key, relation="cite")

    def query(self, query_text, k=5):
        """
        返回查询节点及其根节点
        """
        similar_nodes = self.semantic_graph.retrieve_similar_nodes(query_text, k)
        result_nodes = []
        for node in similar_nodes:
            key = node["key"]
            if key.endswith("_title_authors"):
                result_nodes.append(node)
            else:
                current_key = key
                while "parents" in self.semantic_graph.graph_relations[current_key] and self.semantic_graph.graph_relations[current_key]["parents"]:
                    current_key = list(self.semantic_graph.graph_relations[current_key]["parents"].keys())[0]
                root_node = self._get_root_node(current_key)
                result_nodes.append(node)
                result_nodes.append(root_node)
        return result_nodes

    def _get_root_node(self, key):
        root_value = self.semantic_graph.semantic_map.data[self.semantic_graph.semantic_map.data.index(next((item for item in self.semantic_graph.semantic_map.data if item[0] == key), None))][1]
        return {
            "key": key,
            "value": root_value
        }

    def delete(self, query_text):
        """
         如果查询结果是根节点,则删除,否则不删除
        """
        similar_nodes = self.semantic_graph.retrieve_similar_nodes(query_text, 1)
        if not similar_nodes:
            return None
        key = similar_nodes[0]["key"]
        if key.endswith("_title_authors"):
            try:
                item_to_delete = next((item for item in self.semantic_graph.semantic_map.data if item[0] == key), None)
                if item_to_delete:
                    self.preferences["disliked"].append(item_to_delete)
                    self.semantic_graph.delete_node(key)
                    return similar_nodes[0]
            except ValueError:
                pass
        return None

    def mark_as_liked(self, root_key):
        """
        只能标记根节点
        """
        for item in self.semantic_graph.semantic_map.data:
            if item[0] == root_key:
                self.preferences["liked"].append(item)
                break

    def recommend(self, query_text, k=5):
        query_results = self.query(query_text, len(self.semantic_graph.semantic_map.data))
        root_nodes = set()
        for node in query_results:
            if node["key"].endswith("_title_authors"):
                root_nodes.add(node["key"])

        liked_nodes = []
        other_nodes = []
        for root_key in root_nodes:
            if root_key in [liked[0] for liked in self.preferences["liked"]]:
                liked_node = next((item for item in query_results if item["key"] == root_key), None)
                if liked_node:
                    liked_nodes.append(liked_node)
            elif root_key not in [disliked[0] for disliked in self.preferences["disliked"]]:
                other_node = next((item for item in query_results if item["key"] == root_key), None)
                if other_node:
                    other_nodes.append(other_node)

        recommended_nodes = liked_nodes + other_nodes
        return recommended_nodes[:k]


    def list_all(self):
        """
        列举所有根节点
        """
        all_nodes = self.semantic_graph.semantic_map.data
        root_nodes = []
        for item in all_nodes:
            key = item[0]
            if key.endswith("_title_authors"):
                root_nodes.append({
                    "key": key,
                    "value": item[1]
                })
        return root_nodes
    

# 创建 ArxivSemanticGraph 实例
arxiv_graph = ArxivSemanticGraph()

# 插入一篇论文的数据
arxiv_graph.insert(
    paper_id="paper1",
    title="My Paper",
    authors=["Author 1", "Author 2"],
    pages=["Page 1", "Page 2"],
    chunks=[["Chunk 1.1", "Chunk 1.2"], ["Chunk 2.1", "Chunk 2.2"]],
    photos=["Photo 1", "Photo 2"],
    references=["ref_paper1", "ref_paper2"]
)

# 查询相关节点
query_results = arxiv_graph.query("interesting topic", k=5)
print("Query results:", query_results)

# 删除最相关的一篇论文
arxiv_graph.delete("interesting topic")

# 列出所有根节点
all_roots = arxiv_graph.list_all()
print("All root nodes:", all_roots)

# import arxiv
# import requests
# import os
# from pdfminer.high_level import extract_text


# def fetch_arxiv_pdfs(query, num_results=5, output_dir='arxiv_pdfs'):
#     """
#     在Arxiv数据集上根据查询词爬取相关论文，并以PDF格式保存。

#     参数:
#     query -- 搜索查询词
#     num_results -- 要获取的论文数量，默认为5
#     output_dir -- 保存PDF文件的目录，默认为'arxiv_pdfs'

#     返回:
#     保存的PDF文件路径列表
#     """
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#     client = arxiv.Client()
#     search = arxiv.Search(
#         query=query,
#         max_results=num_results,
#         sort_by=arxiv.SortCriterion.Relevance
#     )
#     pdf_paths = []
#     for result in client.results(search):
#         pdf_url = result.pdf_url
#         file_name = os.path.join(output_dir, result.entry_id.split('/')[-1] + '.pdf')
#         response = requests.get(pdf_url)
#         with open(file_name, 'wb') as f:
#             f.write(response.content)
#         pdf_paths.append(file_name)
#     return pdf_paths



# import re
# from pdfminer.high_level import extract_text


# def decode_arxiv_pdf(pdf_path):
#     """
#     逻辑比较简单,按行识别
#     """
#     text = extract_text(pdf_path)
#     paper_info = {
#         'paper_id': os.path.basename(pdf_path).split('.')[0],
#         'title': '',
#         'authors': [],
#         'abstract': '',
#         'keywords': '',
#         'chapters': [],
#         'paragraphs': [],
#         'photos': [],
#        'references': []
#     }
#     lines = text.split('\n')
#     title_found = False
#     author_found = False
#     abstract_start = None
#     keyword_start = None
#     chapter_start = None
#     para_count = 0
#     for i, line in enumerate(lines):
#         line = line.strip()
#         if not title_found and line:
#             paper_info['title'] = line
#             title_found = True
#             continue
#         if not author_found and line:
#             paper_info['authors'] = line.split(', ')
#             author_found = True
#             continue
#         if 'Abstract' in line:
#             abstract_start = i + 1
#             continue
#         if 'Keywords' in line:
#             keyword_start = i + 1
#             continue
#         if 'Chapter' in line:
#             chapter_idx = len(paper_info['chapters'])
#             paper_info['chapters'].append(line)
#             chapter_start = i
#             if 'Abstract' in text and chapter_start < abstract_start:
#                 continue
#             if 'Keywords' in text and chapter_start < keyword_start:
#                 continue
#         if chapter_start is not None and line:
#             para_dict = {
#                 'chapter_idx': len(paper_info['chapters']) - 1,
#                 'para_idx': para_count,
#                 'content': line
#             }
#             paper_info['paragraphs'].append(para_dict)
#             para_count += 1
#     if abstract_start is not None:
#         abstract_lines = []
#         for line in lines[abstract_start:]:
#             if 'Keywords' in line or 'Chapter' in line:
#                 break
#             if line.strip():
#                 abstract_lines.append(line)
#         paper_info['abstract'] = '\n'.join(abstract_lines)
#     if keyword_start is not None:
#         keyword_lines = []
#         for line in lines[keyword_start:]:
#             if 'Chapter' in line:
#                 break
#             if line.strip():
#                 keyword_lines.append(line)
#         paper_info['keywords'] = ', '.join(keyword_lines)
#     return paper_info





# # if __name__ == "__main__":
# #     query = "machine learning"
# #     pdf_paths = fetch_arxiv_pdfs(query, num_results = 2)
# #     for path in pdf_paths:
# #         paper_info = decode_arxiv_pdf(path)
# #         print(paper_info)



# class RealTimeArxivSemanticGraph(ArxivSemanticGraph):
#     def real_time_recommend(self, query, num_results=5):
#         pdf_paths = fetch_arxiv_pdfs(query, num_results)
#         for path in pdf_paths:
#             paper_info = decode_arxiv_pdf(path)
#             self.insert(
#                 paper_id=paper_info['paper_id'],
#                 title=paper_info['title'],
#                 authors=paper_info['authors'],
#                 abstract=paper_info['abstract'],
#                 keywords=paper_info['keywords'],
#                 chapters=paper_info['chapters'],
#                 paragraphs=paper_info['paragraphs'],
#                 photos=paper_info['photos'],
#                 references=paper_info['references']
#             )
#         self.semantic_graph.build_index()
#         return self.recommend(query, num_results)
    


# if __name__ == "__main__":
#     real_time_graph = RealTimeArxivSemanticGraph()

#     # 测试 real_time_recommend 方法
#     print("Testing real_time_recommend method:")
#     recommendations = real_time_graph.real_time_recommend(query="artificial intelligence", num_results=3)
#     for idx, recommendation in enumerate(recommendations, 1):
#         print(f"推荐 {idx}:")
#         key = recommendation.get('key')
#         value = recommendation.get('value')
#         if key and value:
#             print(f"  节点Key: {key}")
#             if 'title' in value:
#                 print(f"  标题: {value['title']}")
#             if 'authors' in value:
#                 print(f"  作者: {', '.join(value['authors'])}")
#         else:
#             print("  未找到相关推荐信息")

#     # 测试 query 方法
#     print("\nTesting query method:")
#     query_results = real_time_graph.query(query_text="deep learning", k=3)
#     for idx, result in enumerate(query_results, 1):
#         print(f"Query Result {idx}: {result}")

#     # 测试 mark_as_liked 方法
#     print("\nTesting mark_as_liked method:")
#     if recommendations:
#         root_key = recommendations[2]['key']
#         real_time_graph.mark_as_liked(root_key)
#         print(f"Marked {root_key} as liked.")


#     # 测试 delete 方法
#     print("\nTesting delete method:")
#     delete_query = "Impact of Artificial Intelligence on Economic Theory"
#     delete_result = real_time_graph.delete(delete_query)
#     if delete_result:
#         print(f"Deleted paper: {delete_result}")
#     else:
#         print("No paper found to delete.")

#     # 测试 list_all 方法
#     print("\nTesting list_all method:")
#     all_papers = real_time_graph.list_all()
#     for idx, paper in enumerate(all_papers, 1):
#         print(f"Paper {idx}: {paper}")



# #problems:
# # 1. pdf_decoder不能正确识别paper的元数据
# # 2. paper的元数据是固定的,abstract,chapter等,不支持根据pdf动态扩展