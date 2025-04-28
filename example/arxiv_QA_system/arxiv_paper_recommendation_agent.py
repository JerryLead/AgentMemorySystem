import time
import requests
import xml.etree.ElementTree as ET
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


class SemanticMapForPapers:
    def __init__(self, text_embedding_model="all-MiniLM-L6-v2", embedding_dim=384):
        """
        Initialize SemanticMap for academic papers.
        """
        self.text_model = SentenceTransformer(text_embedding_model)
        self.embedding_dim = embedding_dim

        self.data = []
        self.preferences = {"liked": [], "disliked": []}  # Track user preferences
        self.text_index = faiss.IndexFlatL2(embedding_dim)

    def _get_text_embedding(self, text):
        """
        Generate embedding for a given text.
        """
        return self.text_model.encode([text], convert_to_numpy=True)[0]

    def insert(self, paper_id, title, abstract, authors, categories, pdf_url, timestamp):
        """
        Insert a new arXiv paper into the SemanticMap.
        """
        entry = {
            "paper_id": paper_id,
            "title": title,
            "abstract": abstract,
            "authors": authors,
            "categories": categories,
            "pdf_url": pdf_url,
            "timestamp": timestamp,
        }

        # Generate text embedding for title and abstract
        embedding = self._get_text_embedding(title + " " + abstract)
        entry["embedding"] = embedding
        self.text_index.add(np.array([embedding]))
        self.data.append(entry)

    def query(self, query_text, k=5):
        """
        Query the SemanticMap based on a text query.
        """
        query_embedding = self._get_text_embedding(query_text)
        result = self.text_index.search(np.array([query_embedding]), k)
        print(f"search method returned {len(result)} values")
        if len(result)!= 2:
            raise ValueError("search method returned an unexpected number of values")
        distances, indices = result
        return [self.data[i] for i in indices[0] if i < len(self.data)]

    def delete(self, query_text, semantic=True):
        """
        Delete papers that match the query text.
        """
        query_embedding = self._get_text_embedding(query_text)
        distances, indices = self.text_index.search(np.array([query_embedding]), len(self.data))
        to_delete = [self.data[i] for i in indices[0] if distances[0][i] < 0.7] if semantic else []
        for item in to_delete:
            self.preferences["disliked"].append(item)  # Track in preferences
            self.data.remove(item)
        return to_delete

    def mark_as_liked(self, paper_id):
        """
        Mark a paper as liked by the user.
        """
        for item in self.data:
            if item["paper_id"] == paper_id:
                self.preferences["liked"].append(item)
                break

    def recommend(self, query_text, k=5):
        """
        Recommend papers based on query and user preferences.
        """
        # Call query function to retrieve initial results
        results = self.query(query_text, k)

        # Process results with user preferences
        recommendations = []
        for item in results:
            if item in self.preferences["liked"]:  # Boost liked preferences
                recommendations.insert(0, item)
            elif item in self.preferences["disliked"]:  # Exclude disliked papers
                continue
            else:
                recommendations.append(item)
        return recommendations[:k]

    def list_all(self):
        """
        List all papers in the SemanticMap.
        """
        return self.data


class ArxivAgent:
    def __init__(self, query="cs.AI", max_results=10, text_embedding_model="./all-MiniLM-L6-v2", embedding_dim=384):
        self.query = query
        self.max_results = max_results
        self.semantic_map = SemanticMapForPapers(text_embedding_model, embedding_dim)

    def fetch_arxiv_data(self):
        """
        Fetch arXiv papers based on query and max results.
        """
        url = f"http://export.arxiv.org/api/query?search_query=cat:{self.query}&start=0&max_results={self.max_results}"
        response = requests.get(url)
        response.raise_for_status()
        feed = response.text
        return self.parse_arxiv_feed(feed)

    def parse_arxiv_feed(self, feed):
        """
        Parse the arXiv RSS feed.
        """
        root = ET.fromstring(feed)
        papers = []
        for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
            paper_id = entry.find("{http://www.w3.org/2005/Atom}id").text
            title = entry.find("{http://www.w3.org/2005/Atom}title").text.strip()
            abstract = entry.find("{http://www.w3.org/2005/Atom}summary").text.strip()
            authors = [author.find("{http://www.w3.org/2005/Atom}name").text for author in entry.findall("{http://www.w3.org/2005/Atom}author")]
            categories = entry.find("{http://arxiv.org/schemas/atom}primary_category").attrib["term"]
            timestamp = entry.find("{http://www.w3.org/2005/Atom}published").text
            pdf_url = entry.find("{http://www.w3.org/2005/Atom}link[@type='application/pdf']").attrib["href"]
            papers.append({
                "paper_id": paper_id,
                "title": title,
                "abstract": abstract,
                "authors": authors,
                "categories": categories,
                "pdf_url": pdf_url,
                "timestamp": timestamp,
            })
        return papers

    def update(self):
        """
        Update the SemanticMap with new arXiv papers.
        """
        papers = self.fetch_arxiv_data()
        for paper in papers:
            self.semantic_map.insert(
                paper_id=paper["paper_id"],
                title=paper["title"],
                abstract=paper["abstract"],
                authors=paper["authors"],
                categories=paper["categories"],
                pdf_url=paper["pdf_url"],
                timestamp=paper["timestamp"]
            )

    def real_time_update_and_recommend(self, query_text, k=5, interval=60):
        """
        Continuously update the SemanticMap and recommend papers at a specified interval.
        """
        while True:
            self.update()
            recommendations = self.semantic_map.recommend(query_text, k)
            print(f"Recommended papers for '{query_text}' at {time.ctime()}:")
            for paper in recommendations:
                print(paper)
            time.sleep(interval)


if __name__ == "__main__":
    agent = ArxivAgent()
    query_text = "AI research"
    agent.real_time_update_and_recommend(query_text, k=3, interval=60)
