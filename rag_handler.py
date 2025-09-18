import os
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
from typing import List, Dict

class RAGHandler:
    def __init__(self, csv_path='data/temple_data.csv'):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings_path = 'embeddings/vector_store.pkl'
        self.df = pd.read_csv(csv_path)
        self.df.columns = [col.lower().replace(' ', '_') for col in self.df.columns]
        self.documents = self._prepare_documents()
        self.embeddings = self._load_or_create_embeddings()
    
    def _prepare_documents(self) -> List[str]:
        """Prepare documents from temple data for embedding"""
        documents = []
        for _, row in self.df.iterrows():
            doc = f"""
            Temple: {row['structurename']}
            Sector: {row['sector']}
            Ward: {row['ward']}
            Deity: {row['deity']}
            Area: {row['areasqft']} sq ft
            Footfall: {row['footfall']}
            Established: {row['dateofestablishment']}
            Registration: {row['registration']}
            Relevance: {row['relevance']}
            Remarks: {row['remarks']}
            Location: ({row['latitude']}, {row['longitude']})
            """
            documents.append(doc)
        return documents
    
    def _load_or_create_embeddings(self):
        """Load existing embeddings or create new ones"""
        os.makedirs('embeddings', exist_ok=True)
        
        if os.path.exists(self.embeddings_path):
            with open(self.embeddings_path, 'rb') as f:
                embeddings = pickle.load(f)
        else:
            embeddings = self.model.encode(self.documents)
            with open(self.embeddings_path, 'wb') as f:
                pickle.dump(embeddings, f)
        
        return embeddings
    
    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """Search for relevant temples using semantic similarity"""
        query_embedding = self.model.encode([query])
        
        # Calculate cosine similarity
        similarities = np.dot(self.embeddings, query_embedding.T).flatten()
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append({
                'temple': self.df.iloc[idx]['structurename'],
                'sector': self.df.iloc[idx]['sector'],
                'deity': self.df.iloc[idx]['deity'],
                'footfall': self.df.iloc[idx]['footfall'],
                'similarity': float(similarities[idx]),
                'full_info': self.documents[idx]
            })
        
        return results