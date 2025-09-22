import os
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple

class MultiDatasetRAGHandler:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.temple_handler = None
        self.school_handler = None
        
        # Initialize temple RAG if data exists
        if os.path.exists('data/temple_data.csv'):
            self.temple_handler = DatasetRAG(
                csv_path='data/temple_data.csv',
                embeddings_path='embeddings/temple_vectors.pkl',
                dataset_type='temple',
                model=self.model
            )
        
        # Initialize school RAG if data exists
        if os.path.exists('data/school_data.csv'):
            self.school_handler = DatasetRAG(
                csv_path='data/school_data.csv',
                embeddings_path='embeddings/school_vectors.pkl',
                dataset_type='school',
                model=self.model
            )
    
    def search(self, query: str, dataset: str = 'auto', top_k: int = 3) -> Tuple[List[Dict], str]:
        """Search appropriate dataset based on query or specified dataset"""
        
        if dataset == 'auto':
            dataset = self._detect_dataset(query)
        
        if dataset == 'temple' and self.temple_handler:
            return self.temple_handler.search(query, top_k), 'temple'
        elif dataset == 'school' and self.school_handler:
            return self.school_handler.search(query, top_k), 'school'
        elif dataset == 'both':
            results = []
            if self.temple_handler:
                results.extend(self.temple_handler.search(query, top_k // 2))
            if self.school_handler:
                results.extend(self.school_handler.search(query, top_k // 2))
            return results, 'both'
        else:
            return [], 'none'
    
    def _detect_dataset(self, query: str) -> str:
        """Detect which dataset to use based on query content"""
        query_lower = query.lower()
        
        # Updated to include all religious structures
        temple_keywords = ['temple', 'deity', 'worship', 'footfall', 'religious', 
                          'prayer', 'god', 'goddess', 'shrine', 'mandir', 'mosque',
                          'masjid', 'church', 'dargah', 'buddha', 'buddhist', 'gurudwara',
                          'synagogue', 'faith', 'devotee', 'pilgrimage', 'sacred',
                          'spiritual', 'worship place', 'religious structure']
        school_keywords = ['school', 'student', 'teacher', 'education', 'classroom',
                          'board', 'fees', 'principal', 'grade', 'curriculum',
                          'cbse', 'icse', 'study', 'academic','medium of instruction']
        
        temple_score = sum(1 for keyword in temple_keywords if keyword in query_lower)
        school_score = sum(1 for keyword in school_keywords if keyword in query_lower)
        
        if temple_score > school_score:
            return 'temple'
        elif school_score > temple_score:
            return 'school'
        else:
            return 'both'


class DatasetRAG:
    def __init__(self, csv_path: str, embeddings_path: str, dataset_type: str, model):
        self.model = model
        self.embeddings_path = embeddings_path
        self.dataset_type = dataset_type
        self.df = pd.read_csv(csv_path)
        self._clean_columns()
        self.documents = self._prepare_documents()
        self.embeddings = self._load_or_create_embeddings()
    
    def _clean_columns(self):
        """Clean and standardize column names"""
        import re
        new_columns = []
        for col in self.df.columns:
            # Convert to lowercase and replace spaces
            col = col.lower().replace(' ', '_')
            # Convert camelCase to snake_case
            col = re.sub('([A-Z]+)', r'_\1', col).lower()
            col = col.lstrip('_')
            new_columns.append(col)
        self.df.columns = new_columns
    
    def _prepare_documents(self) -> List[str]:
        """Prepare documents based on dataset type"""
        documents = []
        
        if self.dataset_type == 'temple':
            for _, row in self.df.iterrows():
                doc = f"""
                Religious Structure: {row.get('structurename', row.get('structure_name', 'N/A'))}
                Type: Religious Place (Temple/Mosque/Church/Dargah/Buddha Temple/Other)
                Sector: {row.get('sector', 'N/A')}
                Ward: {row.get('ward', 'N/A')}
                Deity/Faith: {row.get('deity', 'N/A')}
                Area: {row.get('areasqft', row.get('area_sq_ft', 'N/A'))} sq ft
                Footfall: {row.get('footfall', 'N/A')}
                Established: {row.get('dateofestablishment', row.get('date_of_establishment', 'N/A'))}
                Registration: {row.get('registration', 'N/A')}
                Remarks: {row.get('remarks', 'N/A')}
                """
                documents.append(doc)
                
        elif self.dataset_type == 'school':
            for _, row in self.df.iterrows():
                doc = f"""
                School: {row.get('school_name', row.get('schoolname', 'N/A'))}
                Address: {row.get('address', 'N/A')}
                Sector: {row.get('sector', 'N/A')}
                Ward: {row.get('ward', 'N/A')}
                Board: {row.get('board', 'N/A')}
                Medium: {row.get('medium_of_instruction', row.get('mediumofinstruction', 'N/A'))}
                Grade: {row.get('grade', 'N/A')}
                Average Fees: ₹{row.get('average_fees', row.get('averagefees', 'N/A'))}
                Students: {row.get('students', 'N/A')}
                Teachers: {row.get('teachers', 'N/A')}
                Classrooms: {row.get('classrooms', 'N/A')}
                Student-Teacher Ratio: {row.get('student_teacher_ratio', row.get('studentteacherratio', 'N/A'))}
                Principal: {row.get('principal', 'N/A')}
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
        """Search for relevant items using semantic similarity"""
        query_embedding = self.model.encode([query])
        
        # Calculate cosine similarity
        similarities = np.dot(self.embeddings, query_embedding.T).flatten()
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if self.dataset_type == 'temple':
                item_name = self.df.iloc[idx].get('structurename', 
                           self.df.iloc[idx].get('structure_name', 'Unknown'))
                results.append({
                    'type': 'religious_structure',
                    'name': item_name,
                    'sector': self.df.iloc[idx].get('sector', 'N/A'),
                    'deity_faith': self.df.iloc[idx].get('deity', 'N/A'),
                    'similarity': float(similarities[idx]),
                    'full_info': self.documents[idx]
                })
            elif self.dataset_type == 'school':
                item_name = self.df.iloc[idx].get('school_name', 
                           self.df.iloc[idx].get('schoolname', 'Unknown'))
                results.append({
                    'type': 'school',
                    'name': item_name,
                    'sector': self.df.iloc[idx].get('sector', 'N/A'),
                    'board': self.df.iloc[idx].get('board', 'N/A'),
                    'similarity': float(similarities[idx]),
                    'full_info': self.documents[idx]
                })
        
        return results