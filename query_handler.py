import sqlite3
import openai
from typing import Dict, Any, List, Tuple
import re

class MultiDatasetQueryHandler:
    def __init__(self, openai_api_key: str, db_path: str = 'data/data.db'):
        self.openai_client = openai.OpenAI(api_key=openai_api_key)
        self.db_path = db_path
        self.table_schemas = self._get_all_schemas()
    
    def _get_all_schemas(self) -> Dict[str, str]:
        """Get schemas for all tables in the database"""
        schemas = {}
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        
        for table in tables:
            table_name = table[0]
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            
            schema = f"Table: {table_name}\nColumns:\n"
            for col in columns:
                schema += f"- {col[1]} ({col[2]})\n"
            schemas[table_name] = schema
        
        conn.close()
        return schemas
    
    def detect_dataset_and_query_type(self, query: str) -> Tuple[str, str]:
        """Detect which dataset to use and classify query type"""
        query_lower = query.lower()
        
        # Dataset detection - Updated to include all religious structures
        temple_keywords = ['temple', 'deity', 'worship', 'footfall', 'religious', 
                          'prayer', 'god', 'goddess', 'shrine', 'mandir', 'mosque',
                          'masjid', 'church', 'dargah', 'buddha', 'buddhist', 'gurudwara',
                          'synagogue', 'faith', 'devotee', 'pilgrimage', 'sacred']
        school_keywords = ['school', 'student', 'teacher', 'education', 'classroom',
                          'board', 'fees', 'principal', 'grade', 'curriculum',
                          'cbse', 'icse', 'study', 'academic', 'learning','medium of instruction']
        
        temple_score = sum(1 for keyword in temple_keywords if keyword in query_lower)
        school_score = sum(1 for keyword in school_keywords if keyword in query_lower)
        
        if temple_score > school_score:
            dataset = 'temple'
        elif school_score > temple_score:
            dataset = 'school'
        else:
            dataset = 'general'
        
        # Query type detection
        sql_keywords = ['how many', 'count', 'average', 'sum', 'maximum', 'minimum', 
                       'list all', 'show all', 'total', 'highest', 'lowest', 
                       'greater than', 'less than', 'between','oldest','top','newest','which temple',
                       'which school','which sector']
        
        rag_keywords = ["tell me about", "information about", "details of",
                        "what do you know about", "describe", "explain about",
                        "explain", "elaborate", "in detail", "overview", "summary",
                        "background", "context", "insights", "highlights", "key facts"]
        
        # Check for SQL-type queries
        if any(keyword in query_lower for keyword in sql_keywords):
            query_type = 'sql'
        # Check for RAG-type queries
        elif any(keyword in query_lower for keyword in rag_keywords):
            query_type = 'rag'
        # Check if query mentions specific data terms
        elif dataset != 'general':
            query_type = 'sql'  # Default to SQL for dataset-specific queries
        else:
            query_type = 'general'
        
        return dataset, query_type
    
    def text_to_sql(self, query: str, dataset: str) -> str:
        """Convert natural language to SQL query for specific dataset"""
        
        # Determine table name (keeping 'temples' for backward compatibility)
        table_name = 'temples' if dataset == 'temple' else 'schools'
        schema = self.table_schemas.get(table_name, "")
        
        # Add context about religious diversity for temple queries
        context = ""
        if dataset == 'temple':
            context = """
            Note: The 'temples' table contains various religious structures including:
            - Hindu temples (mandir)
            - Mosques (masjid)
            - Churches
            - Dargahs
            - Buddhist temples
            - Other religious structures
            The 'deity' column may contain deity names, or religious figures/faiths.
            """
        
        prompt = f"""
        Convert the following natural language query to a SQL query for a SQLite database.
        
        Database Schema:
        {schema}
        {context}
        
        Natural Language Query: {query}
        
        Rules:
        - Use table name '{table_name}'
        - Column names are lowercase with underscores
        - Return ONLY the SQL query, nothing else
        - For text comparisons, use LIKE with % wildcards
        - Limit results to 20 unless specified otherwise
        - Be inclusive of all religious structures when querying the temples table
        
        SQL Query:
        """
        
        response = self.openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a SQL expert. Return only valid SQL queries."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        
        sql_query = response.choices[0].message.content.strip()
        sql_query = sql_query.replace('```sql', '').replace('```', '').strip()
        
        return sql_query
    
    def execute_sql(self, sql_query: str) -> Dict[str, Any]:
        """Execute SQL query and return results"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(sql_query)
            
            columns = [description[0] for description in cursor.description] if cursor.description else []
            results = cursor.fetchall()
            conn.close()
            
            return {
                'success': True,
                'columns': columns,
                'data': results,
                'row_count': len(results)
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def format_sql_response(self, query: str, sql_result: Dict, dataset: str) -> str:
        """Format SQL results into natural language"""
        if not sql_result['success'] or sql_result['row_count'] == 0:
            return None  # Signal for fallback
        
        data_str = f"Columns: {sql_result['columns']}\n"
        data_str += f"Data (showing first 10 rows):\n"
        for row in sql_result['data'][:10]:
            data_str += f"{row}\n"
        
        # Updated context to be more inclusive
        if dataset == 'temple':
            context = "religious structures data (including temples, mosques, churches, dargahs, etc.)"
        else:
            context = "school data"
        
        prompt = f"""
        Convert the following SQL query results about {context} into a natural, conversational response.
        
        User Query: {query}
        
        Results:
        {data_str}
        Total rows: {sql_result['row_count']}
        
        Provide a clear, informative response that directly answers the user's question.
        Format numbers nicely and be concise but complete.
        Be inclusive and respectful of all religious faiths when discussing religious structures.
        """
        
        response = self.openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": f"You are a helpful assistant explaining {context} clearly."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        
        return response.choices[0].message.content
    
    def format_rag_response(self, query: str, search_results: List[Dict], dataset: str) -> str:
        """Format RAG search results using LLM"""
        if not search_results:
            return f"I couldn't find specific information about that in the {dataset} database."
        
        context = "Here is the relevant information from the database:\n\n"
        for i, result in enumerate(search_results, 1):
            context += f"Item {i} ({result.get('type', 'unknown')}):\n"
            context += f"{result['full_info']}\n"
            context += f"Relevance Score: {result['similarity']:.2f}\n\n"
        
        prompt = f"""
        Based on the following information from our database, please provide a comprehensive 
        and helpful answer to the user's question.
        
        User Question: {query}
        
        Database Information:
        {context}
        
        Please provide a natural, conversational response that:
        1. Directly answers the user's question
        2. Includes specific details from the data
        3. Mentions relevant items by name when appropriate
        4. Provides additional context if helpful
        5. Is well-formatted and easy to read
        
        Response:
        """
        
        response = self.openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a knowledgeable assistant helping users find information. Provide detailed, accurate responses based on the given data."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        return response.choices[0].message.content
    
    def handle_general_query(self, query: str) -> str:
        """Handle general queries using OpenAI"""
        response = self.openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": query}
            ],
            temperature=0.7
        )
        
        return response.choices[0].message.content