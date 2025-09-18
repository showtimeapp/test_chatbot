import sqlite3
import openai
from typing import Dict, Any, List
import re
import json

class QueryHandler:
    def __init__(self, openai_api_key: str, db_path: str = 'data/temple.db'):
        self.openai_client = openai.OpenAI(api_key=openai_api_key)
        self.db_path = db_path
        self.table_schema = self._get_table_schema()
    
    def _get_table_schema(self) -> str:
        """Get the database schema for the LLM context"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(temples)")
        columns = cursor.fetchall()
        conn.close()
        
        schema = "Table: temples\nColumns:\n"
        for col in columns:
            schema += f"- {col[1]} ({col[2]})\n"
        return schema
    
    def classify_query(self, query: str) -> str:
        """Classify the type of query: sql, rag, or general"""
        sql_keywords = ['how many', 'count', 'average', 'sum', 'maximum', 'minimum', 
                       'list all', 'show all', 'total', 'highest', 'lowest', 
                       'which temple', 'which sector', 'footfall greater than']
        
        rag_keywords = ['tell me about', 'information about', 'details of', 
                       'what do you know about', 'describe', 'explain about']
        
        query_lower = query.lower()
        
        # Check for SQL-type queries
        if any(keyword in query_lower for keyword in sql_keywords):
            return 'sql'
        
        # Check for RAG-type queries
        if any(keyword in query_lower for keyword in rag_keywords):
            return 'rag'
        
        # Check if query mentions temple-specific terms
        temple_terms = ['temple', 'deity', 'footfall', 'sector', 'ward', 'registration']
        if any(term in query_lower for term in temple_terms):
            return 'sql'
        
        # Default to general LLM query
        return 'general'
    
    def text_to_sql(self, query: str) -> str:
        """Convert natural language to SQL query using OpenAI"""
        prompt = f"""
        Convert the following natural language query to a SQL query for a SQLite database.
        
        Database Schema:
        {self.table_schema}
        
        Natural Language Query: {query}
        
        Rules:
        - Use table name 'temples'
        - Column names are lowercase with underscores
        - Return ONLY the SQL query, nothing else
        - For text comparisons, use LIKE with % wildcards
        - Limit results to 20 unless specified otherwise
        
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
        # Clean the SQL query
        sql_query = sql_query.replace('```sql', '').replace('```', '').strip()
        
        return sql_query
    
    def execute_sql(self, sql_query: str) -> Dict[str, Any]:
        """Execute SQL query and return results"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(sql_query)
            
            # Get column names
            columns = [description[0] for description in cursor.description] if cursor.description else []
            
            # Fetch results
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
    
    def format_sql_response(self, query: str, sql_result: Dict) -> str:
        """Format SQL results into natural language using OpenAI"""
        if not sql_result['success']:
            return f"I encountered an error while processing your query: {sql_result['error']}"
        
        if sql_result['row_count'] == 0:
            return "No results found for your query."
        
        # Prepare data for formatting
        data_str = f"Columns: {sql_result['columns']}\n"
        data_str += f"Data (showing first 10 rows):\n"
        for row in sql_result['data'][:10]:
            data_str += f"{row}\n"
        
        prompt = f"""
        Convert the following SQL query results into a natural, conversational response.
        
        User Query: {query}
        
        Results:
        {data_str}
        Total rows: {sql_result['row_count']}
        
        Provide a clear, informative response that directly answers the user's question.
        Format numbers nicely and be concise but complete.
        """
        
        response = self.openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that explains data results clearly."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
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
    
    def format_rag_response(self, query: str, search_results: List[Dict]) -> str:
        """Format RAG search results using LLM for natural response"""
        if not search_results:
            return "I couldn't find specific information about that in the temple database."
        
        # Prepare context from search results
        context = "Here is the relevant information from the temple database:\n\n"
        for i, result in enumerate(search_results, 1):
            context += f"Temple {i}:\n{result['full_info']}\n"
            context += f"Relevance Score: {result['similarity']:.2f}\n\n"
        
        # Create prompt for LLM
        prompt = f"""
        Based on the following temple information from our database, please provide a comprehensive 
        and helpful answer to the user's question. Use the information provided to give specific 
        details, and if relevant, mention multiple temples that match their query.
        
        User Question: {query}
        
        Temple Database Information:
        {context}
        
        Please provide a natural, conversational response that:
        1. Directly answers the user's question
        2. Includes specific details from the temple data
        3. Mentions relevant temples by name when appropriate
        4. Provides additional context if helpful
        5. Is well-formatted and easy to read
        
        Response:
        """
        
        response = self.openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a knowledgeable assistant helping users find information about temples. Provide detailed, accurate responses based on the given temple data."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        return response.choices[0].message.content