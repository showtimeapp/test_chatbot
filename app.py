from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
from query_handler import MultiDatasetQueryHandler
from rag_handler import MultiDatasetRAGHandler
from database_setup import setup_databases
import logging

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize handlers
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError("Please set OPENAI_API_KEY in .env file")

# Setup database if not exists
if not os.path.exists('data/data.db'):
    logger.info("Creating database from CSV files...")
    setup_databases()

# Initialize handlers
query_handler = MultiDatasetQueryHandler(OPENAI_API_KEY)
rag_handler = MultiDatasetRAGHandler()

@app.route('/api/chat', methods=['POST'])
def chat():
    """Main chat endpoint with multi-dataset support"""
    try:
        data = request.json
        user_query = data.get('query', '')
        force_dataset = data.get('dataset', None)  # Optional: force specific dataset
        
        if not user_query:
            return jsonify({
                'error': 'Query is required',
                'success': False
            }), 400
        
        logger.info(f"Received query: {user_query}")
        
        # Detect dataset and query type
        if force_dataset:
            dataset = force_dataset
            _, query_type = query_handler.detect_dataset_and_query_type(user_query)
        else:
            dataset, query_type = query_handler.detect_dataset_and_query_type(user_query)
        
        logger.info(f"Dataset: {dataset}, Query type: {query_type}")
        
        response = ""
        metadata = {
            'dataset': dataset,
            'query_type': query_type
        }
        
        if query_type == 'sql' and dataset in ['temple', 'school']:
            # Handle SQL queries
            sql_query = query_handler.text_to_sql(user_query, dataset)
            logger.info(f"Generated SQL: {sql_query}")
            metadata['sql_query'] = sql_query
            
            sql_result = query_handler.execute_sql(sql_query)
            
            # Check if SQL failed or returned no results
            if not sql_result['success'] or sql_result.get('row_count', 0) == 0:
                logger.info("SQL failed/no results, falling back to RAG")
                
                # Fallback to RAG
                search_results, used_dataset = rag_handler.search(user_query, dataset)
                response = query_handler.format_rag_response(user_query, search_results, used_dataset)
                
                metadata['fallback'] = 'rag'
                metadata['search_results'] = len(search_results)
                metadata['original_error'] = sql_result.get('error', 'No results found')
                
                if search_results:
                    response = f"{response}"
                else:
                    # Final fallback to general LLM
                    response = query_handler.handle_general_query(user_query)
                    metadata['fallback'] = 'general'
            else:
                # SQL succeeded
                response = query_handler.format_sql_response(user_query, sql_result, dataset)
            
        elif query_type == 'rag':
            # Handle RAG queries
            search_results, used_dataset = rag_handler.search(user_query, dataset)
            response = query_handler.format_rag_response(user_query, search_results, used_dataset)
            
            metadata['search_results'] = len(search_results)
            metadata['used_dataset'] = used_dataset
            
        else:
            # Handle general queries
            response = query_handler.handle_general_query(user_query)
        
        return jsonify({
            'success': True,
            'response': response,
            'metadata': metadata
        })
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    # Check which datasets are available
    available_datasets = []
    if os.path.exists('data/temple_data.csv'):
        available_datasets.append('temples')
    if os.path.exists('data/school_data.csv'):
        available_datasets.append('schools')
    
    return jsonify({
        'status': 'healthy',
        'service': 'Multi-Dataset Chatbot API',
        'available_datasets': available_datasets
    })

@app.route('/api/datasets', methods=['GET'])
def get_datasets():
    """Get information about available datasets"""
    datasets_info = []
    
    conn = sqlite3.connect('data/data.db')
    cursor = conn.cursor()
    
    # Get info for each table
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    
    for table in tables:
        table_name = table[0]
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = cursor.fetchone()[0]
        
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = [col[1] for col in cursor.fetchall()]
        
        datasets_info.append({
            'name': table_name,
            'record_count': count,
            'columns': columns
        })
    
    conn.close()
    
    return jsonify({
        'datasets': datasets_info
    })

if __name__ == '__main__':
    import sqlite3  # Import here for the datasets endpoint
    app.run(debug=True, port=5000)