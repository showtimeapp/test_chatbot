# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import os
# from dotenv import load_dotenv
# from query_handler import QueryHandler
# from rag_handler import RAGHandler
# from database_setup import csv_to_sqlite
# import logging

# # Load environment variables
# load_dotenv()

# # Initialize Flask app
# app = Flask(__name__)
# CORS(app)

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Initialize handlers
# OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
# if not OPENAI_API_KEY:
#     raise ValueError("Please set OPENAI_API_KEY in .env file")

# # Setup database if not exists
# if not os.path.exists('data/temple.db'):
#     logger.info("Creating database from CSV...")
#     csv_to_sqlite()

# # Initialize handlers
# query_handler = QueryHandler(OPENAI_API_KEY)
# rag_handler = RAGHandler()

# @app.route('/api/chat', methods=['POST'])
# def chat():
#     """Main chat endpoint"""
#     try:
#         data = request.json
#         user_query = data.get('query', '')
        
#         if not user_query:
#             return jsonify({
#                 'error': 'Query is required',
#                 'success': False
#             }), 400
        
#         logger.info(f"Received query: {user_query}")
        
#         # Classify query type
#         query_type = query_handler.classify_query(user_query)
#         logger.info(f"Query classified as: {query_type}")
        
#         response = ""
#         metadata = {'query_type': query_type}
        
#         if query_type == 'sql':
#             # Handle SQL queries
#             sql_query = query_handler.text_to_sql(user_query)
#             logger.info(f"Generated SQL: {sql_query}")
#             metadata['sql_query'] = sql_query
            
#             sql_result = query_handler.execute_sql(sql_query)
#             response = query_handler.format_sql_response(user_query, sql_result)
            
#         elif query_type == 'rag':
#             # Handle RAG queries
#             search_results = rag_handler.search(user_query, top_k=3)
            
#             # Pass RAG results through LLM for natural response
#             response = query_handler.format_rag_response(user_query, search_results)
            
#             metadata['search_results'] = len(search_results)
#             metadata['used_rag'] = True
            
#         else:
#             # Handle general queries
#             response = query_handler.handle_general_query(user_query)
        
#         return jsonify({
#             'success': True,
#             'response': response,
#             'metadata': metadata
#         })
        
#     except Exception as e:
#         logger.error(f"Error processing query: {str(e)}")
#         return jsonify({
#             'success': False,
#             'error': str(e)
#         }), 500

# @app.route('/api/health', methods=['GET'])
# def health():
#     """Health check endpoint"""
#     return jsonify({
#         'status': 'healthy',
#         'service': 'Temple Chatbot API'
#     })

# if __name__ == '__main__':
#     app.run(debug=True, port=5000)

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
from query_handler import QueryHandler
from rag_handler import RAGHandler
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

query_handler = QueryHandler(OPENAI_API_KEY)
rag_handler = RAGHandler()

@app.route('/api/chat', methods=['POST'])
def chat():
    """Main chat endpoint"""
    try:
        data = request.json
        user_query = data.get('query', '')

        if not user_query:
            return jsonify({
                'error': 'Query is required',
                'success': False
            }), 400

        logger.info(f"Received query: {user_query}")

        # Classify query type
        query_type = query_handler.classify_query(user_query)
        logger.info(f"Query classified as: {query_type}")

        response = ""
        metadata = {'query_type': query_type}

        if query_type == 'sql':
            # Handle SQL queries
            sql_query = query_handler.text_to_sql(user_query)
            logger.info(f"Generated SQL: {sql_query}")
            metadata['sql_query'] = sql_query

            sql_result = query_handler.execute_sql(sql_query)
            response = query_handler.format_sql_response(user_query, sql_result)

        elif query_type == 'rag':
            # Handle RAG queries
            search_results = rag_handler.search(user_query, top_k=3)
            response = query_handler.format_rag_response(user_query, search_results)

            metadata['search_results'] = len(search_results)
            metadata['used_rag'] = True

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
    return jsonify({
        'status': 'healthy',
        'service': 'Temple Chatbot API'
    })

if __name__ == '__main__':
    # Railway requires host=0.0.0.0 and dynamic port
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
