from flask import Flask, request, jsonify, Response, stream_with_context
from llama_index.core import Settings, Document, VectorStoreIndex , SimpleDirectoryReader
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core.tools import QueryEngineTool
from llama_index.core.query_engine import RouterQueryEngine,MultiStepQueryEngine
from llama_index.core.selectors import LLMSingleSelector, LLMMultiSelector
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.indices.query.schema import QueryType
from llama_index.core.response_synthesizers import TreeSummarize

import pandas as pd
import faiss
import fitz
import os
from flask_cors import CORS
from llama_index.core.indices.query.query_transform.base import (
    StepDecomposeQueryTransform,
)
from llama_index.core import (
    Settings, Document, VectorStoreIndex, StorageContext, load_index_from_storage
)
from llama_index.core.agent import ReActAgent
from datetime import datetime
from werkzeug.utils import secure_filename
import glob
import json
import uuid
from typing import Dict, List
from googletrans import Translator

import  nest_asyncio
nest_asyncio.apply()

os.environ["GOOGLE_API_KEY"] = "AIzaSyDMxTJ6z8bUM83ymUoHAXGIW7YpXJ91v0A"

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

model = "gemini-2.0-flash"
system_prompt = """You are an AI assistant for Indian law enforcement and citizens seeking legal advice. 
You are knowledgeable about the Indian Penal Code (IPC), police procedures, and legal rights.

Key Capabilities:
1. Answer questions about FIRs, police powers, and legal definitions
2. Translate queries from Indian languages to English (internally) and respond in the original language
3. Provide emergency numbers and immediate action steps for serious situations
4. Cite specific sections of laws and regulations with source references
5. Maintain conversation context and refer to previous discussion points
6. Provide step-by-step guidance for legal procedures

When providing answers:
- Always cite your sources with specific sections and page numbers
- For emergencies, lead with emergency contact numbers and immediate actions
- Break down complex legal concepts into simple, understandable terms
- Maintain cultural sensitivity and use local context
- If unsure, clearly state limitations and suggest consulting legal professionals

Language Handling:
- Accept queries in any Indian language
- Translate internally to English for processing
- Respond in the original query language
- Default to English if language is not specified"""

temperature = 0.5
llm = GoogleGenAI(model=model,temp=0.4, system_prompt=system_prompt)
# llm = LlamaCPP(
#     model_path="./gemma-3-1b-it-Q4_K_M.gguf",  # Change to your GGUF file path
#     temperature=0.7,
#     max_new_tokens=512,
#     context_window=2048,
#     model_kwargs={"n_gpu_layers": 35},  # set 0 for CPU
#     verbose=True
# )



import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5", device=device)

Settings.llm = llm
Settings.embed_model = embed_model
splitter = SentenceSplitter(chunk_size=1024,chunk_overlap=100)

# Initialize translator
translator = Translator()

# Load CORE_FILES from settings.json
with open('settings.json', 'r') as f:
    CORE_FILES = json.load(f)

def processOrLoad(file_path,EMBED_DIMENSION,storage_path):
    try:
        storage_context = StorageContext.from_defaults(persist_dir=storage_path)
        return load_index_from_storage(storage_context)
    except:
        docs = SimpleDirectoryReader(input_files=[file_path]).load_data()
        for doc in docs:
            doc.text_template ="Metadata:\n{metadata_str}\n---\nContent:\n{content}"
            if "page_label" not in doc.excluded_embed_metadata_keys:
                doc.excluded_embed_metadata_keys.append("page_label")
        nodes = splitter.get_nodes_from_documents(docs)
        indexes = faiss.IndexFlatL2(EMBED_DIMENSION)
        vector_store = FaissVectorStore(faiss_index=indexes)
        index = VectorStoreIndex(nodes, vector_store=vector_store)
        index.storage_context.persist(persist_dir=storage_path)
        return index


query_engine_tools = []
for category, info in CORE_FILES.items():
    file_path = info['path']
    storage_path = f"./storage/{category}"
    if not os.path.exists(storage_path):
        os.makedirs(storage_path)
    index = processOrLoad(file_path, 384, storage_path)
    print(f"Processed {category} and stored in {storage_path}")
    qengine = index.as_query_engine(similarity_top_k=10, response_mode="compact")
    query_engine_tools.append(
        QueryEngineTool.from_defaults(
            query_engine=qengine, 
            description=info['description']
        )
    )

rquery_engine = RouterQueryEngine(
    selector=LLMMultiSelector.from_defaults(),
    query_engine_tools=query_engine_tools,
    verbose=True,
    
)

router_tool = QueryEngineTool.from_defaults(
    query_engine=rquery_engine,
     description="Router tool for routing queries",
)

agent = ReActAgent.from_tools(
    tools=[router_tool],
    llm=llm,
    verbose=True,
    system_prompt=system_prompt,
 
)

agent.chat(system_prompt)

# Chat history storage
chat_histories: Dict[str, List] = {}

def detect_and_translate(text: str) -> tuple[str, str, str]:
    """
    Detect language and translate if not English.
    Returns: (translated_text, source_language, confidence)
    """
    try:
        # Detect language
        detection = translator.detect(text)
        source_lang = detection.lang
        confidence = detection.confidence

        # Translate if not English
        if source_lang != 'en':
            translation = translator.translate(text, dest='en')
            return translation.text, source_lang, str(confidence)
        
        return text, 'en', str(confidence)
    except Exception as e:
        print(f"Translation error: {e}")
        return text, 'en', '1.0'

def translate_response(text: str, target_lang: str) -> str:
    """Translate response back to original language if needed"""
    if target_lang != 'en':
        try:
            translation = translator.translate(text, dest=target_lang)
            return translation.text
        except Exception as e:
            print(f"Translation error: {e}")
    return text

def summarize_document(file_key: str) -> dict:
    """Summarize a document using TreeSummarize"""
    try:
        if file_key not in CORE_FILES:
            raise ValueError(f"Invalid file key: {file_key}")
            
        file_path = CORE_FILES[file_key]['path']
        storage_path = f"./storage/{file_key}"
        
        # Get the index
        index = processOrLoad(file_path, 384, storage_path)
        
        # Create a summarize query engine
        query_engine = index.as_query_engine(
            response_mode="tree_summarize",
            verbose=True
        )
        
        # Generate summary
        summary = query_engine.query(
            "Please provide a comprehensive summary of this document, including key points, main sections, and important legal aspects."
        )
        
        return {
            "summary": str(summary),
            "file": CORE_FILES[file_key]['path'],
            "description": CORE_FILES[file_key]['description']
        }
    except Exception as e:
        raise Exception(f"Error summarizing document: {str(e)}")

def compare_documents(file_key1: str, file_key2: str, aspect: str = None) -> dict:
    """Compare two documents based on specific aspects or overall content"""
    try:
        if file_key1 not in CORE_FILES or file_key2 not in CORE_FILES:
            raise ValueError("Invalid file keys provided")
            
        # Get indexes for both documents
        index1 = processOrLoad(CORE_FILES[file_key1]['path'], 384, f"./storage/{file_key1}")
        index2 = processOrLoad(CORE_FILES[file_key2]['path'], 384, f"./storage/{file_key2}")
        
        # Create query engines
        engine1 = index1.as_query_engine(response_mode="compact")
        engine2 = index2.as_query_engine(response_mode="compact")
        
        # Prepare comparison query
        if aspect:
            query = f"Analyze and compare the {aspect} aspects of these documents."
        else:
            query = "Compare and contrast the main points, scope, and legal implications of these documents."
            
        # Get responses from both documents
        response1 = engine1.query(query)
        response2 = engine2.query(query)
        
        # Use LLM to synthesize comparison
        comparison_prompt = f"""
        Compare these two documents:
        Document 1 ({file_key1}): {str(response1)}
        Document 2 ({file_key2}): {str(response2)}
        
        Provide a detailed comparison focusing on:
        1. Key similarities
        2. Major differences
        3. Complementary aspects
        4. Legal implications
        """
        
        comparison = llm.complete(comparison_prompt)
        
        return {
            "comparison": str(comparison),
            "file1": CORE_FILES[file_key1]['path'],
            "file2": CORE_FILES[file_key2]['path'],
            "aspect": aspect if aspect else "overall"
        }
    except Exception as e:
        raise Exception(f"Error comparing documents: {str(e)}")

def find_related_cases(query: str, top_k: int = 5) -> list:
    """Find related legal cases based on a query"""
    try:
        # Use all indexes to find relevant cases
        all_results = []
        for category, info in CORE_FILES.items():
            index = processOrLoad(info['path'], 384, f"./storage/{category}")
            engine = index.as_query_engine(
                similarity_top_k=top_k,
                response_mode="compact"
            )
            
            # Query for related cases
            response = engine.query(
                f"Find legal cases or precedents related to: {query}"
            )
            
            if hasattr(response, 'source_nodes'):
                for node in response.source_nodes:
                    if hasattr(node, 'metadata'):
                        case = {
                            'text': node.text,
                            'file': node.metadata.get('file_name', 'Unknown'),
                            'page': node.metadata.get('page_label', 'Unknown'),
                            'relevance_score': node.score if hasattr(node, 'score') else None
                        }
                        all_results.append(case)
        
        # Sort by relevance score if available
        all_results.sort(key=lambda x: x['relevance_score'] if x['relevance_score'] is not None else 0, reverse=True)
        
        return all_results[:top_k]
    except Exception as e:
        raise Exception(f"Error finding related cases: {str(e)}")

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({"error": "No message provided"}), 400
    
    # Get or create chat session
    chat_id = data.get('chat_id', str(uuid.uuid4()))
    if chat_id not in chat_histories:
        chat_histories[chat_id] = []
    
    try:
        # Detect and translate user message
        translated_message, source_lang, confidence = detect_and_translate(data['message'])

        # Add user message to history
        chat_histories[chat_id].append({
            "role": "user",
            "content": data['message'],
            "translated_content": translated_message,
            "source_language": source_lang,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat()
        })
        
        def generate():
            # Get response with sources
            response = agent.chat(translated_message)
            source_nodes = []
            
            # Extract source nodes if available
            if hasattr(response, 'source_nodes'):
                source_nodes = response.source_nodes
            elif hasattr(response, 'metadata') and 'source_nodes' in response.metadata:
                source_nodes = response.metadata['source_nodes']
            
            # Format sources
            sources = []
            for node in source_nodes:
                if hasattr(node, 'metadata'):
                    source = {
                        'text': node.text[:200] + "...",
                        'file': node.metadata.get('file_name', 'Unknown'),
                        'page': node.metadata.get('page_label', 'Unknown')
                    }
                    sources.append(source)
            
            # Translate response back to original language
            translated_response = translate_response(str(response), source_lang)

            # Add assistant response to history
            chat_histories[chat_id].append({
                "role": "assistant",
                "content": str(response),
                "translated_content": translated_response,
                "sources": sources,
                "timestamp": datetime.now().isoformat()
            })
            
            # Return response with chat_id and history
            yield json.dumps({
                "chat_id": chat_id,
                "response": translated_response,
                "sources": sources,
                "history": chat_histories[chat_id],
                "confidence": confidence
            })
        
        return Response(stream_with_context(generate()), content_type='application/json')
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/chat/<chat_id>/history', methods=['GET'])
def get_chat_history(chat_id):
    if chat_id not in chat_histories:
        return jsonify({"error": "Chat session not found"}), 404
    return jsonify({"history": chat_histories[chat_id]})

@app.route('/chat/<chat_id>', methods=['DELETE'])
def delete_chat_history(chat_id):
    if chat_id in chat_histories:
        del chat_histories[chat_id]
    return jsonify({"message": "Chat history deleted successfully"})

@app.route('/query', methods=['GET'])
def ask():
    question = request.args.get('query')
    if not question:
        return jsonify({"error": "No question provided"}), 400
    
    try:
        # Detect and translate question
        translated_question, source_lang, confidence = detect_and_translate(question)

        def generate():
            # Get response and source nodes
            response = agent.chat(translated_question)
            source_nodes = []
            
            # Extract source nodes if available
            if hasattr(response, 'source_nodes'):
                source_nodes = response.source_nodes
            elif hasattr(response, 'metadata') and 'source_nodes' in response.metadata:
                source_nodes = response.metadata['source_nodes']
            
            # Format sources
            sources = []
            for node in source_nodes:
                if hasattr(node, 'metadata'):
                    source = {
                        'text': node.text[:200] + "...",  # First 200 chars
                        'file': node.metadata.get('file_name', 'Unknown'),
                        'page': node.metadata.get('page_label', 'Unknown')
                    }
                    sources.append(source)
            
            # Translate response back to original language
            translated_response = translate_response(str(response), source_lang)

            # Return response with sources
            yield json.dumps({
                "response": translated_response,
                "sources": sources,
                "confidence": confidence
            })
        
        return Response(stream_with_context(generate()), content_type='application/json')
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/reset', methods=['POST'])
def reset():
    agent.reset()
    agent.chat(system_prompt)
    return jsonify({"message": "Agent reset successfully"}), 200

@app.route('/get_files', methods=['GET'])
def get_files():
    try:
        # Debug log to check CORE_FILES content
        #print("CORE_FILES content:", CORE_FILES)
        # Return core files with their descriptions and paths
        return jsonify({k: {"description": v["description"], "path": v["path"]} for k, v in CORE_FILES.items()}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

ALLOWED_EXTENSIONS = {'pdf'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/replace/<file_key>', methods=['POST'])
def replace_core_file(file_key):
    if file_key not in CORE_FILES:
        return jsonify({"error": f"Invalid file key: {file_key}"}), 400
    
    if 'file' not in request.files:
        return jsonify({"error": "No file part in request"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    if file and allowed_file(file.filename):
        target_path = CORE_FILES[file_key]['path']
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        file.save(target_path)
        return jsonify({"message": f"File '{file_key}' replaced successfully."}), 200
    else:
        return jsonify({"error": "Invalid file type"}), 400

@app.route('/edit_description/<file_key>', methods=['POST'])
def edit_description(file_key):
    if file_key not in CORE_FILES:
        return jsonify({"error": f"Invalid file key: {file_key}"}), 400
    
    data = request.get_json()
    if not data or 'description' not in data:
        return jsonify({"error": "No description provided"}), 400
        
    try:
        # Update the description in memory
        CORE_FILES[file_key]['description'] = data['description']
        
        # Write the updated CORE_FILES to settings.json
        with open('settings.json', 'w') as f:
            json.dump(CORE_FILES, f, indent=4)
            
        return jsonify({
            "message": f"Description for '{file_key}' updated successfully",
            "new_description": data['description']
        }), 200
        
    except Exception as e:
        return jsonify({"error": f"Failed to update description: {str(e)}"}), 500

@app.route('/rebuild_indexes', methods=['POST'])
def rebuild_indexes():
    try:
        # Delete all storage directories
        storage_dir = "./storage"
        if os.path.exists(storage_dir):
            import shutil
            shutil.rmtree(storage_dir)
            
        # Rebuild query engine tools
        global query_engine_tools, rquery_engine, router_tool, agent
        query_engine_tools = []
        
        # Recreate indexes
        for category, info in CORE_FILES.items():
            file_path = info['path']
            storage_path = f"./storage/{category}"
            os.makedirs(storage_path, exist_ok=True)
            
            # Force new index creation
            index = processOrLoad(file_path, 384, storage_path)
            print(f"Rebuilt index for {category}")
            
            qengine = index.as_query_engine(similarity_top_k=10, response_mode="compact")
            query_engine_tools.append(
                QueryEngineTool.from_defaults(
                    query_engine=qengine,
                    description=info['description']
                )
            )
        
        # Reinitialize router and agent
        rquery_engine = RouterQueryEngine(
            selector=LLMMultiSelector.from_defaults(),
            query_engine_tools=query_engine_tools,
            verbose=True
        )
        
        router_tool = QueryEngineTool.from_defaults(
            query_engine=rquery_engine,
            description="Router tool for routing queries",
        )
        
        agent = ReActAgent.from_tools(
            tools=[router_tool],
            llm=llm,
            verbose=True,
        )
        
        return jsonify({"message": "All indexes rebuilt successfully"}), 200
        
    except Exception as e:
        return jsonify({"error": f"Failed to rebuild indexes: {str(e)}"}), 500

@app.route('/summarize/<file_key>', methods=['GET'])
def get_document_summary(file_key):
    """Get a comprehensive summary of a document"""
    try:
        summary = summarize_document(file_key)
        return jsonify(summary), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/compare', methods=['POST'])
def compare_docs():
    """Compare two documents"""
    data = request.get_json()
    if not data or 'file_key1' not in data or 'file_key2' not in data:
        return jsonify({"error": "Both file keys are required"}), 400
    
    try:
        comparison = compare_documents(
            data['file_key1'], 
            data['file_key2'],
            data.get('aspect')  # Optional aspect parameter
        )
        return jsonify(comparison), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/find_related', methods=['GET'])
def get_related_cases():
    """Find related legal cases"""
    query = request.args.get('query')
    if not query:
        return jsonify({"error": "Query parameter is required"}), 400
    
    try:
        top_k = int(request.args.get('limit', '5'))
        cases = find_related_cases(query, top_k)
        return jsonify({
            "query": query,
            "cases": cases,
            "total_results": len(cases)
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)