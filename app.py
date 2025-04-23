from flask import Flask, request, jsonify
from llama_index.core import Settings, Document, VectorStoreIndex , SimpleDirectoryReader
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core.tools import QueryEngineTool
from llama_index.core.query_engine import RouterQueryEngine,MultiStepQueryEngine
from llama_index.core.selectors import LLMSingleSelector, LLMMultiSelector
from llama_index.core.node_parser import SentenceSplitter

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

import  nest_asyncio
nest_asyncio.apply()

os.environ["GOOGLE_API_KEY"] = "AIzaSyDMxTJ6z8bUM83ymUoHAXGIW7YpXJ91v0A"

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

model = "gemini-2.0-flash"
system_prompt = (
    """"You are an AI assistant for Indian law enforcement and citizens and legal advice.You are knowledgeable about the Indian Penal Code (IPC), police procedures, and legal rights. "
    You can answer questions about FIRs, police powers, and legal definitions. 
    you can should translate all query into english
    you should use only tool alway pass in english\n but answer in query language default is english
    if there are in emergency or serious matter then you should provide emergency number and step to be taken 
    """
)
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

@app.route('/query', methods=['GET'])
def ask():
    question = request.args.get('query')
    if not question:
        return jsonify({"error": "No question provided"}), 400
    try:
        response = agent.chat(question)
        
        return jsonify({"response": str(response)})
        
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

if __name__ == '__main__':
    app.run(debug=True, port=5000)