from flask import Flask, request, jsonify
from llama_index.core import Settings, Document, VectorStoreIndex
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core.tools import QueryEngineTool
from llama_index.core.query_engine import RouterQueryEngine,MultiStepQueryEngine
from llama_index.core.selectors import LLMSingleSelector, LLMMultiSelector
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
# Configure API and Environmen
os.environ["GOOGLE_API_KEY"] = "AIzaSyC4QTtNK9uy-Kt0ElZq-q81FRzMsT_3ha0"

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Configure LLM
llm = GoogleGenAI(model="gemini-2.0-flash",
                  system_prompt="You are an AI assistant for Indian law enforcement and citizens, designed to provide accurate legal information while dynamically adapting communication to the user's role. When detecting the user is a police officer, deliver technical, precise operational details using professional terminology. When engaging with citizens, use clear, compassionate language explaining rights and procedures. Your core principles include maintaining neutrality, protecting privacy, preventing information misuse, and directing users to official resources when necessary. Always ground responses in verified legal documents, cross-reference official sources, and aim to enhance understanding of legal processes while serving both institutional and public interests with empathy and accuracy.",
                  temperature=0.5,)
Settings.llm = llm

# Configure Embedding Model
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.embed_model = embed_model

# -------------------------- PDF Extraction --------------------------
def extract_text_from_pdf(pdf_path):
    pdf_text = ""
    doc = fitz.open(pdf_path)
    for page in doc:
        pdf_text += page.get_text()
    doc.close()
    return pdf_text

pdf_path = './police_act_1861.pdf'
pdf_text = extract_text_from_pdf(pdf_path)
pdf_doc = Document(text=pdf_text, metadata={"source": "Police Act"})

standing_order_path = './pso.pdf'
standing_order_text = extract_text_from_pdf(standing_order_path)
standing_order_doc = Document(text=standing_order_text, metadata={"source": "Standing Order"})

ipc_path = './IPC_codes.pdf'
ipc_text = extract_text_from_pdf(ipc_path)
ipc_doc = Document(text=ipc_text, metadata={"source": "THE INDIAN PENAL CODE"})

# -------------------------- CSV Extraction --------------------------
# file_path = './AD1.csv'
# data = pd.read_csv(file_path)

# csv_text = data.to_string()

csv_path="./fir.pdf"
csv_text = extract_text_from_pdf(csv_path)
csv_doc = Document(text=csv_text, metadata={"source": "First Information Report"})

# -------------------------- Emergency Numbers Extraction --------------------------
emergency_path = './emergency_numbers.pdf'
emergency_text = extract_text_from_pdf(emergency_path)
emergency_doc = Document(text=emergency_text, metadata={"source": "Emergency Numbers"})

# -------------------------- Initialize FAISS Vector Stores --------------------------
EMBED_DIMENSION = 768

def load_or_create_index(documents, vector_store, storage_path):
    try:
        # Try to load existing index
        storage_context = StorageContext.from_defaults(persist_dir=storage_path)
        return load_index_from_storage(storage_context)
    except:
        # Create new index if loading fails
        index = VectorStoreIndex.from_documents(documents, vector_store=vector_store)
        index.storage_context.persist(persist_dir=storage_path)
        return index

# Initialize vector stores and indexes
print("Loading/Creating Vector Stores")
faiss_pdf_index = faiss.IndexFlatL2(EMBED_DIMENSION)
pdf_vector_store = FaissVectorStore(faiss_index=faiss_pdf_index)
pdf_index = load_or_create_index([pdf_doc], pdf_vector_store, "./storage/police_act")

faiss_order_index = faiss.IndexFlatL2(EMBED_DIMENSION)
order_vector_store = FaissVectorStore(faiss_index=faiss_order_index)
order_index = load_or_create_index([standing_order_doc], order_vector_store, "./storage/standing_order")

faiss_csv_index = faiss.IndexFlatL2(EMBED_DIMENSION)
csv_vector_store = FaissVectorStore(faiss_index=faiss_csv_index)
csv_index = load_or_create_index([csv_doc], csv_vector_store, "./storage/fir")

faiss_ipc_index = faiss.IndexFlatL2(EMBED_DIMENSION)
ipc_vector_store = FaissVectorStore(faiss_index=faiss_ipc_index)
ipc_index = load_or_create_index([ipc_doc], ipc_vector_store, "./storage/ipc_codes")

faiss_emergency_index = faiss.IndexFlatL2(EMBED_DIMENSION)
emergency_vector_store = FaissVectorStore(faiss_index=faiss_emergency_index)
emergency_index = load_or_create_index([emergency_doc], emergency_vector_store, "./storage/emergency_numbers")

# -------------------------- Query Engines --------------------------
print("Creating Query Engines")
pdf_query_engine = pdf_index.as_query_engine(similarity_top_k=10,response_mode="compact")
order_query_engine = order_index.as_query_engine(similarity_top_k=10,response_mode="compact")
csv_query_engine = csv_index.as_query_engine(similarity_top_k=10,response_mode="compact")
ipc_query_engine = ipc_index.as_query_engine(similarity_top_k=10,response_mode="compact")
emergency_query_engine = emergency_index.as_query_engine(similarity_top_k=10, response_mode="compact")
print("Query Engines Created")
pdf_tool = QueryEngineTool.from_defaults(query_engine=pdf_query_engine, description="Query the Police Act 1861.")
csv_tool = QueryEngineTool.from_defaults(query_engine=csv_query_engine, description="Query the First Information Report details")

order_tool = QueryEngineTool.from_defaults(query_engine=order_query_engine, description="Query the Standing Order police standing orders refer to a set of permanent or long-standing instructions or procedures issued by the police authority, outlining how police officers should conduct themselves and handle various situations")
ipc_tool = QueryEngineTool.from_defaults(query_engine=ipc_query_engine, description="Query the Indian Penal Code.")
emergency_tool = QueryEngineTool.from_defaults(
    query_engine=emergency_query_engine,
    description="Query emergency contact numbers."
)

# -------------------------- Multi-Selector Routing Query Engine --------------------------

rquery_engine = RouterQueryEngine(
    selector=LLMMultiSelector.from_defaults(),
    query_engine_tools=[pdf_tool, order_tool, csv_tool, ipc_tool, emergency_tool],
    
)

router_tool = QueryEngineTool.from_defaults(
    query_engine=rquery_engine,
     description="Router tool for query engines"
)

agent = ReActAgent.from_tools(
    tools=[router_tool],
    llm=llm,
    verbose=True,
)

UPLOAD_FOLDER = './uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Define core file categories
CORE_FILES = {
    'police_act': {'path': './police_act_1861.pdf', 'type': 'pdf'},
    'standing_order': {'path': './pso.pdf', 'type': 'pdf'},
    'ipc_codes': {'path': './IPC_codes.pdf', 'type': 'pdf'},
    'police_cases': {'path': './AD1.csv', 'type': 'csv'},
    'emergent_contracts': {'path': './emergent_contracts.pdf', 'type': 'pdf'}
}

def merge_pdf_files(original_path, new_path):
    original_text = extract_text_from_pdf(original_path)
    new_text = extract_text_from_pdf(new_path)
    merged_doc = Document(text=original_text + "\n\n" + new_text, 
                         metadata={"source": os.path.basename(original_path)})
    return merged_doc

def merge_csv_files(original_path, new_path):
    original_df = pd.read_csv(original_path)
    new_df = pd.read_csv(new_path)
    merged_df = pd.concat([original_df, new_df]).drop_duplicates()
    merged_df.to_csv(original_path, index=False)
    return Document(text=merged_df.to_string(), 
                   metadata={"source": "Police Cases"})

@app.route('/admin/update-core-file', methods=['POST'])
def update_core_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    category = request.form.get('category')
    operation = request.form.get('operation', 'replace')  # 'replace' or 'merge'
    confirmed = request.form.get('confirmed', 'false') == 'true'
    
    if category not in CORE_FILES:
        return jsonify({'error': 'Invalid file category'}), 400
    
    if not confirmed:
        warning_message = (
            f"Warning: You are about to {operation} the {category} file. "
            "This operation will modify the existing data and trigger reindexing. "
            "This cannot be undone. Are you sure you want to continue?"
        )
        return jsonify({
            'warning': warning_message,
            'requiresConfirmation': True,
            'operation': operation,
            'category': category
        }), 200
    
    if file and category:
        try:
            filename = secure_filename(file.filename)
            temp_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(temp_path)
            
            core_file = CORE_FILES[category]
            
            # Merge or replace based on operation type
            if operation == 'merge':
                if core_file['type'] == 'pdf':
                    merged_doc = merge_pdf_files(core_file['path'], temp_path)
                else:  # CSV
                    merged_doc = merge_csv_files(core_file['path'], temp_path)
            else:  # replace
                if core_file['type'] == 'pdf':
                    text = extract_text_from_pdf(temp_path)
                    merged_doc = Document(text=text, metadata={"source": os.path.basename(core_file['path'])})
                else:  # CSV
                    df = pd.read_csv(temp_path)
                    df.to_csv(core_file['path'], index=False)
                    merged_doc = Document(text=df.to_string(), metadata={"source": "Police Cases"})
            
            # Update index
            vector_store = FaissVectorStore(faiss_index=faiss.IndexFlatL2(EMBED_DIMENSION))
            new_index = VectorStoreIndex.from_documents([merged_doc], vector_store=vector_store)
            new_index.storage_context.persist(f"./storage/{category}")
            
            os.remove(temp_path)
            
            return jsonify({'message': f'Successfully {operation}d {category}'})
        except Exception as e:
            return jsonify({'error': str(e)}), 500

@app.route('/admin/files', methods=['GET'])
def get_indexed_files():
    files = []
    for category, info in CORE_FILES.items():
        if os.path.exists(info['path']):
            files.append({
                'name': os.path.basename(info['path']),
                'path': info['path'],
                'category': category,
                'type': info['type'],
                'lastIndexed': datetime.fromtimestamp(os.path.getmtime(info['path'])).strftime('%Y-%m-%d %H:%M:%S')
            })
    return jsonify({'files': files})

@app.route('/admin/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        try:
            # Reindex based on file type
            if filename.endswith('.pdf'):
                text = extract_text_from_pdf(filepath)
                doc = Document(text=text, metadata={"source": filename})
                new_index = VectorStoreIndex.from_documents([doc])
                new_index.storage_context.persist(f"./storage/{filename}_index")
            elif filename.endswith('.csv'):
                df = pd.read_csv(filepath)
                text = df.to_string()
                doc = Document(text=text, metadata={"source": filename})
                new_index = VectorStoreIndex.from_documents([doc])
                new_index.storage_context.persist(f"./storage/{filename}_index")
                
            return jsonify({'message': 'File uploaded and indexed successfully'})
        except Exception as e:
            return jsonify({'error': f'Error processing file: {str(e)}'}), 500

@app.route('/admin/reindex', methods=['POST'])
def reindex_file():
    data = request.json
    filepath = data.get('filePath')
    confirmed = data.get('confirmed', False)
    
    if not confirmed:
        warning_message = (
            "Warning: You are about to reindex this file. "
            "This operation may take some time and will update the search index. "
            "Are you sure you want to continue?"
        )
        return jsonify({
            'warning': warning_message,
            'requiresConfirmation': True,
            'filepath': filepath
        }), 200
    
    try:
        if filepath.endswith('.pdf'):
            text = extract_text_from_pdf(filepath)
            doc = Document(text=text, metadata={"source": os.path.basename(filepath)})
            new_index = VectorStoreIndex.from_documents([doc])
            new_index.storage_context.persist(f"./storage/{os.path.basename(filepath)}_index")
        elif filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
            text = df.to_string()
            doc = Document(text=text, metadata={"source": os.path.basename(filepath)})
            new_index = VectorStoreIndex.from_documents([doc])
            new_index.storage_context.persist(f"./storage/{os.path.basename(filepath)}_index")
            
        return jsonify({'message': 'File reindexed successfully'})
    except Exception as e:
        return jsonify({'error': f'Error reindexing file: {str(e)}'}), 500

@app.route('/query')
def handle_query():
    query = request.args.get('query')
    try:
        response = agent.chat(query)
        return jsonify({"response": str(response)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/reset', methods=['POST'])
def reset_chat():
    try:
        agent.reset()
        return jsonify({"message": "Chat reset successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
