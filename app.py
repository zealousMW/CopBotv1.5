from flask import Flask, request, jsonify, send_file, send_from_directory
from llama_index.core import Settings, Document, VectorStoreIndex , SimpleDirectoryReader
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core.tools import FunctionTool
from llama_index.core.query_engine import MultiStepQueryEngine
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
from llama_index.core.agent import ReActAgent, ReActAgent
from datetime import datetime
from werkzeug.utils import secure_filename
import glob
import json

import  nest_asyncio
nest_asyncio.apply()

os.environ["GOOGLE_API_KEY"] = "AIzaSyDMxTJ6z8bUM83ymUoHAXGIW7YpXJ91v0A"

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Add static file serving for temp directory
@app.route('/temp/<path:filename>')
def serve_file(filename):
    try:
        return send_from_directory('temp', filename)
    except Exception as e:
        return jsonify({"error": str(e)}), 404

model = "gemini-2.0-flash"
system_prompt = (
    """You are an AI assistant for Indian law enforcement and citizens and legal advice. You should:

    When handling queries:
    1. Expand specific queries into their broader legal categories
       - If about specific animals -> expand to animal protection laws
       - If about specific places -> expand to property and trespassing laws
       - If about specific people -> expand to personal rights and protection laws
    
    2. Always check for:
       - Related criminal offenses
       - Associated civil violations
       - Prevention of cruelty laws
       - Public safety regulations
       - Environmental protection rules
    
    3. Use knowledge tools to find:
       - Applicable laws and regulations
       - Legal precedents
       - Relevant penalties
       - Reporting procedures
    
    4. Present information with:
       - Clear legal implications
       - Preventive measures
       - Reporting channels
       - Emergency contacts if needed
    
    Important:
    - Treat specific examples as part of broader legal categories
    - Always consider ethical and legal implications
    - Identify both direct and related violations
    - Provide prevention and reporting guidance
    
    Always translate non-English queries to English before processing.
    Format final response in the query's original language.
    """
)
temperature = 0.5
llm = GoogleGenAI(model=model,temp=0.4)
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
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5", device=device)

Settings.llm = llm
Settings.embed_model = embed_model
splitter = SentenceSplitter(chunk_size=1024,chunk_overlap=256)

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
    index = processOrLoad(file_path, 768, storage_path)
    print(f"Processed {category} and stored in {storage_path}")
    qengine = index.as_query_engine(similarity_top_k=30, response_mode="compact")
    
    # Create function tool for each query engine
    def create_query_fn(engine, category):
        def query_function(query: str) -> str:
            """
            Search the {category} knowledge base for legal information.
            
            First expand the query to consider:
            - Relevant legal terms
            - Related procedures and rights
            - Similar legal scenarios
            
            Args:
                query (str): A legal query that will be analyzed and expanded for comprehensive search
            Returns:
                str: Detailed response from the legal knowledge base
            """
            response = engine.query(query)
            return str(response)
        return query_function
    
    query_fn = create_query_fn(qengine, category)
    query_fn.__name__ = f"query_{category}"
    
    query_engine_tools.append(
        FunctionTool.from_defaults(
            fn=query_fn,
            name=f"query_{category}",
            description=info['description']
        )
    )

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

def create_fir_draft(
    query: str,
    police_station: str = None,
    district: str = None,
    date_occurrence: str = None,
    date_reported: str = None,
    complainant_details: str = None,
    offence_description: str = None,
    ipc_sections: str = None,
    place_occurrence: str = None,
    criminal_details: str = None,
    investigation_steps: str = None,
    incident_description: str = None
) -> dict:
    """
    Create a First Information Report (FIR) draft based on provided information
    
    Args:
        query (str): Original query to extract missing information
        police_station (str): Name of police station
        district (str): District name
        date_occurrence (str): Date and time of occurrence
        date_reported (str): Date and time reported
        complainant_details (str): Name and address of complainant
        offence_description (str): Brief description of offence
        ipc_sections (str): Applicable IPC sections
        place_occurrence (str): Place of occurrence and distance from PS
        criminal_details (str): Criminal name/address if known
        investigation_steps (str): Initial investigation steps
        incident_description (str): Detailed incident description
    Returns:
        dict: Formatted FIR draft and PDF URL
    """
    # Generate FIR number using timestamp
    fir_number = f"FIR{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    # Current date/time for dispatch
    dispatch_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Create text version
    text_content = f"""
FIRST INFORMATION REPORT
========================
Police Station: {police_station if police_station else '[POLICE STATION REQUIRED]'}
District: {district if district else '[DISTRICT REQUIRED]'}
FIR Number: {fir_number}

Date/Time of Occurrence: {date_occurrence if date_occurrence else '[DATE/TIME OF OCCURRENCE REQUIRED]'}
Date/Time Reported: {date_reported if date_reported else '[DATE/TIME REPORTED REQUIRED]'}

Complainant Details:
{complainant_details if complainant_details else '[COMPLAINANT DETAILS REQUIRED]'}

Brief Description of Offence:
{offence_description if offence_description else '[OFFENCE DESCRIPTION REQUIRED]'}

IPC Section(s):
{ipc_sections if ipc_sections else '[IPC SECTIONS REQUIRED]'}

Place of Occurrence:
{place_occurrence if place_occurrence else '[PLACE OF OCCURRENCE REQUIRED]'}

Criminal Details (if known):
{criminal_details if criminal_details else 'Unknown/Not Provided'}

Investigation Steps:
{investigation_steps if investigation_steps else 'Pending Investigation'}

Date/Time of Dispatch from PS: {dispatch_datetime}

Detailed Incident Description:
{incident_description if incident_description else '[DETAILED DESCRIPTION REQUIRED]'}
"""

    # Generate PDF
    pdf_path = f"temp/fir_{fir_number}.pdf"
    os.makedirs("temp", exist_ok=True)
    
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=A4,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=72
    )
    
    styles = getSampleStyleSheet()
    story = []
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=16,
        spaceAfter=30
    )
    
    section_style = ParagraphStyle(
        'SectionTitle',
        parent=styles['Heading2'],
        fontSize=12,
        spaceAfter=12
    )
    
    content_style = ParagraphStyle(
        'Content',
        parent=styles['Normal'],
        fontSize=10,
        spaceAfter=12
    )
    
    # Add content
    story.append(Paragraph("FIRST INFORMATION REPORT", title_style))
    story.append(Paragraph(f"FIR Number: {fir_number}", section_style))
    story.append(Spacer(1, 12))
    
    # Add sections
    sections = [
        ("JURISDICTION", f"Police Station: {police_station or '[REQUIRED]'}\nDistrict: {district or '[REQUIRED]'}"),
        ("DATE & TIME", f"Occurrence: {date_occurrence or '[REQUIRED]'}\nReported: {date_reported or '[REQUIRED]'}\nDispatch: {dispatch_datetime}"),
        ("COMPLAINANT", complainant_details or '[REQUIRED]'),
        ("OFFENCE DETAILS", offence_description or '[REQUIRED]'),
        ("IPC SECTIONS", ipc_sections or '[REQUIRED]'),
        ("LOCATION", place_occurrence or '[REQUIRED]'),
        ("ACCUSED DETAILS", criminal_details or 'Unknown'),
        ("INVESTIGATION", investigation_steps or 'Pending'),
        ("INCIDENT DESCRIPTION", incident_description or '[REQUIRED]')
    ]
    
    for title, content in sections:
        story.append(Paragraph(title, section_style))
        story.append(Paragraph(content.replace('\n', '<br/>\n'), content_style))
        story.append(Spacer(1, 12))
    
    doc.build(story)
    
    # Update PDF URL to use temp route
    return {
        "text": text_content,
        "pdf_url": f"/temp/fir_{fir_number}.pdf",
        "fir_number": fir_number,
        "is_fir": True
    }

# Add FIR tool to the tools list before agent creation
fir_tool = FunctionTool.from_defaults(
    fn=create_fir_draft,
    name="create_fir_draft",
    description="Creates a draft FIR (First Information Report) with provided details. Will prompt for missing required information."
)

query_engine_tools.append(fir_tool)

system_prompt += """
For FIR-related tasks:
1. If user asks to create FIR:
   - Use create_fir_draft to generate draft
   - Provide both draft text and PDF link to user
"""

agent = ReActAgent.from_tools(
    tools=query_engine_tools,
    llm=llm,
    system_prompt=system_prompt,
    verbose=True,
)

from deep_translator import GoogleTranslator
import fasttext
import os
import urllib.request

# Download and load FastText model if not exists
def load_fasttext_model():
    model_path = "lid.176.ftz"
    if not os.path.exists(model_path):
        print("Downloading FastText language detection model...")
        urllib.request.urlretrieve(
            "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz",
            model_path
        )
    return fasttext.load_model(model_path)

# Initialize translator and language detector
lang_model = load_fasttext_model()

def translate_text(text, source='auto', target='en'):
    translator = GoogleTranslator(source=source, target=target)
    return translator.translate(text)

def detect_and_translate(text):
    try:
        # FastText returns list of labels and probabilities
        predictions = lang_model.predict(text, k=1)
        detected_lang = predictions[0][0].replace('__label__', '')  # Remove prefix
        
        if detected_lang != 'en':
            eng_text = translate_text(text, source=detected_lang, target='en')
            return eng_text, detected_lang
        return text, 'en'
    except Exception as e:
        print(f"Translation error: {e}")
        return text, 'en'

@app.route('/query', methods=['GET'])
def ask():
    question = request.args.get('query')
    if not question:
        return jsonify({"error": "No question provided"}), 400
    try:
        # Detect language and translate to English if needed
        eng_question, src_lang = detect_and_translate(question)
        print(f"Detected language: {src_lang}")
        print(f"Translated question: {eng_question}")
        
        # Get response from agent
        response = agent.chat(eng_question)
        response_text = str(response)
        is_fir = "FIRST INFORMATION REPORT" in response_text
        
        if src_lang != 'en':
            response_text = translate_text(response_text, source='en', target=src_lang)
        
        return jsonify({
            "response": response_text,
            "detected_language": src_lang,
            "is_fir": is_fir,
            "fir_number": response.fir_number if hasattr(response, 'fir_number') else None,
            "pdf_url": response.pdf_url if hasattr(response, 'pdf_url') else None
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/reset', methods=['POST'])
def reset():
    try:
        # Reset agent
        agent.reset()
        
        # Clear temp directory
        temp_dir = "./temp"
        if os.path.exists(temp_dir):
            for file in os.listdir(temp_dir):
                file_path = os.path.join(temp_dir, file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")
        
        return jsonify({"message": "Agent reset and temp files cleared successfully"}), 200
    except Exception as e:
        return jsonify({"error": f"Reset failed: {str(e)}"}), 500

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
        global query_engine_tools, agent
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
            
            # Create function tool for each query engine
            def create_query_fn(engine, category):
                def query_function(query: str) -> str:
                    """
                    Query the {category} knowledge base
                    Args:
                        query (str): The query to search for in {category} documents
                    Returns:
                        str: Response from the knowledge base
                    """
                    response = engine.query(query)
                    return str(response)
                return query_function
            
            query_fn = create_query_fn(qengine, category)
            query_fn.__name__ = f"query_{category}"
            
            query_engine_tools.append(
                FunctionTool.from_defaults(
                    fn=query_fn,
                    name=f"query_{category}",
                    description=info['description']
                )
            )
        
        agent = ReActAgent.from_tools(
            tools=query_engine_tools,
            llm=llm,
            system_prompt=system_prompt,
            verbose=True,
        )
        
        return jsonify({"message": "All indexes rebuilt successfully"}), 200
        
    except Exception as e:
        return jsonify({"error": f"Failed to rebuild indexes: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)