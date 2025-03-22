from flask import Flask, request, send_file, jsonify
from flask_restful import Api, Resource
import io
import base64
from dotenv import load_dotenv
import os
from app import load_document, setup_vectorstore, get_response
from flask_cors import CORS

load_dotenv()
working_dir = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
CORS(app)
api = Api(app)

vectorstore = None  # Initialize global vectorstore variable

@app.route('/folders', methods=['GET'])
def get_folders():
    documents_dir = os.path.join(working_dir, "documents")
    if not os.path.exists(documents_dir):
        os.makedirs(documents_dir)
    
    folders = [d for d in os.listdir(documents_dir) 
              if os.path.isdir(os.path.join(documents_dir, d))]
    return jsonify(folders)

@app.route('/select-folder', methods=['POST'])
def select_folder():
    global vectorstore  # Add this line to explicitly use global variable
    data = request.get_json()
    selected_folder = data.get('folder')
    if not selected_folder:
        return {'error': 'No folder selected'}, 400
    
    folder_path = os.path.join(working_dir, "documents", selected_folder)
    if not os.path.exists(folder_path):
        return {'error': 'Folder not found'}, 404
    
    # Get all PDF files in the selected folder
    pdf_files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]
    if not pdf_files:
        return {'error': 'No PDF files found in folder'}, 400
    
    # Load and process documents
    all_documents = []
    for pdf_file in pdf_files:
        file_path = os.path.join(folder_path, pdf_file)
        documents = load_document(file_path)
        all_documents.extend(documents)
    
    # Store vectorstore in global variable or session
    vectorstore = setup_vectorstore(all_documents)
    
    return {'message': 'Folder selected and documents processed successfully'}, 200

class FileUpload(Resource):
    def post(self):
        # Check if a file was included in the request
        if 'file' not in request.files:
            return {'error': 'No file uploaded'}, 400
        
        file = request.files['file']
        # Check if a file was actually selected
        if file.filename == '':
            return {'error': 'No file selected'}, 400
        
        # Save the file to the working directory
        file_path = f"{working_dir}/{file.filename}"
        file.save(file_path)
        
        # Process the document
        documents = load_document(file_path)
        vectorstore = setup_vectorstore(documents)
        
        return {'message': 'File uploaded and processed successfully'}, 200

class Chat(Resource):
    def post(self):
        global vectorstore
        if vectorstore is None:
            return {'error': 'Please select a folder first'}, 400
            
        try:
            data = request.get_json()
            question = data.get('question')
            chat_history = data.get('chat_history', [])
            
            if not question:
                return {'error': 'Question is required'}, 400
            
            # Get response from vectorstore - now expecting three values
            assistant_response, audio_fp, images = get_response(vectorstore, question, chat_history)
            
            # Convert image bytes to base64 for JSON serialization
            encoded_images = []
            for page_number, image_bytes in images:
                encoded_image = base64.b64encode(image_bytes).decode('utf-8')
                encoded_images.append({'page_number': page_number, 'image': encoded_image})
            
            # Convert audio to base64
            audio_base64 = base64.b64encode(audio_fp.getvalue()).decode('utf-8')
            
            return {
                'response': assistant_response,
                'audio': audio_base64,
                'images': encoded_images
            }, 200
            
        except Exception as e:
            print(f"Error in Chat.post: {str(e)}")
            return {'error': str(e)}, 500

# Register resources
api.add_resource(FileUpload, '/upload')
api.add_resource(Chat, '/chat')

if __name__ == '__main__':
    app.run(port=5000)
