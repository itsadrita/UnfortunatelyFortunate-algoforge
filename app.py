from dotenv import load_dotenv
from gtts import gTTS
import os
import io
import streamlit as st
import fitz  # PyMuPDF
from langchain_text_splitters.character import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.docstore.document import Document

load_dotenv()
working_dir = os.path.dirname(os.path.abspath(__file__))

def load_document(file_path):
    """Loads a PDF document using PyMuPDF and returns the extracted text, page number, and image."""
    print("Loading document using PyMuPDF...")
    with fitz.open(file_path) as doc:
        documents = []
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text("text")  # Extract text from the page
            
            # Extract image from the page
            pixmap = page.get_pixmap()
            img_bytes = pixmap.tobytes()
            
            # Create a Document object with text, page number, and image
            document = Document(
                page_content=text,
                metadata={"page_number": page_num, "image": img_bytes}
            )
            documents.append(document)
            
    print(f"Loaded {len(documents)} pages.")
    return documents

def setup_vectorstore(documents):
    """Sets up a FAISS vector store from the loaded documents."""
    print("Setting up vector store...")
    embeddings = HuggingFaceEmbeddings()
    text_splitter = CharacterTextSplitter(
        separator="\n",  # Use newline as separator
        chunk_size=1000,
        chunk_overlap=200
    )
    doc_chunks = text_splitter.split_documents(documents)
    print(f"Split document into {len(doc_chunks)} chunks.")
    vectorstore = FAISS.from_documents(doc_chunks, embeddings)
    print("Vector store setup complete.")
    return vectorstore

def get_response(vectorstore, question, chat_history):
    """Fetches the response using the vector store and ChatGroq."""
    print("Searching in vector store and generating response...")
    try:
        # Search in vector store for relevant chunks
        docs = vectorstore.similarity_search(question, k=3)
        
        context = ""
        images = []
        for doc in docs:
            page_number = doc.metadata["page_number"]
            image_bytes = doc.metadata["image"]
            context += f"Page {page_number}:\n{doc.page_content}\n\n"
            images.append((page_number, image_bytes))
        
        # Build prompt with conversation history and context
        full_prompt = "\n".join([f"User: {msg['content']}" if msg['role'] == 'user' else f"Assistant: {msg['content']}" for msg in chat_history])
        full_prompt += f"\nUser: {question}\nContext: {context}\nAssistant:"

        # Call the LLaMA model via ChatGroq
        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
        response = llm.invoke(full_prompt)
        
        # Generate audio from response
        tts = gTTS(text=response.content, lang='en')
        audio_fp = io.BytesIO()
        tts.write_to_fp(audio_fp)
        audio_fp.seek(0)
        
        # Return all three values
        return response.content, audio_fp, images

    except Exception as e:
        print(f"Error in get_response: {str(e)}")
        return str(e), io.BytesIO(), []  # Return error message, empty audio, and empty images

# Streamlit page setup
st.set_page_config(
    page_title="Chat with Doc",
    page_icon="📄",
    layout="centered"
)

st.title("🦙 Chat with Doc - LLAMA 3.1")

# Initialize chat history in Streamlit session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Replace the file uploader section with document folder handling
documents_dir = os.path.join(working_dir, "documents")

# Create documents directory if it doesn't exist
if not os.path.exists(documents_dir):
    os.makedirs(documents_dir)

# List all folders in the documents directory
folders = [d for d in os.listdir(documents_dir) 
          if os.path.isdir(os.path.join(documents_dir, d))]

# Add a selectbox to choose the folder
selected_folder = st.selectbox(
    "Select a document folder",
    folders,
    index=None,
    placeholder="Choose a folder..."
)

if selected_folder:
    folder_path = os.path.join(documents_dir, selected_folder)
    
    # Get all PDF files in the selected folder
    pdf_files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]
    
    if pdf_files:  # If there are PDF files in the folder
        # Load and combine all PDFs in the folder
        if "vectorstore" not in st.session_state:
            print("Loading documents and setting up vectorstore...")
            all_documents = []
            for pdf_file in pdf_files:
                file_path = os.path.join(folder_path, pdf_file)
                documents = load_document(file_path)
                all_documents.extend(documents)
            
            st.session_state.vectorstore = setup_vectorstore(all_documents)
    else:
        st.warning("No PDF files found in the selected folder.")

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input field for asking questions
user_input = st.chat_input("Ask your question...")

if user_input:
    print(f"User input: {user_input}")

    # Save user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)

    # Get assistant response using vectorstore and LLaMA model
    with st.chat_message("assistant"):
        # Only three values here
        assistant_response, audio_fp, images = get_response(st.session_state.vectorstore, user_input, st.session_state.chat_history)
        print(f"Assistant response: {assistant_response}")

        # Display assistant response
        st.markdown(assistant_response)
        
        # Display relevant images with page numbers
        for page_number, image_bytes in images:
            st.image(image_bytes, caption=f"Page {page_number}", use_column_width=True)
        
        st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})