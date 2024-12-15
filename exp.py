import os
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai

# Langchain and RAG imports
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()

image = os.getenv('LOGO_PATH')

# Configure page
st.set_page_config(page_title="NOC Assist RAG Chatbot", page_icon="🔍")
st.title("RAN Ops Assist 🔍📡")
st.info('Always follow Quality Points', icon="ℹ️") 
st.logo(image,size="medium", link=None, icon_image=None)


with st.sidebar:
    st.sidebar.header("Config")
    # New Chat
    if st.button("New Chat"):
        st.session_state.clear()
        st.rerun()      

# Google AI API Key input
# google_api_key = st.text_input("Google AI API Key", type="password")
os.environ["GEMINI_API_KEY"] = os.getenv('GEMINI_API_KEY')
google_api_key = os.environ["GEMINI_API_KEY"]

# Embedding and vector store setup
@st.cache_resource
def setup_rag_components():
    # Initialize embedding
    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # PDF loader and document processing
    path = os.getenv('PDF_PATH')
    loader = PyPDFDirectoryLoader(path)
    extracted_docs = loader.load()
    
    # Text splitting
    splits = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    docs = splits.split_documents(extracted_docs)
    
    # Create vector store
    vector_store = FAISS.from_documents(documents=docs, embedding=embedding)
    retriever = vector_store.as_retriever()
    
    return retriever

# Create RAG chain
def create_rag_chain(llm, retriever):
    # Prompt template
    prompt = ChatPromptTemplate.from_template(
        """ 
        Everytime don't give the same response, only follow the below format when questions asked about alarms from context.\n
        Act as a Telecom NOC Engineer with expertise in Radio Access Networks (RAN).
        Response should be in short format.
        Your responses should follow this structured format:
            1. Response: Provide an answer based on the given situation, with slight improvements for better clarity but from the context.
            2. Explanation of the issue: Include a brief explanation on why the issue might have occurred.
            3. Recommended steps/actions: Suggest further steps to resolve the issue.
            4. Quality steps to follow:
                - Check for relevant INC/CRQ tickets.
                - Follow the TSDANC format while creating INC.
                - Mention previous closed INC/CRQ information if applicable.
                - If there are >= 4 INCs on the same issue within 90 days, highlight the ticket to the SAM-SICC team and provide all relevant details.
        
        Context: {context}
        Question: {input}
        """
    )
    
    # Create document and retrieval chains
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    return retrieval_chain

# Main app logic
def main():
    # API Key validation
    if not google_api_key:
        st.info("Please add your Google AI API key to continue.", icon="🗝️")
        return

    # Configure Gemini API
    genai.configure(api_key=google_api_key)

    # Initialize Gemini LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp", 
        google_api_key=google_api_key,
        convert_system_message_to_human=True
    )

    
    # Setup RAG components
    try:
        retriever = setup_rag_components()
        rag_chain = create_rag_chain(llm, retriever)
    except Exception as e:
        st.error(f"Error setting up RAG components: {e}")
        return
    
    # Initialize session state for messages
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
        # Add initial greeting
        initial_greeting = """
        👋 Welcome to RAN Ops Assist! 
        
        I'm your AI-powered NOC (Network Operations Center) assistant, specialized in Radio Access Network (RAN) operations. 
        
        I can help you with:
        - Troubleshooting network issues
        - Providing insights on alarms and incidents
        - Guiding you through NOC best practices
        
        How can I assist you today with your telecom network operations?
        """
        
        st.session_state.messages.append({
            "role": "assistant", 
            "content": initial_greeting
        })


    # Initialize session state for messages
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display existing messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("What would you like to know about NOC operations?"):
        # Store and display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        try:
            # Invoke RAG chain
            response = rag_chain.invoke({"input": prompt})
            
            # Display and store assistant response
            with st.chat_message("assistant"):
                st.markdown(response['answer'])
            
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response['answer']
            })
            

        except Exception as e:
            st.error(f"An error occurred while generating response: {e}")

# Run the main app
if __name__ == "__main__":
    main()
