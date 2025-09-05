# Complete RAG Pipeline with Llama2/Mistral using LangChain and Chroma
# Make sure to install: pip install langchain chromadb ollama pypdf sentence-transformers

import os
from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


class RAGPipeline:
    def __init__(self, model_name="llama2"):
        """
        Initialize RAG Pipeline
        model_name: 'llama2', 'mistral', 'codellama', etc.
        """
        self.model_name = model_name
        self.vectorstore = None
        self.qa_chain = None
        
        # Initialize LLM
        self.llm = OllamaLLM(
            model=model_name,
            temperature=0.1,
            # Adjust these based on your hardware
            num_predict=512,  # Max tokens to generate
        )
        
        # Initialize embeddings (using free HuggingFace model)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}  # Change to 'cuda' if you have GPU
        )
        
        print(f"✅ RAG Pipeline initialized with {model_name}")
    
    def load_and_process_pdf(self, pdf_path: str, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Load PDF and split into chunks
        """
        print(f"📄 Loading PDF: {pdf_path}")
        
        # Load PDF
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        
        print(f"📊 Loaded {len(documents)} pages")
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        chunks = text_splitter.split_documents(documents)
        print(f"✂️ Split into {len(chunks)} chunks")
        
        return chunks
    
    def create_vectorstore(self, chunks, persist_directory="./chroma_db"):
        """
        Create and populate Chroma vector store
        """
        print("🔍 Creating embeddings and vector store...")
        
        # Create Chroma vector store
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=persist_directory
        )

        print(f"💾 Vector store created and saved to {persist_directory}")
        
        return self.vectorstore
    
    def load_existing_vectorstore(self, persist_directory="./chroma_db"):
        """
        Load existing vector store
        """
        self.vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings
        )
        print(f"📂 Loaded existing vector store from {persist_directory}")
    
    def setup_qa_chain(self, k=4):
        """
        Setup the QA chain with retrieval
        k: number of documents to retrieve
        """
        if not self.vectorstore:
            raise ValueError("Vector store not initialized. Run create_vectorstore first.")
        
        # Create custom prompt template
        prompt_template = """Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Keep the answer concise and relevant to the question.

        Context: {context}

        Question: {question}
        Answer:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template, 
            input_variables=["context", "question"]
        )
        
        # Create retrieval QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": k}
            ),
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )
        
        print("🔗 QA Chain setup complete!")
    
    def query(self, question: str, show_sources=True):
        """
        Query the RAG system
        """
        if not self.qa_chain:
            raise ValueError("QA chain not setup. Run setup_qa_chain first.")
        
        print(f"❓ Question: {question}")
        print("🤔 Thinking...")
        
        # Get response
        response = self.qa_chain.invoke({"query": question})
        
        answer = response['result']
        sources = response['source_documents']
        
        print(f"💡 Answer: {answer}")
        
        if show_sources:
            print(f"\n📚 Sources ({len(sources)} documents):")
            for i, doc in enumerate(sources, 1):
                # Get page number if available
                page = doc.metadata.get('page', 'Unknown')
                preview = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                print(f"  {i}. Page {page}: {preview}")
        
        return answer, sources


def main():
    """
    Main function to demonstrate the RAG pipeline
    """
    # Initialize RAG pipeline
    rag = RAGPipeline(model_name="mistral")  # Change to "mistral" if you prefer
    
    # Step 1: Load and process PDF
    pdf_path = "C:\\Users\\mrym\\Desktop\\3dsmax2009.pdf"  # Replace with your PDF path
    
    # Check if we already have a vector store
    if os.path.exists("./chroma_db"):
        print("Found existing vector store, loading...")
        rag.load_existing_vectorstore()
    else:
        print("Creating new vector store...")
        chunks = rag.load_and_process_pdf(pdf_path)
        rag.create_vectorstore(chunks)
    
    # Step 2: Setup QA chain
    rag.setup_qa_chain(k=4)
    
    # Step 3: Interactive querying
    print("\n" + "="*50)
    print("🚀 RAG System Ready! Ask questions about your manual.")
    print("Type 'quit' to exit")
    print("="*50)
    
    while True:
        question = input("\n🔍 Your question: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            break
        
        if not question:
            continue
        
        try:
            answer, sources = rag.query(question)
            print("\n" + "-"*40)
        except Exception as e:
            print(f"❌ Error: {e}")


# Example usage functions
def quick_setup_example():
    """
    Quick setup example for testing
    """
    # Initialize
    rag = RAGPipeline(model_name="llama2")
    
    # Load PDF (replace with your path)
    chunks = rag.load_and_process_pdf("manual.pdf")
    
    # Create vector store
    rag.create_vectorstore(chunks)
    
    # Setup QA
    rag.setup_qa_chain()
    
    # Test queries
    test_questions = [
        "What are the main components?",
        "How do I troubleshoot issues?",
        "What are the safety precautions?"
    ]
    
    for question in test_questions:
        answer, _ = rag.query(question)
        print(f"\nQ: {question}")
        print(f"A: {answer}")


if __name__ == "__main__":
    # Run main interactive loop
    main()
    
    # Or run quick example:
    # quick_setup_example()


"""
SETUP INSTRUCTIONS:

1. Install Ollama:
   - Go to https://ollama.ai/download
   - Download and install for your OS
   
2. Pull the model:
   ollama pull llama2
   # or
   ollama pull mistral

3. Install Python packages:
   pip install langchain chromadb ollama pypdf sentence-transformers

4. Replace "your_manual.pdf" with your actual file path (.pdf or .html)

5. Run the script!

TIPS:
- First run will be slower due to embedding creation
- Subsequent runs load the existing vector store quickly
- Adjust chunk_size and chunk_overlap based on your manual structure
- Try different values of k (number of retrieved docs) for better results
- Use mistral if you want a smaller, faster model
"""