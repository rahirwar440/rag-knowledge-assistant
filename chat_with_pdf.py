import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# Step 1: Load API key from .env file
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

print("Step 1: Loading PDF...")
loader = PyPDFLoader("test.pdf")
documents = loader.load()
print(f"Loaded {len(documents)} pages")

print("Step 2: Splitting into chunks...")
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = splitter.split_documents(documents)
print(f"Created {len(chunks)} chunks")

print("Step 3: Creating embeddings (this may take a minute the first time)...")
embeddings = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
    huggingfacehub_api_token=os.getenv("HF_TOKEN")
)

print("Step 4: Storing in ChromaDB...")
vectorstore = Chroma.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

print("Step 5: Setting up LLM (Groq)...")
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-8b-instant")

prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the following context. If the answer isn't in the context, say you don't know.

Context:
{context}

Question: {question}
""")

print("\n✅ Setup complete! Ab tum sawaal pooch sakte ho (exit likhke ruk sakte ho)\n")

while True:
    query = input("Tumhara sawaal: ")
    if query.lower() == "exit":
        break

    # Retrieve relevant chunks
    docs = retriever.invoke(query)
    context = "\n\n".join(doc.page_content for doc in docs)

    # Ask LLM
    messages = prompt.format_messages(context=context, question=query)
    response = llm.invoke(messages)

    print("\nAnswer:", response.content)
    print("\n--- Sources ---")
    for doc in docs:
        print(f"Page {doc.metadata.get('page', 'unknown')}: {doc.page_content[:100]}...")
    print()