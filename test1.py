##(python 12 )

import data as st 
import fitz  # PyMuPDF
from langchain.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.messages import HumanMessage, AIMessage
from langchain.prompts import MessagesPlaceholder
import sqlite3


# Streamlit UI
st.title("PDF Query with LangChain")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    # Save the uploaded file temporarily
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())
    
    # Load and process the PDF with PyMuPDF
    def load_pdf(file_path):
        doc = fitz.open(file_path)
        text = ""
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text += page.get_text()
        return text
    
    pdf_text = load_pdf("temp.pdf")
    # print(pdf_text)
    
    # Split and chunk the text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
    chunks = text_splitter.create_documents(pdf_text)
    
    # Add to vector database
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=OllamaEmbeddings(model="nomic-embed-text", show_progress=True),
        collection_name="local-rag"
    )
    
    # Set up the LLM and retriever
    local_model = "llama3"
    llm = ChatOllama(model=local_model)
    
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI language model assistant. Your task is to generate five
        different versions of the given user question to retrieve relevant documents from
        a vector database. By generating multiple perspectives on the user question, your
        goal is to help the user overcome some of the limitations of the distance-based
        similarity search. Provide these alternative questions separated by newlines.
        Original question: {question}"""
    )
    
    retriever = MultiQueryRetriever.from_llm(
        vector_db.as_retriever(),
        llm,
        prompt=QUERY_PROMPT
    )
    
    # Define the RAG prompt
    template = """Answer the question based ONLY on the following context:
    {context}
    Question: {question}
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    chat_history = []

    # Chat history prompt template
    chat_prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an AI named Mike, you answer questions with simple answers and no funny stuff.",
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )

    question = st.text_input("Enter your question:")
    
    if question:
        response = chain.invoke({"input": question, "chat_history": chat_history})
        chat_history.append(HumanMessage(content=question))
        chat_history.append(AIMessage(content=response))

        st.write(response)