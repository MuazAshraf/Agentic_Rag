from dotenv import load_dotenv
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from operator import itemgetter
from langchain.schema.runnable import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import ArxivLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
load_dotenv('.env')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Load the document pertaining to a particular topic
docs = ArxivLoader(query="Retrieval Augmented Generation", load_max_docs=5).load()

# Split the dpocument into smaller chunks
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=350, chunk_overlap=50
)

chunked_documents = text_splitter.split_documents(docs)

# Instantiate the Embedding Model
embeddings = OpenAIEmbeddings()
# Create Index- Load document chunks into the vectorstore
faiss_vectorstore = FAISS.from_documents(
    documents=chunked_documents,
    embedding=embeddings,
)
# Create a retriver
retriever = faiss_vectorstore.as_retriever()

#Generate a Rag Prompt
RAG_PROMPT = """\
Use the following context to answer the user's query. If you cannot answer the question, please respond with 'I don't know'.

Question:
{question}

Context:
{context}
"""

rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT)

#Instantiate the LLM
openai_chat_model = ChatOpenAI(model="gpt-4o-mini")

#Build LCEL RAG Chain
retrieval_augmented_generation_chain = (
       {"context": itemgetter("question") 
    | retriever, "question": itemgetter("question")}
    | RunnablePassthrough.assign(context=itemgetter("context"))
    | {"response": rag_prompt | openai_chat_model, "context": itemgetter("context")}
)
#
retrieval_augmented_generation_chain

#Ask Query
retrieval_augmented_generation_chain.invoke({"question" : "What is Retrieval Augmented Generation?"})
