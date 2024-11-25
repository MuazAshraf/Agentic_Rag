from flask import Flask, request, jsonify
from uuid import uuid4
from dotenv import load_dotenv
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from operator import itemgetter
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.tools.ddg_search import DuckDuckGoSearchRun
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langgraph.prebuilt import ToolExecutor
from langchain_core.utils.function_calling import convert_to_openai_function
from typing import TypedDict, Annotated, Sequence
import operator
from langchain_core.messages import BaseMessage
from langgraph.prebuilt import ToolInvocation
import json
from langchain_core.messages import FunctionMessage
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import ArxivLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.tools import tool
from pydantic import BaseModel, Field
# Initialize Flask app
app = Flask(__name__)

# Load environment variables (if any)
from dotenv import load_dotenv
load_dotenv('.env')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
LANGCHAIN_API_KEY = os.getenv('LANGCHAIN_API_KEY')
LANGCHAIN_TRACING_V2 = os.getenv('LANGCHAIN_TRACING_V2')
LANGCHAIN_PROJECT = os.getenv('LANGCHAIN_PROJECT')

# Load and split documents for RAG (Arxiv)
docs = ArxivLoader(query="RAG",load_max_docs=2,).load()
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=350, chunk_overlap=50)
chunked_documents = text_splitter.split_documents(docs)

# Instantiate the Embedding Model and FAISS index
embeddings = OpenAIEmbeddings()
faiss_vectorstore = FAISS.from_documents(documents=chunked_documents, embedding=embeddings)
retriever = faiss_vectorstore.as_retriever()


# Create the RAG prompt template
RAG_PROMPT = """\
Use the following context to answer the user's query. If you cannot answer the question, please respond with 'I don't know'.

Question:
{question}

Context:
{context}
"""
rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT)

# Instantiate OpenAI model
openai_chat_model = ChatOpenAI(model="gpt-4o-mini")

# Build the RAG chain
retrieval_augmented_generation_chain = (
    {"context": itemgetter("question") | retriever, "question": itemgetter("question")}
    | RunnablePassthrough.assign(context=itemgetter("context"))
    | {"response": rag_prompt | openai_chat_model, "context": itemgetter("context")}
)


class RAGQueryInput(BaseModel):
    query: str = Field(description="The question to be answered about the docs it has stored in his memory")

@tool("rag_query", args_schema=RAGQueryInput)
def rag_query(query: str) -> dict:
    """Useful for answering questions about the docs it has stored in his memory."""
    result = retrieval_augmented_generation_chain.invoke({"question": query})
    return {
        "response": result["response"].content,
        "context": result.get("context", "No context found")
    }

# Set up ToolExecutor with the RAG tool
duckduckgo_tool = DuckDuckGoSearchRun(vervose=True)
arxiv_tool = ArxivQueryRun(description="Retrieve academic papers and information from Arxiv.",verbose = True)

tool_belt = [rag_query, duckduckgo_tool, arxiv_tool]
tool_executor = ToolExecutor(tool_belt)

functions = [convert_to_openai_function(t) for t in tool_belt]
model = openai_chat_model.bind_functions(functions)

# Define Flask endpoint to handle query requests
@app.route('/ask', methods=['POST'])
def ask():
    # Get JSON request
    request_data = request.get_json()

    if not request_data or 'question' not in request_data:
        return jsonify({"error": "No question provided"}), 400

    question = request_data['question']

    # Set up LangGraph for decision-making
    class AgentState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], operator.add]
        tool_used: str

    def call_model(state):
        messages = state["messages"]
        response = model.invoke(messages)
        return {"messages": [response], "tool_used": state.get("tool_used", "None")}

    def call_tool(state):
        last_message = state["messages"][-1]

        action = ToolInvocation(
            tool=last_message.additional_kwargs["function_call"]["name"],
            tool_input=json.loads(last_message.additional_kwargs["function_call"]["arguments"])
        )
        
        tool_name = action.tool
        print("Tool Used: ", tool_name)

        response = tool_executor.invoke(action)
        print("Response: ", response)
        function_message = FunctionMessage(content=str(response), name=action.tool)

        return {"messages": [function_message], "tool_used": tool_name}

    # Build the LangGraph workflow
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", call_model)
    workflow.add_node("action", call_tool)

    # Set the entry point
    workflow.set_entry_point("agent")

    # Build a conditional edge for routing
    def should_continue(state):
        last_message = state["messages"][-1]

        if "function_call" not in last_message.additional_kwargs:
            return "end"

        return "continue"

    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue": "action",
            "end": END
        }
    )

    # Connect the conditional edge to the agent and action nodes
    workflow.add_edge("action", "agent")

    # Compile the workflow
    app = workflow.compile()

    # Call the LangGraph to process the query
    inputs = {
        "messages": [HumanMessage(content=question)],
        "tool_used": "None"
    }
    response = app.invoke(inputs)

    # Return the response as JSON
    return jsonify({
        "question": question,
        "response": response['messages'][-1].content,
        "tool_used": response.get('tool_used', "None")
    })

if __name__ == '__main__':
    app.run(debug=True)
