from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

from models import check_if_model_is_available
from document_loader import load_documents
import argparse
import sys

from llm import getChatChain

from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager

# app = FastAPI()


def get_model(question: str) -> None:
    """
    First we pull the embeddings and the llm model. Them we create a vector database using Chromadb.

    Args:
        llm_model_name (str): llm model
        embedding_model_name (str): embedding model (generate vector embeddings!)
        documents_path (str): input document

    Raises:
        answer
    """
    # Check to see if the models available, if not attempt to pull them
    try:
        check_if_model_is_available(args.model)
        check_if_model_is_available(args.embedding_model)
    except Exception as e:
        print(e)
        sys.exit()

    # Creating database form documents
    try:
        db = load_documents_into_database(args.embedding_model, args.path)
    except FileNotFoundError as e:
        print(e)
        sys.exit()

    llm = Ollama(model=args.model)
    chat = getChatChain(llm, db)

    answer = chat(question)

    return answer['answer']

model = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # question = payload.question
    model['answer'] = get_model
    yield
    model.clear()


app = FastAPI(lifespan=lifespan)


TEXT_SPLITTER = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

class FeaturesModel(BaseModel):
    question: str


@app.post("/ask_question/")
async def send_inference(payload: FeaturesModel):
    question = payload.question
    # response = inference(args.model, args.embedding_model, args.path, question)
    response = model["answer"](question)

    return {
        "msg": "result",
        "answer": response
    }


def load_documents_into_database(model_name: str, documents_path: str) -> Chroma:
    """
    Loads documents from the specified directory into the Chroma database
    after splitting the text into chunks.

    Returns:
        Chroma: The Chroma database with loaded documents.
    """

    print("Loading documents")
    raw_documents = load_documents(documents_path)
    documents = TEXT_SPLITTER.split_documents(raw_documents)

    print("Creating embeddings and loading documents into Chroma")
    db = Chroma.from_documents(
        documents,
        OllamaEmbeddings(model=model_name),
    )
    return db


def inference(llm_model_name: str, embedding_model_name: str, documents_path: str, question) -> None:
    """
    First we pull the embeddings and the llm model. Them we create a vector database using Chromadb.

    Args:
        llm_model_name (str): llm model
        embedding_model_name (str): embedding model (generate vector embeddings!)
        documents_path (str): input document

    Raises:
        answer
    """
    # Check to see if the models available, if not attempt to pull them
    try:
        check_if_model_is_available(llm_model_name)
        check_if_model_is_available(embedding_model_name)
    except Exception as e:
        print(e)
        sys.exit()

    # Creating database form documents
    try:
        db = load_documents_into_database(embedding_model_name, documents_path)
    except FileNotFoundError as e:
        print(e)
        sys.exit()

    llm = Ollama(model=llm_model_name)
    chat = getChatChain(llm, db)

    answer = chat(question)

    return answer['answer']


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local LLM with RAG with Ollama.")
    parser.add_argument(
        "-m",
        "--model",
        # default="mistral",
        default="stablelm2",
        help="LLM model to use.",
    )
    parser.add_argument(
        "-e",
        "--embedding_model",
        # default="nomic-embed-text",
        default="mxbai-embed-large",
        help="Embedding model to use.",
    )
    parser.add_argument(
        "-p",
        "--path",
        default="data",
        help="Directory containing documents to load.",
    )
    return parser.parse_args()


if __name__ == "__main__":   
    import uvicorn
    args = parse_arguments()

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000        
    )
