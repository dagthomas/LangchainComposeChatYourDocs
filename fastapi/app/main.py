import threading
import queue
import openai
import aiofiles
import langchain
from dotenv import load_dotenv
from pydantic import BaseModel
from qdrant_client import QdrantClient
import logging
from typing import List
import urllib3
import os
import tempfile
import sys
import magic
import pandas as pd
import typing as t
from slugify import slugify

# custom
import submodules.prompts as prompts
from langchain.schema import HumanMessage, SystemMessage

from langchain.document_loaders.base import Document
from langchain.document_loaders import ApifyDatasetLoader
from langchain.cache import InMemoryCache
from langchain.document_loaders import WebBaseLoader
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders import SRTLoader
from langchain.document_loaders import UnstructuredWordDocumentLoader
from langchain.document_loaders import UnstructuredEPubLoader
from langchain.document_loaders import UnstructuredPowerPointLoader
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Qdrant
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.callbacks.base import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File
from fastapi.security.http import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.responses import StreamingResponse
from fastapi.security import OAuth2PasswordBearer
from fastapi.middleware.cors import CORSMiddleware


langchain.llm_cache = InMemoryCache()


load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AUTHORIZED_API_KEY = os.getenv("AUTHORIZED_API_KEY")
os.environ["APIFY_API_TOKEN"] = os.getenv("APIFY_API_TOKEN")
openai.api_key = OPENAI_API_KEY

host = "qdrant"
client = QdrantClient(host=host, prefer_grpc=True)

http = urllib3.PoolManager(cert_reqs="CERT_NONE", retries=False)
logging.captureWarnings(True)
get_bearer_token = HTTPBearer(auto_error=False)
# def bearer token auth
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


class UnauthorizedMessage(BaseModel):
    detail: str = "Bearer token missing or unknown"


known_tokens = set([AUTHORIZED_API_KEY])


async def get_token(
    auth: t.Optional[HTTPAuthorizationCredentials] = Depends(get_bearer_token),
) -> str:
    # Simulate a database query to find a known token
    if auth is None or (token := auth.credentials) not in known_tokens:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=UnauthorizedMessage().detail,
        )
    return token


# start app
app = FastAPI(
    title="LangChain Starter API",
)

origins = [
    "*",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def startup():
    print("Server Startup!")


class GPTQuery(BaseModel):
    prompt: str
    system_intel: str
    temperature: float


class Query(BaseModel):
    query: str
    collection: str


class Apify(BaseModel):
    dataset: str
    collection: str


class Collection(BaseModel):
    collection: str
    prompt: str
    temperature: float


class Webpage(BaseModel):
    url: str


class Webpages(BaseModel):
    urls: List[str]
    collection_name: str


@app.get("/")
async def read_root():
    message = f"Hello world! From FastAPI running on Uvicorn with Gunicorn. Using Python {sys.version_info.major}.{sys.version_info.minor}"
    return {message}


# Fastapi endpoint for returning a list of collections


@app.get("/collections")
async def read_collections(token: str = Depends(get_token)):
    data = client.get_collections()
    return data.collections


@app.post("/documents")
async def create_item(item: Query, token: str = Depends(get_token)):
    qdrant = Qdrant(
        client, item.collection, embedding_function=OpenAIEmbeddings().embed_query
    )
    docs = qdrant.similarity_search_with_score(item.query)
    return docs


@app.post("/collections")
async def create_item(item: Collection, token: str = Depends(get_token)):
    qdrant = Qdrant(
        client, item.collection, embedding_function=OpenAIEmbeddings().embed_query
    )
    docs = qdrant.similarity_search(item.prompt)
    llm = ChatOpenAI(temperature=item.temperature, model_name="gpt-3.5-turbo")
    #  , metadata_keys=['source']
    chain = load_qa_with_sources_chain(llm, chain_type="stuff")
    result = chain(
        {"input_documents": docs, "question": item.prompt}, return_only_outputs=True
    )
    return result


loader_classes = {
    ".pdf": PyPDFLoader,
    ".xls": CSVLoader,
    ".xlsx": CSVLoader,
    ".csv": CSVLoader,
    ".epub": UnstructuredEPubLoader,
    ".pptx": UnstructuredPowerPointLoader,
    ".docx": UnstructuredWordDocumentLoader,
    ".txt": SRTLoader,
    ".srt": SRTLoader,
}


async def ingest_data(tmp_file, slug, file_type, chunk_size, chunk_overlap):
    if file_type in loader_classes:
        loader_class = loader_classes[file_type]
        if loader_class == CSVLoader:
            excel = pd.read_excel(tmp_file)
            excel.to_csv(
                f"./files/{AUTHORIZED_API_KEY}/{slug}.csv", index=None, header=True
            )
        loader = loader_class(tmp_file)
    else:
        return "Filetype not supported"
    documents = loader.load()
    # cache the embeddings
    if not hasattr(ingest_data, "embeddings"):
        ingest_data.embeddings = OpenAIEmbeddings()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    print(documents)
    docs = text_splitter.split_documents(documents)
    Qdrant.from_documents(
        docs, ingest_data.embeddings, host=host, collection_name=slug, prefer_grpc=True
    )
    return slug


@app.post("/upload")
async def upload_file(file: UploadFile = File(...), token: str = Depends(get_token)):
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(await file.read())
        tmp_file_path = tmp_file.name
    filetype = os.path.splitext(file.filename)
    response = await ingest_data(
        tmp_file_path,
        slugify(os.path.splitext(file.filename)[0]),
        filetype[1],
        chunk_size=256,
        chunk_overlap=40,
    )
    tmp_file.close()
    os.unlink(tmp_file.name)
    return response


@app.post("/webpage")
async def create_webpage(item: Webpage, token: str = Depends(get_token)):
    collection_name = slugify(item.url.split("/")[-1])
    loader = WebBaseLoader(item.url)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=40)
    docs = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    Qdrant.from_documents(
        docs, embeddings, host=host, collection_name=collection_name, prefer_grpc=True
    )
    return collection_name


@app.post("/webpages")
async def create_webpages(item: Webpages, token: str = Depends(get_token)):
    loader = WebBaseLoader(item.urls)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=40)
    docs = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    Qdrant.from_documents(
        docs,
        embeddings,
        host=host,
        collection_name=item.collection_name,
        prefer_grpc=True,
    )

    return item.collection_name


@app.post("/openai")
async def openai_query(item: GPTQuery, token: str = Depends(get_token)):
    system_intel = item.system_intel
    prompt = item.prompt
    result = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        temperature=item.temperature,
        messages=[
            {"role": "system", "content": system_intel},
            {"role": "user", "content": prompt},
        ],
    )

    return result.choices[0].message.content


@app.post("/apify")
async def stream(item: Apify, token: str = Depends(get_token)):
    loader = ApifyDatasetLoader(
        dataset_id=item.dataset,
        dataset_mapping_function=lambda dataset_item: Document(
            page_content=dataset_item["aml_text"],
            metadata={
                "source": dataset_item["url"],
                "title": dataset_item["aml_title"],
                "paragraph": dataset_item["paragraph"],
            },
        ),
    )
    embeddings = OpenAIEmbeddings()
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function = len)
    n = 50
    final = [
        documents[i * n : (i + 1) * n] for i in range((len(documents) + n - 1) // n)
    ]
    docs = text_splitter.split_documents(final[0])

    Qdrant.from_documents(docs, embeddings, host=host, collection_name=item.collection)
    for docs in final[1:]:
        print("iterate")
        doc = text_splitter.split_documents(docs)
        if hasattr(doc, "page_content"):
            print("has")
            Qdrant.add_documents(
                doc, embeddings, host=host, collection_name=item.collection
            )


class ThreadedGenerator:
    def __init__(self):
        self.queue = queue.Queue()

    def __iter__(self):
        return self

    def __next__(self):
        item = self.queue.get()
        if item is StopIteration:
            raise item
        return item

    def send(self, data):
        self.queue.put(data)

    def close(self):
        self.queue.put(StopIteration)


class ChainStreamHandler(StreamingStdOutCallbackHandler):
    def __init__(self, gen):
        super().__init__()
        self.gen = gen

    def on_llm_new_token(self, token: str, **kwargs):
        self.gen.send(token)


def llm_thread(g, prompt, system_intel, temperature):
    try:
        chat = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            verbose=True,
            streaming=True,
            callback_manager=CallbackManager([ChainStreamHandler(g)]),
            temperature=temperature,
        )

        chat([SystemMessage(content=system_intel), HumanMessage(content=prompt)])

    finally:
        g.close()


def chat(prompt, system_intel, temperature):
    g = ThreadedGenerator()
    threading.Thread(
        target=llm_thread, args=(g, prompt, system_intel, temperature)
    ).start()
    return g


@app.post("/openai/stream")
async def stream(item: GPTQuery, token: str = Depends(get_token)):
    return StreamingResponse(
        chat(item.prompt, item.system_intel, item.temperature),
        media_type="text/event-stream",
    )


@app.post("/collections/stream")
async def stream(item: Collection, token: str = Depends(get_token)):
    qdrant = Qdrant(
        client, item.collection, embedding_function=OpenAIEmbeddings().embed_query
    )
    retriever = qdrant.as_retriever(search_type="similarity")
    query = item.prompt
    relevant_docs = retriever.get_relevant_documents(query)
    # docs = qdrant.similarity_search(item.prompt)
    template = prompts.documentSearch(item.prompt, relevant_docs)
    return StreamingResponse(
        chat(item.prompt, template, item.temperature), media_type="text/event-stream"
    )
