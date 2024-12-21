# Import Library
import pandas as pd
import os
import tiktoken
from dotenv import load_dotenv
import fastapi
from langchain.vectorstores import FAISS

import uuid
import io
import shutil
import uvicorn
import glob
import time
import json
import openai
import numpy as np

from tqdm.auto import tqdm
from fastapi import FastAPI, Form, UploadFile, File, HTTPException, Query, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, StreamingResponse, JSONResponse, RedirectResponse
from langchain_core.prompts import PromptTemplate
from langchain import hub

from langchain.document_loaders import DirectoryLoader, TextLoader, JSONLoader, DataFrameLoader
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.chains import LLMChain, RetrievalQA
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory


from pydantic import BaseModel, Field, ValidationError


app = FastAPI(
    title="LangChain Server",
    version="o1",
    description="",
)
# Set all CORS enabled origins
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
langchain_tracing = os.getenv("LANGCHAIN_TRACING_V2")
langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
langchain_project = os.getenv("LANGCHAIN_PROJECT")
langchain_endpoint = os.getenv("LANGCHAIN_ENDPOINT")

if openai_api_key:
    os.environ["OPENAI_API_KEY"] = openai_api_key
if langchain_tracing:
    os.environ["LANGCHAIN_TRACING_V2"] = langchain_tracing
if langchain_api_key:
    os.environ["LANGCHAIN_API_KEY"] = langchain_api_key  
if langchain_project:
    os.environ["LANGCHAIN_PROJECT"] = langchain_project 
if langchain_endpoint:
    os.environ["LANGCHAIN_ENDPOINT"] = langchain_endpoint                  

# print(f"OpenAI API Key: {os.environ.get('OPENAI_API_KEY')}")
# print(f"Tracing Enabled: {langchain_tracing}")
# print(f"LangChain API Key: {langchain_api_key}")

df = pd.read_csv('bq-results-20240205-004748-1707094090486.csv').head(2000)

# Combine
df['combined_info'] = df.apply(lambda row: f"Order time: {row['created_at']}. Customer Name: {row['name']}. Product Department: {row['product_department']}. Product: {row['product_name']}. Category : {row['product_category']}. Price: ${row['sale_price']}. Stock quantity: {row['stock_quantity']}", axis=1)

# Load Processed Dataset
loader = DataFrameLoader(df, page_content_column="combined_info")
docs  = loader.load()

# Document splitting
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(docs)

# embeddings model
embeddings = OpenAIEmbeddings(
        model="text-embedding-ada-002")
# Vector DB
vectorstore  = FAISS.from_documents(texts, embeddings)

# Prompt Engineering
manual_template = """ 
Kindly suggest three similar products based on the description I have provided below:

Product Department: {department},
Product Category: {category},
Product Brand: {brand},
Maximum Price range: {price}.

Please provide complete answers including product department name, product category, product name, price, and stock quantity.
"""
prompt_manual = PromptTemplate(
    input_variables=["department","category","brand","price"],
    template=manual_template,
)

llm = ChatOpenAI(openai_api_key=openai_api_key,model_name='gpt-3.5-turbo', temperature=0)

chain = LLMChain(
    llm=llm,
    prompt = prompt_manual,
    verbose=True)

# Prompt Engineering
chatbot_template = """ 
You are a friendly, conversational retail shopping assistant that help customers to find product that match their preferences. 
From the following context and chat history, assist customers in finding what they are looking for based on their input. 
For each question, suggest three products, including their category, price and current stock quantity.
Sort the answer by the cheapest product.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

chat history: {history}

input: {question} 
Your Response:
"""
chatbot_prompt = PromptTemplate(
    input_variables=["context","history","question"],
    template=chatbot_template,
)

# Creating the LangChain conversational chain
memory = ConversationBufferMemory(memory_key="history", input_key="question", return_messages=True)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type='stuff',
    retriever=vectorstore.as_retriever(),
    verbose=True,
    chain_type_kwargs={
        "verbose": True,
        "prompt": chatbot_prompt,
        "memory": memory}
)
