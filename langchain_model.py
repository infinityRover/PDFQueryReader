!pip install langchain
!pip install openai
!pip install PyPDF2
!pip install faiss-cpu
!pip install tiktoken
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS

import os 
os.environ["OPENAI_API_KEY"]="Your API Key"

from typing_extensions import Concatenate

# read text from pdf
raw_text=''
for i, page in enumerate(pdfreader.pages):
    content=page.extract_text()
    if content:
        raw_text+=content

# we need to split the text using charcter text split such that it should not increase the size token
text_splitter=CharacterTextSplitter(
    separator='\n',
    chunk_size=800,
    chunk_overlap=200,
    length_function=len,
)
texts=text_splitter.split_text(raw_text)


# download embeddings from OpenAI
# Embeddings- OpenAI’s text embeddings measure the relatedness of text strings.
embeddings=OpenAIEmbeddings()

document_search=FAISS.from_texts(texts, embeddings)

from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

query="What are the policies related to Financial Sector Regulations"
docs=document_search.similarity_search(query)
chain.run(input_documents=docs, question=query)
