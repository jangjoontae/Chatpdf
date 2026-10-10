__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain_classic.callbacks.base import BaseCallbackHandler
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.chat_models import init_chat_model
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langsmith import Client
import tempfile
import os
from dotenv import load_dotenv


load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
# api_key = st.text_input("OPEN_API_KEY", type="password")

st.title(":globe_with_meridians: ChatPDF 💯")
st.write("---") 

uploaded_file = st.file_uploader("PDF 파일을 올려주세요", type="pdf")

def pdf_to_document(uploaded_file):
  temp_dir = tempfile.TemporaryDirectory() # 임시폴더 생성
  temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
  with open(temp_filepath, "wb") as f:
    f.write(uploaded_file.getvalue())
  loader = PyPDFLoader(temp_filepath)#임시폴더에서 업로드된 pdf로딩
  pages = loader.load_and_split()
  return pages

if uploaded_file is not None:
  pages = pdf_to_document(uploaded_file)
  # 텍스트 분할
  text_splitter = RecursiveCharacterTextSplitter(
    #Set a really small chuck size, just to show
    chunk_size = 100,
    chunk_overlap = 20,
    length_function=len,
    is_separator_regex=False,
  )

  texts = text_splitter.split_documents(pages)

  # 임베딩 후 벡터디비 저장
  embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large", api_key=api_key)
  db = Chroma.from_documents(texts, embeddings_model)

  # 캐싱 문제 발생시 아래 코드를 2줄을 주석해제 하여서 사용(배포시)
  import chromadb
  chromadb.api.client.SharedSystemClient.clear_system_cache()

  # 스트리밍 처리할 Handler 생성
  class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text=initial_text
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text+=token
        self.container.markdown(self.text)

  # user input
  st.header('PDF에게 질문하세요.')
  question = st.text_input('질문을 입력하세요')
  if st.button('질문하기'):
    with st.spinner('Wait for it...'):
      # 검색 기능 구현
      llm = init_chat_model("gpt-5.5", api_key=api_key)
      retriever_from_llm = MultiQueryRetriever.from_llm(
        retriever = db.as_retriever(), llm=llm
      )

      client = Client()
      prompt=client.pull_prompt("rlm/rag-prompt",dangerously_pull_public_prompt=True)

      # 검색 결과 format
      def format_docs(docs):
        return '\n\n'.join(doc.page_content for doc in docs)

      # 출력공간 확보 stream 부분
      chat_box = st.empty()
      stream_handler = StreamHandler(chat_box)
          
      generate_llm = init_chat_model(model="gpt-5.5", openai_api_key=api_key, streaming=True, callbacks=[stream_handler])

      rag_chain = (
        {'context':retriever_from_llm|format_docs, "question":RunnablePassthrough()} 
        | prompt
        | generate_llm
        | StrOutputParser()
      )

      # 실행
      result = rag_chain.invoke(question)
      # st.write(result)

  # 검색 기능 구현
  # llm = init_chat_model(model="gpt-4o-mini", api_key=api_key)

# Chroma 벡터 저장소에 대한 Retriever 인스턴스 생성
  # retriever_from_llm = MultiQueryRetriever.from_llm(
  #   retriever = db.as_retriever(), llm=llm
  # )


  # # 프롬프트, LLM 체인 실행
  # question = "주인공은 왜 바로 집에 가지 않고 술을 처먹고 있었을까?"
  # docs=retriever_from_llm.invoke(question)
