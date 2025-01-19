from dotenv import load_dotenv
import os
from openai import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import openai
import gdown
import zipfile
import streamlit as st
# Streamlit Secrets에서 API 키 불러오기

st_api_key = st.secrets["OPENAI_API_KEY"]

class Rag_callcenter :
    def __init__(self, model="text-embedding-3-large", path='faiss_index_callcenter') :

        if not os.path.exists(path):
            # Replace this with your Google Drive file ID
            file_id = "1oT1j4SYLFSpCxYCN8jGofWVQgSmuWYdw"

            # Build the Google Drive URL
            url = f"https://drive.google.com/uc?id={file_id}"

            # Destination path for the downloaded file
            destination = "faiss_index_callcenter.zip"

            # Download the file
            gdown.download(url, destination, quiet=False)

            # Path to the .zip file
            zip_file_path = "faiss_index_callcenter.zip"

            # Destination folder
            destination_folder = "faiss_index_callcenter"

            # Ensure the destination folder exists
            os.makedirs(destination_folder, exist_ok=True)

            # Extract the .zip file
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                zip_ref.extractall(destination_folder)

        # load_dotenv()  # important
        # self.api_key = os.getenv("OPENAI_API_KEY")

        self.client = OpenAI(api_key=st_api_key)  # this is also the default, it can be omitted
        self.embedding_model = OpenAIEmbeddings(openai_api_key=st_api_key, model=model)

        self.vectorstore = FAISS.load_local(path, embeddings=self.embedding_model, allow_dangerous_deserialization=True)
        print(f"Basic model is {model}")

    def rag_output(self, query, k=2, model='gpt-4o') :
        self.query = query
        self.retrieved_docs = self.vectorstore.similarity_search(self.query, k=k)
        context = ""
        for doc in self.retrieved_docs:
            context += f"질문: {doc.page_content}\n답변: {doc.metadata['answer']}\n"
            context += f"출처: {doc.metadata['source']}\n\n"

        self.prompt = f"""
아래는 사용자가 요청한 정보와 관련된 문서 내용입니다:
\n{context}    
사용자의 질문에 따라 응답을 생성하세요: "{query}"
        """
        messages = [
        {"role": "user", "content": f"""{self.prompt}"""}
        ]

        response = openai.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=4096,
            n=1,
            stop=None,
            temperature=0.7  # model randomness
        )
        self.generated_response = response.choices[0].message.content
        print("Generated Answer:")
        print(self.generated_response)
        print("\nSources:")
        source_list = []
        for i, doc in enumerate(self.retrieved_docs):
            print(f"{i+1}. {doc.metadata['source']}")
            source_list.append(f"{i+1}. {doc.metadata['source']}\n  질문: {doc.page_content}\n  답변: {doc.metadata['answer']}\n")
        self.sources = '\n'.join(source_list)
        self.final_output = self.generated_response + '\n\n'+ 'Sources:' +'\n\n' + self.sources
        return self.final_output

rag = Rag_callcenter()


st.title("RAG-Based 전북 콜센터 Generator")

query = st.text_input("Enter your query:")

if st.button("Generate"):
    final_output = rag.rag_output(query=query, k=1)  # k는 참조할 사례 갯수
    st.text_area("전북콜센터 상담:", value=final_output, height=300)