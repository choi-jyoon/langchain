from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain.chains import create_retrieval_chain

load_dotenv()

openai_api_key = os.getenv('OPENAI_API_KEY')

llm = ChatOpenAI(api_key=openai_api_key)
output = llm.invoke("2024년 청년지원정책을 알려줘")


prompt = ChatPromptTemplate.from_messages(
    [
        ('system', '너는 청년을 행복하게 하기 위한 정부정책 안내 컨설턴트야'),
        ('user', '{input}')
    ]
)

chain = prompt | llm    # prompt와 llm 연결 

print(chain.invoke({'input': "2024년 청년지원정책을 알려줘"}))


# 크롤링
loader = WebBaseLoader("https://www.moel.go.kr/policy/policyinfo/support/list4.do")

docs = loader.load()

# 임베딩
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)


text_splitter = RecursiveCharacterTextSplitter()    # 텍스트 문장단위로 분할
documents = text_splitter.split_documents(docs)
# FAISS를 사용하여 문장들을 임베딩하고 검색을 위한 벡터로 변환
vector = FAISS.from_documents(documents, embeddings)

prompt = ChatPromptTemplate.from_template('''Answer the following question based only on the provided context:

<context>
{context}
</context>
          
Question: {input}''')

document_chain = create_stuff_documents_chain(llm, prompt)

print(document_chain.invoke({
    "input":"국민취업지원제도가 뭐야",
    "context":[Document(page_content="""국민 취업지원제도란?
    취업을 원하는 사람에게 취업지원서비스를 일괄적으로 제공하고 저소득 구직자에게는 최소한의 소득도 지원하는 한국형 실업부조입니다. 2024년부터 15~69세 저소득층, 청년 등 취업취약계층에게 맞춤형 취업지원서비스와 소득지원을 함께 제공합니다.
    [출처] 2024년 달라지는 청년 지원 정책을 확인하세요.|작성자 정부24""")]
}))

retriver = vector.as_retriever()
retriver_chain = create_retrieval_chain(retriver, document_chain)

response = retriver_chain.invoke({"input": "상담센터 전화번호 뭐야?"})
print(response["answer"])
