from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory, ConversationSummaryBufferMemory
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.prompts.prompt import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_KEY = os.getenv("apikey")

template = """
  You are an Medical Assistant. Conversation between a human and an AI Assistant and related context are given. use context and also your data alsu provide some of usefull links. If question is not related to medical , just say that "I cannot Assist with that! It's not related for Medical. ", also i need long paragpraphs and include more data.
  related data provide in "CONTEXT:" all cases need to prevention methods and how prevent in you say
  ANSWER TEMPLATE:
    [Answer]
  CONTEXT:
  {context}

  QUESTION: 
  {question}

  CHAT HISTORY:
  {chat_history}

  ANSWER:
  """

prompt = PromptTemplate(input_variables=["chat_history", "question", "context"], template=template)

# define embedding
embeddings = OpenAIEmbeddings(
    openai_api_key=OPENAI_KEY
)
# define memory
memory = ConversationBufferMemory(memory_key="chat_history", ai_prefix="AI Lawyer", return_messages=True)

openai = OpenAI(temperature=0.8, openai_api_key=OPENAI_KEY)
# memory = ConversationSummaryBufferMemory(llm=openai, max_token_limit=1000)
# db3 = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
faiss_db = FAISS.load_local("faiss_index", embeddings)

# define chain
chat_llm = ConversationalRetrievalChain.from_llm(openai, faiss_db.as_retriever(search_kwargs={"k": 8}), memory=memory,combine_docs_chain_kwargs={"prompt": prompt}, verbose=True)


def create_db(file):
    # load documents
    loader = PyPDFLoader(file)
    documents = loader.load()
    documents = documents[:16]
    # split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=80)
    docs = text_splitter.split_documents(documents)
    db = FAISS.from_documents(docs, embeddings)
    db.save_local("faiss_index")
    # vectordb.persist()


def get_chat_history():
    return memory.load_memory_variables({})
    # return ConversationBufferMemory(memory_key="chat_history", return_messages=True)


def memory_clear():
    memory.clear()
    return "New Chat Created"


def chat(question):
    chat_history = get_chat_history()

    res = chat_llm({"question": question, "chat_history": chat_history})
    # n=0
    # memory.clear()
    memory.save_context({"input": question}, {"output": res['answer']})
    return res['answer']

create_db("medi.pdf")

