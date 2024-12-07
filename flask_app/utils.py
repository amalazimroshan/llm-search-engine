import os
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
import re
import json
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema import StrOutputParser
import pprint

load_dotenv()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
SERPER_API_KEY = os.environ.get("SERPER_API_KEY")
OPENAI_API_KEY = None
os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

gemini_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

def search_articles(query,num_results):
    """
    Searches for articles related to the query using Serper API.
    Returns a list of dictionaries containing article URLs, headings, and text.
    """
    # implement the search logic - retrieves articles
    url = "https://google.serper.dev/search"

    payload = json.dumps({
      "q": query,
      "num":num_results
    })
    headers = {
      'X-API-KEY': SERPER_API_KEY,
      'Content-Type': 'application/json'
    }
    response = requests.request("POST", url, headers=headers, data=payload)
    return response.json().get('organic',[])

    # print(json_data)
    # knowledge = json_data.get('knowledgeGraph', None)
    # answerbox = json_data.get('answerBox',None)
    # articles = json_data.get('organic',[])
    # related_searches = json_data.get('relatedSearches',[])
    # people_ask_for = json_data.get('peopleAlsoAsk', [])
    # return knowledge, answerbox, articles, related_searches, people_ask_for


def fetch_article_content(url):
    """
    Fetches the article content, extracting headings and text.
    """
    # implementation of fetching headings and content from the articles
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')        
        for script in soup(["script","style"]):
            script.decompose()
        content = soup.get_text(separator=' ', strip=True)
        content = ' '.join(content.split())
    except Exception as e:
        print(f"Error scrapping {url}:{e}")
        return ""
    return content


def concatenate_content(articles):
    """
    Concatenates the content of the provided articles into a single string.
    """
    full_text = ""
    # formatting + concatenation of the string is implemented here
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    documents = []
    for article in articles:
        content = fetch_article_content(article['link'])
        doc_chunks = text_splitter.split_text(content)
        
        for chunk in doc_chunks:
            documents.append(Document(
                page_content = chunk,
                metadata = {
                    'source':article['link'],
                    'title':article.get('title','')
                }
            ))
    vectorstore = Chroma.from_documents(
                    documents=documents,                 # Data
                    embedding=gemini_embeddings,    # Embedding model
                    persist_directory="./chroma_db" # Directory to save data
                    )
    vectorstore.persist()


def generate_answer(query):
    """s
    Generates an answer from the concatenated content using GPT-4.
    The content and the user's query are used to generate a contextual answer.
    """
    # Create the prompt based on the content and the query
    response = None
    vectorstore_disk = Chroma(
                        persist_directory="./chroma_db",       # Directory of db
                        embedding_function=gemini_embeddings   # Embedding model
                   )

    retriever = vectorstore_disk.as_retriever(search_kwargs={"k": 5})
    content = retriever.get_relevant_documents(query)
    combined_content = "\n".join([doc.page_content for doc in content])
    pprint.pprint(content)
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.8, top_p=0.85)
    llm_prompt_template = """You are an assistant for question-answering tasks.
    Here is the conversation history to give you context.
    Use the following context to answer the question.
    If you don't know the answer, just say that you don't know.
    Use five sentences maximum and keep the answer concise.\n
    Question: {query} \nContext: {content} \nAnswer:"""
    llm_prompt = PromptTemplate.from_template(llm_prompt_template).format(
        query=query,
        content=combined_content
    )
    response = llm.predict(llm_prompt)
    # llm_prompt.invoke({"query":query, "content":combined_content})
    print("i have reached genereate answer")
    print(response)
    # implement openai call logic and get back the response
    return response
