from dataclasses import dataclass
from typing import List
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from rag_app.get_chroma_db import get_chroma_db
import os
import openai
from dotenv import load_dotenv
load_dotenv()
import openai
from typing import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, END

openai.api_key = os.environ['OPENAI_API_KEY']

CHROMA_PATH = "chroma"

LANG_PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}

Follow these instructions:
Elaborate on your response and give examples when necessary as if you are a science educator (but don't say you are one). 
Quote the episode title that you got the information from.
Stick closely to the excerpts but use your own knowledge where necessary.
"""

IMG_PROMPT_TEMPLATE = """
Generate a futuristic or sci-fi image for the following context: {img_prompt} 

Follow these instructions:
Don't try to cram everything into the picture. 
Pick one or two key concepts and illustrate them. 
YOU MUST Omit all language in the picture.
"""

LANG_MODEL_ID = ""
IMG_MODEL_ID = ""

@dataclass
class QueryResponse:
    query_text: str
    response_text: str
    img_text: str
    sources: List[str]

def query_rag(query_text: str) -> QueryResponse:
    # Get the DB.
    
    db = get_chroma_db()
    
    # Prepare chains
    retriever = db.as_retriever(search_kwargs={"k": 5})

    prompt_template = ChatPromptTemplate.from_template(LANG_PROMPT_TEMPLATE)
    
    # model = GoogleGenerativeAI(
    #         model="gemini-pro",
    #         max_output_tokens=1024,
    #         google_api_key=os.environ["GOOGLE_API_KEY"]
    #     )

    model = ChatOpenAI(
            model='gpt-4',
            max_tokens=1024
            )

    chain_with_prompt = prompt_template | model | StrOutputParser()

    class AgentState(TypedDict):
        question: str
        raw_docs: list[BaseMessage]
        formatted_docs: list[str]
        generation: str
        sources: list[str]
        image: str

    def get_docs(state: AgentState):
        #print("get_docs:", state)
        question = state["question"]
        docs = retriever.invoke(question)
        state["sources"] = [doc.metadata.get("id") for doc in docs]
        state["raw_docs"] = docs
        return state
    
    def format_docs(state:AgentState):
        #print("format_docs:",state)
        documents = state["raw_docs"]
        state["formatted_docs"] = "\n\n---\n\n".join(["Episode Title:" + doc.metadata.get('ep_name', None) 
                                    + "\nDate published:" + doc.metadata.get('dt_published', None) 
                                    + "\nExcerpt:" + doc.page_content 
                                    for doc in documents])
        return state
    
    def generate(state:AgentState):
        #print("generate:", state)
        question = state["question"]
        formatted_docs = state["formatted_docs"]
        result = chain_with_prompt.invoke({"question": question, "context":formatted_docs})
        state["generation"] = result
        return state
    
    def generate_img(state:AgentState):
        img_prompt = f'''
        Generate a Sci-fi image for the following context: 
        
        {state["question"]} 

        Don't try to cram everything into the picture. 
        Pick one or two key concepts and illustrate them. 
        YOU MUST Omit all words in the picture'''
        img = openai.OpenAI().images.generate(
                model="dall-e-3",
                prompt=img_prompt,
                size='1024x1024',
                quality="standard",
                n=1
                )
        state["image"] = img.data[0].url
        return state
    
    workflow = StateGraph(AgentState)
    workflow.add_node("get_docs", get_docs)
    workflow.add_node("format_docs", format_docs)
    workflow.add_node("generate", generate)
    workflow.add_node("generate_img", generate_img)
    workflow.add_edge("get_docs", "format_docs")
    workflow.add_edge("format_docs", "generate")
    workflow.add_edge("generate", "generate_img")
    workflow.add_edge("generate_img", END)
    workflow.set_entry_point("get_docs")

    rag_app = workflow.compile()

    result = rag_app.invoke({"question":query_text})

    print(f"Response: {result['generation']}\nSources: {result['sources']}")

    return QueryResponse(
        query_text=query_text, response_text=result['generation'], sources=result['sources'], img_text=result['image']
        )

if __name__ == "__main__":
    query_rag("How long can a civilization theoretically survive?")
