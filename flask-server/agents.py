from langchain_ollama import ChatOllama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_nomic.embeddings import NomicEmbeddings
import json
from langchain_core.messages import HumanMessage, SystemMessage
import os
from datetime import datetime
import re

local_llm = "llama3.2:3b-instruct-fp16"
llm = ChatOllama(model=local_llm, temperature=0)
llm_json_mode = ChatOllama(model=local_llm, temperature=0, format="json")

receptionist_prompt = """
You are a receptionist at a law firm. A client has approached and the following is the problem they are facing.
Your job is to redirect them to the right specialised lawyer you who can help them. Return JSON with a single key, lawyer, that is one of the following options that the client is likely to find the best help from
[criminal, family, wills and estates]
"""

def receptionist_prompt_response(prompt):
  print("receptionist_prompt_response")
  result = llm_json_mode.invoke(
    [SystemMessage(content=receptionist_prompt)]
    + [HumanMessage(content=prompt)]
  )
  return json.loads(result.content)


urls = [
  # "https://www.gov.uk/guidance/sign-in-to-your-hmrc-business-tax-account",
  # "https://lilianweng.github.io/posts/2023-06-23-agent/",
  # "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
  # "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
  "/Users/adiprabs/Coding/RAG/Wills_and_Estates.pdf",
]

relevant_docs = {
  "wills and estates" : ["/Users/adiprabs/Coding/RAG/Wills_and_Estates.pdf"],
  "criminal" : ["/Users/adiprabs/Coding/RAG/Wills and Estates.pdf"],
  "family" : ["/Users/adiprabs/Coding/RAG/Wills and Estates.pdf"],
}

# Load documents
def load_docs(category):
  print("load_docs")
  '''
  Load documents from relevant directories for respective lawyer and returns retriever
  '''
  docs = []
  for path in relevant_docs[category]:
    if path.startswith('http'):
      docs.append(WebBaseLoader(path).load())
    else:
      docs.append(PyPDFLoader(path).load())
  docs_list = [item for sublist in docs for item in sublist]


  # Split documents
  text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1000, chunk_overlap=200
  )
  doc_splits = text_splitter.split_documents(docs_list)

  # Add to vectorDB
  vectorstore = SKLearnVectorStore.from_documents(
    documents=doc_splits,
    embedding=NomicEmbeddings(model="nomic-embed-text-v1.5", inference_mode="local"),
  )

  # Create retriever
  return vectorstore.as_retriever(k=3)

# %%
### Generate

# Prompt
rag_prompt = """
You are a lawyer tasked with getting all relevant information for a potential case. 
You need to ask relevant questions to get the information you need to help your client.
Here is the context to use to answer the question:

{context} 

Think carefully about the above context. 

Now, review your conversation history with the user

{history}

Ask a question that will help you get information you do not already have that may be essential for the case. Only repeat questions if you feel earlier answers have not given adequate information, while specifying what exactly you need. 
You will need the following details:

{details}

If you have no more questions, specifically say the phrase "Now over".

Question:"""


# Post-processing
def format_docs(docs):
  print("format_docs")
  return "\n\n".join(doc.page_content for doc in docs)


messages = ""

information_needed = {
  "wills and estates" : "full names, dates of birth, current marital status, address and contact information, aliases and other names, children (their names and dob), dependents, all assests owned jointly and individually, debts and liabilities associated with assets, any assests aborad, heirlooms, backup executor",
  "contract" : "full names, dates of birth, current marital status, address and contact information, aliases and other names, children (their names and dob), dependents, all assests owned jointly and individually, debts and liabilities associated with assets, any assests aborad, heirlooms, backup executor",
}

# Test
def lawyer_response(initial_query, category, retriever):
  print("lawyer_response")
  docs = retriever.invoke(initial_query)
  docs_txt = format_docs(docs)
  messages = f"user: {initial_query}\n"
  for i in range(10):
    rag_prompt_formatted = rag_prompt.format(context=docs_txt, history=messages, details=information_needed[category])
    generation = llm.invoke([HumanMessage(content=rag_prompt_formatted)])
    messages += f"lawyer: {generation.content}\n"
    print(generation.content)
    if "Now over" in generation.content:
      break
    answer = input(">>>")
    print(answer)
    if answer == "/exit":
      break
    messages += f"user: {answer}\n"

  return (messages, docs_txt)

secretary_prompt = """
You are the tasked with compiling all relevant information from a potential client. Below is the conversation history between the client and the lawyer.
compile a report will all relevant information that the lawyer may need to help the client.

{history}

Present a concise report in with the following titles and their relevant information:
name, case summary, location, will, executor, beneficiaries, guardian, funeral arrangements, other wishes. At then end of the report, cite relevant laws or principles that may apply to this case from the following context.

If any information is missing or insufficient, specify missing for that tag.
"""

def extract_client_name(history):
  # Look for name pattern in conversation
  name_match = re.search(r'Name:\s*([^\n]+)', history, re.IGNORECASE)
  if name_match:
      return name_match.group(1).strip()
  return "unnamed_client"

def secretary_response(msg_history):
  print("secretary_response")
  
  # Generate report
  secretary_prompt_formatted = secretary_prompt.format(history=msg_history)
  result = llm.invoke([HumanMessage(content=secretary_prompt_formatted)])
  report_content = result.content
  
  # Save report
  timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  client_name = "unamed_client" #extract_client_name(msg_history)
  filename = f"{client_name}_{timestamp}.md"
  
  # Create reports directory if it doesn't exist
  reports_dir = os.path.join(os.path.dirname(__file__), "reports")
  os.makedirs(reports_dir, exist_ok=True)
  
  # Save report as markdown
  report_path = os.path.join(reports_dir, filename)
  with open(report_path, "w") as f:
    f.write(f"# Legal Report for {client_name}\n")
    f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write(report_content)
  
  return result.content

example = """

"""
class Lawyer:
  def __init__(self, category, initial_query):
    print(f"Lawyer for {category} initialized")
    self.category = category
    self.context = format_docs(load_docs(category).invoke(initial_query))
    self.history = f"user: {initial_query}\n"
    self.details = information_needed[category]
    self.qs_asked = 0
    self.max_questions = 2  # Set maximum questions

  def ask_question(self):
    print("lawyer is asking question")
    if self.qs_asked >= self.max_questions:
      return self.send_history_to_secretary()
        
    rag_prompt_formatted = rag_prompt.format(
      context=self.context, 
      history=self.history, 
      details=self.details
    )
    generation = llm.invoke([HumanMessage(content=rag_prompt_formatted)])
    self.history += f"lawyer: {generation.content}\n"
    self.qs_asked += 1
    return generation.content

  def answer_question(self, answer):
    print("lawyer is receiving answer")
    self.history += f"user: {answer}\n"
    print(self.history)
    return self.ask_question()

  def send_history_to_secretary(self):
    print("lawyer is sending history to secretary")
    report = secretary_response(self.history)
    return {
      "message": "Thank you for providing all the information. I'll have our secretary prepare a report.",
      "report": report
    }

if __name__ == "__main__":
  summary = input("Tell us what you'd like to file a case for: ")
  if summary == "exit":
    exit("exited")
  lawyer = receptionist_prompt_response(summary)["lawyer"]
  retriever = load_docs(lawyer)
  (conversation, context) = lawyer_response(summary, lawyer, retriever)
  secretary_response(conversation, context)