import tkinter as tk
from tkinter import filedialog
import os
import json
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader, TextLoader
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_nomic.embeddings import NomicEmbeddings
from langchain_ollama import ChatOllama

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_nomic.embeddings import NomicEmbeddings

urls = [
  # "https://www.gov.uk/guidance/sign-in-to-your-hmrc-business-tax-account",
  # "https://lilianweng.github.io/posts/2023-06-23-agent/",
  # "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
  # "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
  "/Users/adiprabs/Coding/RAG/Wills and Estates.pdf",
  "/Users/adiprabs/Coding/RAG/john.txt",
  "/Users/adiprabs/Coding/RAG/dispute.pdf",
  "/Users/adiprabs/Coding/RAG/contract.pdf",
]

# Initialize LLMs
local_llm = "llama3.2:3b-instruct-fp16"
llm = ChatOllama(model=local_llm, temperature=0)
llm_json_mode = ChatOllama(model=local_llm, temperature=0, format="json")

# Initialize variables
conversation_history = []
loaded_documents = []
vectorstore = None
retriever = None

from typing import List, Dict
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

def rate_document_relevance(question: str, documents: List[str]) -> List[Dict]:
  """Rate each document's relevance to the question on a 0-10 scale"""
  
  # Encode question and documents
  question_embedding = model.encode([question])[0]
  doc_embeddings = model.encode(documents)
  
  # Calculate similarities
  similarities = [cosine_similarity([question_embedding], [doc_emb])[0][0] 
                  for doc_emb in doc_embeddings]
  
  # Convert to 0-10 scale
  scores = [float(min(10, max(0, sim * 10))) for sim in similarities]
  
  return [
    {
        "document": doc,
        "relevance_score": score,
        "explanation": f"Document received a relevance score of {score:.1f}/10"
    }
    for doc, score in zip(documents, scores)
  ]

def check_answer_grounding(answer: str, documents: List[str]) -> Dict:
  """Check if the answer is grounded in the provided documents"""
  
  # Encode answer and documents
  answer_embedding = model.encode([answer])[0]
  doc_embeddings = model.encode(documents)
  
  # Calculate max similarity
  max_sim = max(cosine_similarity([answer_embedding], doc_embeddings)[0])
  grounding_score = float(min(10, max(0, max_sim * 10)))
  
  return {
    "grounding_score": grounding_score,
    "is_grounded": grounding_score > 7.0,
    "explanation": f"Answer received a grounding score of {grounding_score:.1f}/10"
  }


def get_answer(question: str):
  if vectorstore is None:
    print("No documents loaded. Please load documents first.")
    return "No documents loaded. Please load documents first."
  
  conversation_history.append({"role": "user", "content": question})

  # Retrieve documents
  docs = retriever.invoke(question)
  docs_txt = "\n\n".join(doc.page_content for doc in docs)

  # Include conversation history in the prompt
  history = "\n".join([f"{turn['role'].capitalize()}: {turn['content']}" for turn in conversation_history])

  # Create the prompt
  rag_prompt_formatted = f"""You are a legal assistant for question-answering tasks. 

Here is the legal context to use to answer the question:

{docs_txt}

Here is the conversation history:

{history}

Provide an answer to the user's last question using only the above context. Provide clarification on the people involved and the terms used to describe them. 
If the answer is not available in the given context, say so. Give reference to laws or legal precedents that are relevant, but do not give any legal advice

Use three sentences maximum and keep the answer concise, unless asked otherwise to elaborate.

Answer:"""

  # Generate the answer
  generation = llm.invoke([HumanMessage(content=rag_prompt_formatted)])
  answer = generation.content
  print(f"Assistant: {answer}")
  conversation_history.append({"role": "assistant", "content": answer})
  return answer

def main():
  root = tk.Tk()
  root.title("My GUI App")

  # Text widget to show loading status and other messages
  status_text = tk.Text(root, wrap=tk.WORD, height=4, width=60)
  status_text.pack(pady=5)

  def load_files():
    status_text.delete("1.0", tk.END)
    status_text.insert(tk.END, "Loading documents, please wait...\n")
    
    global vectorstore, retriever, loaded_documents
    loaded_documents = []  # Reset the list


    for path in urls:
      try:
        if path.startswith('http'):
          loaded_documents.append(WebBaseLoader(path).load())
          status_text.insert(tk.END, f"Loaded: {path}\n")
        elif path.endswith('txt'):
          loaded_documents.append(TextLoader(path, autodetect_encoding=True).load())
          status_text.insert(tk.END, f"Loaded: {path}\n")
        elif path.endswith('pdf'):
          loaded_documents.append(PyPDFLoader(path).load())
          status_text.insert(tk.END, f"Loaded: {path}\n")
        else:
          status_text.insert(tk.END, f"Unsupported format: {path}\n")
      except Exception as e:
        status_text.insert(tk.END, f"Error loading {path}: {str(e)}\n")
        print(f"Error loading {path}: {str(e)}")
    docs_list = [item for sublist in loaded_documents for item in sublist]


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
    docs_count = len(docs_list)
    max_k = min(4, docs_count)  # Ensure k does not exceed available documents
    retriever = vectorstore.as_retriever(k=max_k)
    status_text.insert(tk.END, f"Retriever set to k={max_k}\n")
    status_text.insert(tk.END, "Documents loaded successfully!\n")

  def ask_question():
    # Create a pop-up to enter the question
    popup = tk.Toplevel(root)
    popup.title("Ask a Question")

    tk.Label(popup, text="Enter your question:").pack(padx=10, pady=5)
    question_entry = tk.Entry(popup, width=50)
    question_entry.pack(padx=10, pady=5)

    def submit_question():
      user_question = question_entry.get()
      # Replace with your logic to produce an answer
      answer = get_answer(user_question)

      # Create a text widget to show the answer with wrapping
      answer_window = tk.Toplevel(root)
      answer_window.title("Answer")
      answer_text = tk.Text(answer_window, wrap=tk.WORD, width=60, height=10)
      answer_text.pack(padx=10, pady=10)
      answer_text.insert(tk.END, answer)

      #save answer to file
      def save_answer():
        filepath = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text Files", "*.txt")],
            title="Save Answer As"
        )
        if filepath:
            with open(filepath, "w") as f:
                f.write(answer_text.get("1.0", tk.END))

      save_button = tk.Button(answer_window, text="Save to File", command=save_answer)
      save_button.pack(pady=2)

      popup.destroy()

    submit_button = tk.Button(popup, text="Submit", command=submit_question)
    submit_button.pack(pady=10)

  menubar = tk.Menu(root)
  file_menu = tk.Menu(menubar, tearoff=0)
  file_menu.add_command(label="Exit", command=root.quit)
  menubar.add_cascade(label="File", menu=file_menu)
  root.config(menu=menubar)

  button_frame = tk.Frame(root)
  button_frame.pack(pady=20)

  load_button = tk.Button(button_frame, text="Load Files", command=load_files)
  load_button.pack(side=tk.LEFT, padx=5)

  ask_button = tk.Button(button_frame, text="Ask Question", command=ask_question)
  ask_button.pack(side=tk.LEFT, padx=5)

  exit_button = tk.Button(button_frame, text="Exit", command=root.quit)
  exit_button.pack(side=tk.LEFT, padx=5)

  root.mainloop()

if __name__ == "__main__":
  main()