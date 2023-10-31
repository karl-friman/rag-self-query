# -*- coding: utf-8 -*-

# This requires Docker containers for Chroma, Redis and Qdrant to be running.
import os, constants

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from langchain.vectorstores.qdrant import Qdrant

from langchain.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain import hub
from langchain.document_loaders import DirectoryLoader, PyPDFLoader

# Set OpenAI API key from constants file
os.environ["OPENAI_API_KEY"] = constants.OPENAI_API_KEY

def process_llm_response(llm_response):
    # Outputting the answer or debugging information in case of an error
    # if "result" in llm_response:
    #     # print(f"Answer: {llm_response['result']}")
    #     print("hi")
    # else:
    #     print(
    #         "Result key not found in the returned object. Here's the full object for debugging:"
    #     )
    #     print(llm_response)

    print("\n\nSources:")
    for source in llm_response["source_documents"]:
        print(source.metadata["source"])


def main():
    # Loading documents from a specified directory
    loader = DirectoryLoader(
        "./data/", glob="./*.pdf", loader_cls=PyPDFLoader
    )
    documents = loader.load()
    # Splitting documents into manageable text chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500, chunk_overlap=100
    )
    all_splits = text_splitter.split_documents(documents)
    print(
        f"Processed {len(documents)} documents split into {len(all_splits)} chunks"
    )

    try:
        # !!! You will need this the first time you run this script as the collection needs to be created !!!
        # Setting up QdrantClient and creating a collection for vector storage
        from qdrant_client.http.models import Distance, VectorParams
        try:
            qdrant_client = QdrantClient(url="34.141.229.240", port=6333)
            qdrant_client.delete_collection(
                collection_name="test_collection",
            )
            qdrant_client = QdrantClient(url="34.141.229.240", port=6333)
            qdrant_client.create_collection(
                collection_name="test_collection",
                vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
            )
        except Exception as e:
            print(f"Failed to initialize Qdrant client or create collection: {e}")

        # Initializing Qdrant vectorstore with document embeddings
        # url = "http://localhost:6333"
        url = "http://34.141.229.240:6333"
        vectorstore = Qdrant.from_documents(
            collection_name="test_collection",
            embedding=OpenAIEmbeddings(),
            documents=all_splits,
            url=url,
        )
    except Exception as e:
        print(f"Failed to initialize Qdrant vectorstore: {e}")

    # Loading the Language Model with a callback manager
    llm = Ollama(
        model="openhermes2-mistral",
        verbose=True,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    )

    # Setting up a QA Chain with a specific prompt
    QA_CHAIN_PROMPT = hub.pull("rlm/rag-prompt-llama")
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
        return_source_documents=True,
    )

    # Asking a question on the Toolformer PDF
    question = "What is the toolformer?"
    print(f"Question: {question}")
    try:
        # Getting the answer using the QA chain
        result = process_llm_response(qa_chain({"query": question}))

    except Exception as e:
        # Handling exceptions during the QA process
        print("An error occurred:", e)
        print("Here's the partial result for debugging:")
        print(result)

    # Asking a question based on the other PDF.
    question = "What is the name of the cat?"
    print(f"Question: {question}")
    try:
        # Getting the answer using the QA chain
        result = process_llm_response(qa_chain({"query": question}))

    except Exception as e:
        # Handling exceptions during the QA process
        print("An error occurred:", e)
        print("Here's the partial result for debugging:")
        print(result)

if __name__ == "__main__":
    main()
