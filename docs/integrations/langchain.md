# LangChain Integration

RTMDK provides adapters for LangChain.

## Basic Usage

```python
from rtmdk import RTMDKMemory, RTMDKConfig
from rtmdk.langchain_adapter import RTMDKRetriever
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

cfg = RTMDKConfig.local()
mem = RTMDKMemory(config=cfg, embedder=your_embedder)

# Add documents
mem.add_node("Document 1 content")
mem.add_node("Document 2 content")

# Create retriever
retriever = RTMDKRetriever(memory=mem)

# Use in RAG chain
qa = RetrievalQA.from_chain_type(
    llm=OpenAI(),
    chain_type="stuff",
    retriever=retriever,
)
result = qa.run("What is in the documents?")
```

## LCEL Pipeline

```python
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("""
Answer based on context:
{context}

Question: {question}
""")

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
)
```
