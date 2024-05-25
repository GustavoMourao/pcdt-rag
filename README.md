# pcdt-rag
RAG pipeline in PCDT (proposal)

This repository aims to apply RAG thecnique (Retrieval Augmented Generation) in the conext of geeting informations from PCDT file ("Protocolo Clínico e Diretrizes Terapêuticas do Tabagismo").

In this sense its important to emphasize some aspects from the application:

1. The application run locallly using Large Language Models (LLMs) with Ollama to perform Retrieval-Augmented Generation (RAG) for answering questions based on sample PDFs.

2. We have used Ollama to create embeddings with the nomic-embed-text (We have splitted the raw text in chunks of 500)

3. In order to perform vector-database storage we used Chroma: Chroma is a AI-native open-source vector database

4. Finally the prompt is send through FastAPI (REST protocol) and the answer is available in the same protocol


The main scope of the application is represented in the Figure below:
![Alt Text](https://github.com/GustavoMourao/pcdt-rag/blob/main/images/1_i3UYywX0p6KMB4CldUWO-A.webp)


## Instalation and inference procediment



## References:
```
[1] https://ollama.com/blog/embedding-models

[2] https://huggingface.co/nomic-ai/nomic-embed-text-v1.5

[3] https://python.langchain.com/v0.1/docs/integrations/vectorstores/chroma/

[4] https://aclanthology.org/2023.eacl-main.148.pdf

[5] https://github.com/amscotti/local-LLM-with-RAG
```
