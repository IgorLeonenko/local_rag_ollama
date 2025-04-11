## 🦙 Local RAG Agent with Llama 3.2
This concept demonstrate application implements RAG system using Llama 3.2 via Ollama, with Qdrant as the vector database and phidata library.

### Features
- Fully local RAG
- Powered by Llama 3.2 through Ollama
- Vector search using Qdrant
- No external API dependencies

## How to get Started?

### Local setup:
1. Install the required dependencies

```bash
pip install -r requirements.txt
```

2. Install and start [Qdrant](https://qdrant.tech/) vector database locally

```bash
docker pull qdrant/qdrant
docker run -p 6333:6333 -v qdrant_data:/qdrant/storage qdrant/qdrant #(this way you able to run qdrant docker instance with data persistancy)
```

4. Install [Ollama](https://ollama.com/download) and pull Llama 3.2 for LLM and OpenHermes as the embedder for OllamaEmbedder
```bash
ollama pull llama3.2
ollama pull openhermes
```

5. Run ollama
```bash
ollama serve
```
or just run installed ollama app

6. Set Qdrant collection name in
```bash
COLLECTION_NAME = ""
```

6. Run the AI RAG Agent
```bash
python local_rag_agent.py
```

### Docker:

1. Run docker command
```bash
docker compose up --build
```
> [!IMPORTANT]
> Using docker, remember to set container size for at least 50GB space and 8GB of RAM

## How to use
1. Open your web browser and navigate to the [http://localhost:8501](http://localhost:8501)

2. Upload your klowledgebase in PDF documents format. This process could take a while.

3. Ask Agent about anything you upload as knowledgebase. As additional option you can send LLM response via email.