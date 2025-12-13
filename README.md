---
title: Navigator Server
emoji: 🧭
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 7860
---

# 🧭 Navigator Server - AI Educational Diagnostician Backend

This is the backend API for the Navigator educational diagnostics system, powered by FastAPI, LangGraph, and AI agents.

## 🚀 Features

- **Foundational Skill Diagnostician Agent** - Analyzes student answers and pinpoints conceptual gaps
- **RAG (Retrieval-Augmented Generation)** - Uses Qdrant vector database for semantic search
- **Cohere Reranking** - Improved retrieval accuracy
- **OpenAI GPT-4o-mini** - LLM-powered evaluation and feedback
- **RAGAS Evaluation** - Quantifies answer quality metrics

## 📡 API Endpoints

### Core Endpoints
- `GET /api/health` - Health check
- `POST /api/chat` - General chat completion
- `POST /api/rag-chat` - RAG-powered chat with curriculum documents

### Diagnostics & Evaluation
- `POST /api/search` - Semantic search in curriculum content
- `POST /api/evaluate` - AI agent evaluation of student answers
- `GET /api/rag-status` - Check RAG system status
- `GET /api/conversations/{user_id}` - Get conversation history

## 🛠️ Tech Stack

- **FastAPI** - High-performance Python web framework
- **LangGraph + LangChain** - Agent orchestration
- **Qdrant** - Vector database for semantic search
- **OpenAI** - GPT-4o-mini for LLM reasoning
- **Cohere** - Reranking for retrieval
- **Tavily** - Web search fallback

## 🔑 Environment Variables Required

This Space requires the following environment variables to be set:

- `OPENAI_API_KEY` - Your OpenAI API key
- `COHERE_API_KEY` - Your Cohere API key (for reranking)
- `TAVILY_API_KEY` - Your Tavily API key (for web search)
- `QDRANT_URL` - Set to `./qdrant_local` (local persistent storage)
- `COLLECTION_NAME` - Set to `science_curriculum_g3_g6`

## 📚 Ontario Science Curriculum Focus

This system is designed for Ontario Science curriculum (Grades 3-6), with initial focus on:
- Bees and Pollination
- Water Cycle
- Ecosystems

## 🔗 Related Projects

- **Frontend UI**: [navigator-ui](https://github.com/lalrow/navigator-ui)
- **Backend Server**: [Navigator-Server](https://github.com/lalrow/Navigator)

## 📄 License

ISC License

## 👨‍💻 Author

Lalit Ahlawat (@lalrow)

