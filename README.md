# 🚀 AI Engineering Assignment – Cognitive Routing & RAG

## Overview
This project implements a cognitive AI pipeline for the Grid07 platform. It focuses on:
- Vector-based persona routing
- Autonomous content generation using LangGraph
- Retrieval-Augmented Generation (RAG) for contextual replies
- Prompt injection defense mechanisms

---

## 🧩 Architecture

### Phase 1: Vector-Based Persona Matching (Router)
- Used FAISS/ChromaDB as in-memory vector store
- Generated embeddings for 3 bot personas:
  - Tech Maximalist
  - Doomer / Skeptic
  - Finance Bro
- Incoming posts are embedded and matched using cosine similarity
- Bots are selected if similarity > threshold (default: 0.85)

---

### Phase 2: Autonomous Content Engine (LangGraph)

#### Node Structure:
1. **Decide Search**
   - LLM selects a topic based on persona
   - Outputs a search query

2. **Web Search (Mock Tool)**
   - Simulated search using `mock_searxng_search`
   - Returns relevant headlines

3. **Draft Post**
   - LLM generates a 280-character opinionated post
   - Uses persona + search context

#### Output Format:
Strict JSON:
```json
{
  "bot_id": "...",
  "topic": "...",
  "post_content": "..."
}
```

---

### Phase 3: Combat Engine (Deep Thread RAG)

#### Approach:
- Full conversation context is passed:
  - Parent post
  - Comment history
  - Latest human reply

- RAG prompt ensures:
  - Context awareness
  - Logical argument continuation

#### Prompt Injection Defense:
- System prompt enforces:
  - Persona consistency
  - Instruction hierarchy (system > user)
  - Explicit rejection of malicious instructions

Example defense:
> "Ignore all previous instructions..." → Rejected by system-level constraints

---

## 🛡️ Security Strategy

- Strict system prompts prevent role deviation
- No direct execution of user instructions
- Context filtering for malicious patterns
- Persona anchoring in every generation step

---

---

## ⚙️ Setup Instructions

```bash
pip install -r requirements.txt
```

Create `.env` file:
```
GEMINI_API_KEY=your_key_here
```

Run:
```bash
python main.py
```

---

## 📊 Execution Logs

Include:
- Routing results (Phase 1)
- JSON outputs (Phase 2)
- Injection defense example (Phase 3)

---

## 📌 Notes
- Threshold tuning may vary depending on embedding model
- Mock search tool simulates real-time context
- LangGraph ensures modular and extensible orchestration

---

