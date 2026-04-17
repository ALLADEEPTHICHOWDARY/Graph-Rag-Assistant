# Hybrid Graph RAG System (Neo4j + Vector Search)

## Overview

I built this project to explore how combining knowledge graphs with vector search can improve RAG systems.

It implements a Hybrid Retrieval-Augmented Generation (RAG) system that combines:

* Vector Search (FAISS) for semantic retrieval
* Knowledge Graph (Neo4j) for structured relationships
* LLM (FLAN-T5) for triplet extraction and answer generation

Unlike traditional RAG systems that rely only on embeddings, this approach adds a graph layer to improve context understanding and explainability.

---

## Architecture

```
User Query
     ↓
Hybrid Retrieval Layer
 ├── Graph Query (Neo4j)
 └── Vector Search (FAISS)
     ↓
Context Aggregation
     ↓
LLM (FLAN-T5)
     ↓
Final Answer
```

---

## Features

* Hybrid retrieval (Graph + Vector)
* Automatic knowledge graph creation from unstructured text
* Neo4j integration for visualization and querying
* FAISS-based semantic search
* LLM-powered triplet extraction
* Streamlit UI with chat interface
* Fallback mechanism for consistent graph generation

---

## Knowledge Graph Design

* Nodes represent concepts/entities
* Relationships represent semantic connections
* Core structure enforced:

```
Artificial Intelligence
    → Machine Learning
        → Deep Learning
            → Neural Networks
```

* Additional branches:

  * Natural Language Processing
  * Computer Vision
  * Robotics

---

## Sample Graph Output

The graph below shows relationships extracted from Wikipedia data and structured into a knowledge graph.

(Add your Neo4j screenshot here)

---

## Tech Stack

* Python
* Neo4j – Graph Database
* LangChain – Data processing and retrieval
* FAISS – Vector similarity search
* HuggingFace Transformers – LLM (FLAN-T5)
* Streamlit – UI

---

## Setup Instructions

### 1. Clone the Repository

```
git clone https://github.com/your-username/graph-rag-assistant.git
cd graph-rag-assistant
```

### 2. Install Dependencies

```
pip install -r requirements.txt
```

### 3. Configure Environment

Create `.env` file:

```
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
```

### 4. Run the Application

```
streamlit run app.py
```

---

## How It Works

1. User enters a topic (e.g., Artificial Intelligence)
2. System fetches data from Wikipedia
3. LLM extracts triplets (subject, relation, object)
4. Triplets are stored in Neo4j
5. Vector embeddings are stored in FAISS
6. On query:

   * Graph and vector results are combined
   * LLM generates the final answer

---

## Design Trade-offs

* LLM-based extraction can introduce noise
* Rule-based filtering is used to improve graph quality
* Fallback mechanism ensures consistent graph generation
* Prioritized performance and reliability over perfect accuracy

---

## Future Improvements

* Use advanced models for better triplet extraction
* Replace rule-based filtering with NLP pipelines
* Add graph ranking and weighting
* Improve relation semantics beyond "related to"
* Deploy as a full-stack application

---

## Key Learning Outcomes

* Built a hybrid RAG system combining symbolic and semantic retrieval
* Worked with Neo4j graph modeling and Cypher queries
* Implemented LLM-based information extraction
* Understood real-world trade-offs in AI system design

---


