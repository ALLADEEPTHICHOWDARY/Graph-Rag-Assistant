from neo4j import GraphDatabase
from langchain_community.document_loaders import WikipediaLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline
from dotenv import load_dotenv
import os
import re
import string

load_dotenv()

# -------------------------------
# Neo4j
# -------------------------------
driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
)

# -------------------------------
# Model
# -------------------------------
generator = pipeline("text2text-generation", model="google/flan-t5-base")

vectorstore = None

# -------------------------------
# CLEAN ENTITY
# -------------------------------
def clean_entity(text):
    text = text.lower().strip()
    text = text.translate(str.maketrans('', '', string.punctuation))

    stopwords = {
        "the","is","are","was","were","be","been","being",
        "this","that","these","those","with","from","into",
        "about","between","during","before","after",
        "to","of","in","on","for","by","and","or","as",
        "their","them","they","there",
        "include","including","used","using","based",
        "multiple","several","many","different","various",
        "some","such","found","ranging","group","units",
        "perform","enable","associated","typically",
        "methods","process","system","another","called"
    }

    if len(text) < 3:
        return None

    if text in stopwords:
        return None

    return text

# -------------------------------
# NORMALIZATION (FIX SPLITS)
# -------------------------------
def normalize_entity(text):
    mapping = {
        "artificial": "artificial intelligence",
        "intelligence": "artificial intelligence",
        "machine": "machine learning",
        "learning": "machine learning",
        "neural": "neural networks",
        "networks": "neural networks",
    }
    return mapping.get(text, text)

# -------------------------------
# ENTITY VALIDATION (BALANCED)
# -------------------------------
def is_valid_entity(e):
    bad = {
        "allowed","focuses","belief","concerned",
        "achieving","utilizing","connected","compose",
        "another","called","either","complex","common"
    }

    if e in bad:
        return False

    if len(e) < 3:
        return False

    return True

# -------------------------------
# RELATION CLEANING
# -------------------------------
def refine_relation(r):
    r = r.lower()

    if "include" in r:
        return "includes"
    elif "use" in r:
        return "uses"
    elif "learn" in r:
        return "learns"
    elif "model" in r:
        return "models"
    elif "process" in r:
        return "processes"

    return "related to"

# -------------------------------
# LOAD DATA
# -------------------------------
def load_wikipedia(topic):
    loader = WikipediaLoader(query=topic, load_max_docs=1)
    return loader.load()

# -------------------------------
# EXTRACT TRIPLETS
# -------------------------------
def extract_triplets(text):
    prompt = f"""
Extract knowledge graph triples.

Return 5-10 triples.

Format:
(subject, relation, object)

Text:
{text[:1200]}
"""

    result = generator(prompt, max_length=300, do_sample=False)[0]['generated_text']

    triplets = []

    matches = re.findall(r"\((.*?)\)", result)

    for match in matches:
        parts = match.split(",")

        if len(parts) == 3:
            s, r, o = [p.strip() for p in parts]

            s = clean_entity(s)
            o = clean_entity(o)

            if not s or not o:
                continue

            # normalize
            s = normalize_entity(s)
            o = normalize_entity(o)

            if s == o:
                continue

            # validate
            if not is_valid_entity(s) or not is_valid_entity(o):
                continue

            r = refine_relation(r)

            # control noise
            if r == "related to":
                if len(s.split()) == 1 and len(o.split()) == 1:
                    continue

            triplets.append((s, r, o))

    # fallback (ensures graph builds)
    if not triplets:
        words = re.findall(r"\b[a-zA-Z]{5,}\b", text.lower())

        for i in range(len(words) - 1):
            s = clean_entity(words[i])
            o = clean_entity(words[i+1])

            if s and o:
                s = normalize_entity(s)
                o = normalize_entity(o)

                if s != o and is_valid_entity(s) and is_valid_entity(o):
                    triplets.append((s, "related to", o))

        triplets = triplets[:6]

    return list(set(triplets))

# -------------------------------
# STRUCTURE (MINIMAL)
# -------------------------------
def enhance_structure(triplets):
    core = [
        ("artificial intelligence", "includes", "machine learning"),
        ("machine learning", "includes", "deep learning"),
        ("deep learning", "includes", "neural networks"),
    ]

    triplets.extend(core)
    return list(set(triplets))

# -------------------------------
# STORE
# -------------------------------
def store_triplet(s, r, o):
    query = """
    MERGE (a {name: $s})
    MERGE (b {name: $o})
    MERGE (a)-[:RELATION {type: $r}]->(b)
    """
    with driver.session() as session:
        session.run(query, s=s, r=r, o=o)

# -------------------------------
# INGEST
# -------------------------------
def ingest_topic(topic):
    global vectorstore

    docs = load_wikipedia(topic)
    if not docs:
        return 0

    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=80)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    split_docs = splitter.split_documents(docs)
    vectorstore = FAISS.from_documents(split_docs, embeddings)

    all_triplets = []

    for doc in docs:
        chunks = splitter.split_text(doc.page_content)

        for chunk in chunks[:3]:
            all_triplets.extend(extract_triplets(chunk))

    unique_triplets = list(set(all_triplets))
    unique_triplets = enhance_structure(unique_triplets)

    for s, r, o in unique_triplets:
        store_triplet(s, r, o)

    return len(unique_triplets)

# -------------------------------
# GRAPH QUERY
# -------------------------------
def query_graph(query):
    cypher = """
    MATCH (a)-[r]->(b)
    RETURN a.name, r.type, b.name
    LIMIT 5
    """
    with driver.session() as session:
        result = session.run(cypher)
        return [f"{r[0]} - {r[1]} -> {r[2]}" for r in result]

# -------------------------------
# VECTOR QUERY
# -------------------------------
def query_vector(query):
    global vectorstore
    if vectorstore is None:
        return []

    docs = vectorstore.similarity_search(query, k=2)
    return [d.page_content for d in docs]

# -------------------------------
# ANSWER
# -------------------------------
def generate_answer(question, graph_data, vector_data):
    context = "\n\n".join(graph_data + vector_data)[:1500]

    prompt = f"""
Context:
{context}

Question:
{question}

Answer:
"""

    result = generator(prompt, max_length=200, do_sample=False)
    return result[0]['generated_text']