import streamlit as st
from rag_core import ingest_topic, query_graph, query_vector, generate_answer

st.set_page_config(layout="wide")
st.title(" Hybrid Graph RAG")

topic = st.sidebar.text_input("Enter topic")

if st.sidebar.button("Ingest"):
    if topic:
        count = ingest_topic(topic)
        st.sidebar.success(f"{count} triplets stored")

if st.sidebar.button(" Clear Chat"):
    st.session_state.messages = []
    st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

query = st.chat_input("Ask something...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})

    graph_data = query_graph(query)
    vector_data = query_vector(query)

    answer = generate_answer(query, graph_data, vector_data)

    st.session_state.messages.append({"role": "assistant", "content": answer})

    with st.chat_message("assistant"):
        st.markdown(answer)

    with st.expander(" Sources"):
        st.write("Graph:", graph_data)
        st.write("Vector:", vector_data)