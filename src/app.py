import streamlit as st
from main import main

st.set_page_config(layout = "wide")

st.title("RAG Agent Streamlit Interface")

if "agent_initialized" not in st.session_state:
    st.session_state.agent_initialized = False

if "rag_agent" not in st.session_state:
    st.session_state.rag_agent = None

if st.sidebar.button(label = "Initialize Agent", help = "Initialize the RAG agent with the specified model name and URL.", key = "initialize_agent", icon = ":material/robot_2:"):
    with st.spinner("Initializing RAG agent..."):
        st.session_state.rag_agent = main()
        st.session_state.agent_initialized = True
        st.session_state.logs = None
    st.sidebar.success("Agent initialized!")

if "logs" not in st.session_state:
    st.session_state.logs = None 

if not st.session_state.agent_initialized:
    st.info("Please initialize the agent from the sidebar before submitting queries.")
    st.stop()

query = st.text_input("Enter your query:", key = "query_input", placeholder = "e.g., What is your flagship product?")
q = query.strip()

if q is None or q == "":
    st.info("Enter a query above and click enter to start.")
else:
    if q.lower() in ["logs", "log"]:
        if st.session_state.logs is None:
            st.warning("Invalid request. There was no query made previously.")
        else:
            st.subheader("LOGS:")
            for log_entry in st.session_state.logs:
                st.text(log_entry)
    else:
        with st.spinner("Processing your query..."):
            response = st.session_state.rag_agent.agent.process_query(q)

            st.session_state.logs = response.get('log', None)
            result = response.get('result', None)
            
            st.subheader("Result")
            st.info(result)

            st.subheader("Query Result")
            st.markdown(f"**Query:** {response.get('query', q)}")
            st.markdown(f"**Tool Used:** {response.get('tool_used', 'unknown')}")

            if response.get('tool_used') == 'RAG':
                with st.expander("Show Retrieved Chunks"):
                    st.subheader("Retrieved Chunks")
                    for i, chunk in enumerate(response.get('retrieved_chunks', [])):
                        st.markdown(
                            f"**Chunk {i+1} (Source: {chunk.get('source', 'N/A')}, Score: {chunk.get('relevance_score', 0):.2f})**"
                        )
                        content_snippet = chunk.get('content', '')
                        st.text(content_snippet)