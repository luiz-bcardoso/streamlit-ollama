import streamlit as st
import ollama
from pathlib import Path
from docling.document_converter import DocumentConverter
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate

# --- 1. SETUP & CONFIGURATION ---
st.set_page_config(page_title="Academic AI Assistant", layout="wide")

if "summary" not in st.session_state:
    st.session_state.summary = None
if "discussion" not in st.session_state:
    st.session_state.discussion = None

# --- 2. HELPER FUNCTIONS ---

def get_installed_models():
    """Fetches list of available Ollama models (Handles old & new library versions)."""
    try:
        models_info = ollama.list()
        
        # Check if response is an object or dict
        if not isinstance(models_info, dict) and hasattr(models_info, 'models'):
            return [m.model for m in models_info.models]
            
        # Old version returns a dict
        if 'models' in models_info:
            return [m.get('name', m.get('model')) for m in models_info['models']]
            
        return []
    except Exception as e:
        # If Ollama is not running, we return a fallback list so the app doesn't crash
        st.warning(f"Ollama connection issue: {e}")
        return ["gemma3:12b"]

@st.cache_data(show_spinner=False)
def extract_text_from_pdf(uploaded_file):
    """Saves and converts PDF to Markdown. Cached to avoid re-processing."""
    try:
        temp_path = Path("temp_upload.pdf")
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        converter = DocumentConverter()
        result = converter.convert(str(temp_path))
        return result.document.export_to_markdown()
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return None

def run_llm(model_name, temp, tokens, ctx, prompt_text, inputs):
    """Generic function to call Ollama."""
    try:
        llm = ChatOllama(
            model=model_name,
            temperature=temp,
            num_predict=tokens,
            num_ctx=ctx
        )
        chain = PromptTemplate.from_template(prompt_text) | llm
        return chain.invoke(inputs).content
    except Exception as e:
        st.error(f"LLM Error: {e}")
        return "Error generating text."

# --- 3. SIDEBAR (SETTINGS) ---
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Model Selection
    available_models = get_installed_models()
    
    # Default to gemma3 if available, otherwise first item
    default_ix = 0
    if "gemma3:12b" in available_models:
        default_ix = available_models.index("gemma3:12b")
    elif available_models:
        default_ix = 0
        
    selected_model = st.selectbox(
        "Model", 
        available_models if available_models else ["gemma3:12b"], 
        index=default_ix
    )

    # Pull New Model
    with st.expander("Download New Model"):
        new_model_name = st.text_input("Model Name (e.g., llama3)")
        if st.button("Pull Model"):
            with st.spinner(f"Pulling {new_model_name}..."):
                try:
                    ollama.pull(new_model_name)
                    st.success("Done! Refresh page to see it.")
                except Exception as e:
                    st.error(f"Failed: {e}")

    st.divider()
    
    # Parameters
    temp = st.slider("Temperature", 0.0, 1.0, 0.4, help="Creativity vs Precision")
    max_tokens = st.number_input("Max Tokens", 100, 8192, 2048)
    ctx_window = st.select_slider("Context Window", options=[2048, 8192, 32768, 128000], value=32768)

# --- 4. MAIN INTERFACE ---
st.title("📄 Academic Paper Synthesizer")

col1, col2 = st.columns(2)
with col1:
    topic = st.text_area("Target Topic", height=100, placeholder="e.g., Anti-inflammatory effects of...")
with col2:
    project = st.text_area("Your Project Context", height=100, placeholder="e.g., A systematic review on...")

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

# --- 5. EXECUTION LOGIC ---
if uploaded_file:
    with st.spinner("Reading PDF..."):
        doc_text = extract_text_from_pdf(uploaded_file)

    if doc_text and st.button("🚀 Generate Analysis"):
        
        # Step 1: Summary
        with st.status("Analyzing document...", expanded=True) as status:
            st.write("Generating structured summary...")
            prompt_summary = """
            Act as an academic researcher. Analyze the following document.
            
            CONTEXT:
            - Topic: {topic}
            - Project: {project}
            
            DOCUMENT CONTENT:
            {doc_text}
            
            TASK:
            1. Start with an ABNT2 citation.
            2. Summarize the Problem, Methodology, and Results.
            3. Explicitly explain how this paper helps the specific Project mentioned above.
            """
            
            summary_res = run_llm(
                selected_model, temp, max_tokens, ctx_window, 
                prompt_summary, 
                {"topic": topic, "project": project, "doc_text": doc_text}
            )
            st.session_state.summary = summary_res
            st.write("Summary complete.")
            
            # Step 2: Discussion Rewrite
            st.write("Drafting discussion section...")
            prompt_discussion = """
            Act as an expert editor. Rewrite this summary into a formal academic 
            'Discussion' section paragraph in English (or the user's preferred language).
            
            INPUT SUMMARY:
            {summary}
            """
            
            discussion_res = run_llm(
                selected_model, temp, max_tokens, ctx_window,
                prompt_discussion,
                {"summary": summary_res}
            )
            st.session_state.discussion = discussion_res
            status.update(label="All tasks completed!", state="complete")

# --- 6. DISPLAY RESULTS ---
if st.session_state.summary:
    st.subheader("📝 Analytical Summary")
    st.markdown(st.session_state.summary)

if st.session_state.discussion:
    st.subheader("🎓 Discussion Draft")
    st.markdown(st.session_state.discussion)
    
    st.download_button(
        label="Download Discussion (.md)",
        data=st.session_state.discussion,
        file_name="discussion_draft.md",
        mime="text/markdown"
    )