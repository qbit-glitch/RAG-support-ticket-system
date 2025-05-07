
import streamlit as st
from streamlit_chat import message
import tempfile

from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


DB_FAISS_PATH = "./vectorDB/faiss_index"


def query_faiss(user_query, index_path=DB_FAISS_PATH, top_k=5):
    """
    Given a user query, this function loads the persisted FAISS index,
    performs a similarity search, and returns the top_k matching documents.
    
    Args:
        user_query (str): The input query from the user.
        index_path (str): The path where the FAISS index is saved.
        top_k (int): Number of similar documents to return.
        
    Returns:
        list: A list of the top_k matching document objects.
    """
    
    model_name = "all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    
    faiss_db = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    results = faiss_db.similarity_search(user_query, k=top_k)
    # print(results)
    return results




def construct_rag_prompt(query, retrieved_docs):
    """
    Build a prompt that includes retrieved document content and the user's query.
    """
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    print(context)
    prompt = f"""
        You are a customer support AI assistant. Your task is to find the most appropriate resolution for a customer's issue based on their query.

        You are given:
        - A dataset of support tickets consisting of past customer's query that may contain similar issues and their corresponding resolutions

        Retrieved Documents with relevant Resolutions : 
        {context}

        Instructions:
        1. Carefully analyze the customer's query.
        2. Review the retrieved support tickets to find one that closely matches the issue described.
        3. If a relevant resolution is found in the support tickets, return the most suitable resolution.
        4. If no matching issue is found, respond with: "No relevant resolution found in the current knowledge base."

        Respond with only the resolution text if found. And describe the resolution in detail.

        Customer Query : 
        {query}

    """
    return prompt


@st.cache_resource(show_spinner=False)
def load_model():
    model_name = "Qwen/Qwen3-1.7B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    return tokenizer, model

def get_response_qwen3_1_7b(prompt):
    tokenizer, model = load_model()
    messages = [{"role": "user", "content": prompt}]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=32768
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

    try:
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

    print("thinking content:", thinking_content)
    print("content:", content)

    return thinking_content, content

# --- Streamlit UI setup ---
st.set_page_config(page_title="Qwen3-1.7B Chat", page_icon="🤖")
st.title("🤖 Chat with Qwen3-1.7B")


# Initialize chat history
if "history" not in st.session_state:
    st.session_state.history = []

# --- Initialize session state ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Display previous messages ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Chat input ---
user_query = st.chat_input("Type your message...")


if user_query:
    retrieved_docs = query_faiss(user_query)
    modified_query_retrieved_docs = construct_rag_prompt(user_query, retrieved_docs)

    st.chat_message("user").markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})

    # Generate LLM response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            thinking_content, response = get_response_qwen3_1_7b(modified_query_retrieved_docs)
            
            with st.expander("Thinking {Internal Reasoning}"):
                st.markdown(thinking_content)
            st.markdown(response)
            
            st.session_state.messages.append({"role": "assistant", "content": response}) 

        with st.expander("Sources"):
            context = "\n\n".join([doc.page_content for doc in retrieved_docs])
            st.markdown(context)
       
    