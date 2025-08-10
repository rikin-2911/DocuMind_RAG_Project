""" This is a frontend python script for Personal Document Research Assistant Project,
Built using Streamlit. """

# Streamlit page configuration
import streamlit as st
st.set_page_config(page_title="Personal PDF Assistant", layout="centered", page_icon="🔬")

# Importing os and dotenv for API handeling
import os
from dotenv import load_dotenv

# Accessing API keys
load_dotenv()

hf_api_key = os.getenv("HF_API_KEY")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_api_key
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEndpointEmbeddings

# Importing Necessary Langchain libraries
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough # For chaining or LCEL also cab be use
from langchain_core.prompts import PromptTemplate                                   # For prompting
from langchain_core.output_parsers import StrOutputParser                           # For output parsers
from langchain_community.document_loaders import PyPDFLoader                        # For document loader
from langchain.text_splitter import RecursiveCharacterTextSplitter                  # For splitting the docs
from langchain_community.vectorstores import FAISS                                  # For vectore store and also Retriever

""" Streamlit UI"""
# Set website title 
st.title('📚 DocuMind - "Ask. Understand. Summarize."')

# Footer (About App)
st.markdown("### About Application:-")
st.markdown("It's a RAG(Retrieval Augmented Generation) based Application use for Analyzing personal PDF documents and give accurate answers for user's questions or queries.")
st.markdown("It has two main capabilities: \n 1. Question Answering on shared document\n  2. Summarization of shared document.")
st.markdown("Some popular use cases: \n ")
st.write(" 1. Academic Research Assistant (e.g., arxiv, IEEE papers)")
st.write(" 2. Resume Q&A and Summarization")
st.write(" 3. Corporate R&D Knowledge Base")
st.write(" 4. Legal Document Analysis")
st.write(" 5. Policy & Government Document Analysis, etc")

st.markdown("> ## How to use? (Read before Using)")
st.write("1. Upload any of your PDF document like research paper, resume or marksheet (don't worry it doesn't record anything).")
st.write("2. Select one option from the two.")
st.write("3. For Q&A, Ask anything about the document. Hit Enter! Wait for 2-3 Seconds")
st.write("4. For summary, just choose Summarization button.")
st.warning("Questions outside of the document will not be answer properly")
st.write("> ### And here you go! Start Asking. But it is for limited time only (API Usage Constraints🙂). ")
# Upload pdf document
uploaded_file = st.file_uploader("Upload your PDF here :", type=['pdf'])

# Asking for tasks
task = st.selectbox("Choose any one from these two tasks:", ['Question Answering', 'Summarization'])

# user question variable instatianate
question = ""

# If ask for QnA
if task == "Question Answering":
    # Updating user question
    question = st.text_input("Ask a question about the document/pdf:")

# If pdf uploaded
if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    # pdf loader
    pdf_loader = PyPDFLoader('temp.pdf')
    docs = pdf_loader.load()

    # Splitter for chunking documents
    splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    separators=["\n\n", "\n", " ", ""]
    )

    # chunks made by splitter
    chunks = splitter.split_documents(docs)

    #""" Embeddings of chunks and Vector stores"""
    # First we setup our LLM for embedding task
    embedding_llm = HuggingFaceEndpointEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")
    # Now we create a vector store for storing our embeddings/vectors
    vector_store = FAISS.from_documents(chunks, embedding_llm)

    #""" Retrieval of relevant documents by using MMR Retriver"""
    # for retreiving, the process is straight forward  beacuse our data is in the vector store
    # Therefore we use vector store as our retriever 
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k":5}
    )

    #""" Augmentation with retrieved content and user query"""
    # Setting up llm for text generation
    openai_llm = HuggingFaceEndpoint(
        #repo_id="Qwen/Qwen3-235B-A22B-Instruct-2507",
        repo_id="openai/gpt-oss-120b",   
        #repo_id="meta-llama/Llama-3.2-3B-Instruct",
        #repo_id="meta-llama/Llama-3.1-8B-Instruct",
        #repo_id="google/gemma-2-2b-it",
        #repo_id="deepseek-ai/DeepSeek-R1-0528",
        #repo_id="perplexity-ai/r1-1776-distill-llama-70b",
        task="text-generation",
        max_new_tokens=500,
        huggingfacehub_api_token=os.getenv("HF_API_KEY")
    )

    llm = ChatHuggingFace(llm=openai_llm)

    if task == "Question Answering" and question:
        
        # Only question related retrieved content
        def format_docs(retriever_docs):
            context_text = "\n\n".join(doc.page_content for doc in retriever_docs)
            return context_text
    
        # Parallel Chain
        parallel_chain = RunnableParallel({
        'context': retriever | RunnableLambda(format_docs),  # Sequential Chain
        'question':RunnablePassthrough() # question pass as it is
        })

        # prompt for qna
        qna_prompt = PromptTemplate(
            template="""
                You are a QnA chatbot with reasoning and rag capabilities.
                Answer the user's question in concise and accurate manner.
                Include revelant mathematical formulation if present in the context
                Answer ONLY from the provided transcript context.
                If the context is insufficient, just say Insufficient context,
                {context}
                Question: {question}
            """,
            input_variables=['context', 'question'],
            validate_template=True
            )

        # Output Parser
        parser = StrOutputParser()

        # Final chain -> Parallel + Sequential
        final_chain = parallel_chain | qna_prompt | llm | parser
        answer = final_chain.invoke(question)
        st.subheader("Answer:")
        st.success(answer)

    elif task == "Summarization":
        # All chunks because for summary generation
        context_text = "\n\n".join(doc.page_content for doc in chunks)
        summary_prompt=PromptTemplate(
            template="""
                Please summarize the given document context in a accurate manner for good user experience, context as follow:- \n {context}
            """,
            input_variables=['context'],
            validate_template=True
        )

        # Output parser
        parser = StrOutputParser()

        # final chain
        final_prompt = summary_prompt.invoke({'context':context_text})
        if st.button("Summarize"):
            llm_result = llm.invoke(final_prompt)
            summary = parser.invoke(llm_result)
            st.subheader("Document Summary:")
            st.success(summary)


st.write("> #### This application was built on LangChain, LLMs(OpenAI), Python and Streamlit")

# About me
st.markdown("### About Me:-")
st.write("> I'm Rikin Pithadia, a dual-degree student in Data Science & AI from IIT Guwahati and Mechanical Engineering from GEC Gandhinagar having keen interest in various domains like " \
"Internet of Things(IoT), Generative AI, AI Agents and Agentic AI, Machine Learning, Deep Learning, Computer Vision, Natural language Processing (NLP), and many more.../")
st.write("> Contact Info: ")
st.write(">> LinkedIn: [Rikin Pithadia](https://www.linkedin.com/in/rikin-pithadia-20b94729b)")
st.write(">> Mail: [rikinpithadia98@gmail.com]")
st.write(">> GitHub: [rikin-2911](https://www.github.com/rikin-2911) for my other projects")
st.write("> ### You can give your Valuable feedback 📑 about this application on any of this contacts ☝🏼")
st.markdown("### Thank You and Have a Good day or Night!")