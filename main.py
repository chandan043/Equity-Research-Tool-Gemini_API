import os
import requests
from bs4 import BeautifulSoup
import streamlit as st
from PyPDF2 import PdfReader
import google.generativeai as genai
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

class GeminiAPI:
    def __init__(self):
        self.data = ""
        self.urls = []

    def set_urls(self, urls):
        self.urls = urls
        context = ""
        # Extract page content from each URL
        for url in self.urls:
            try:
                response = requests.get(url)
                response.raise_for_status()  # Check for HTTP errors
                soup = BeautifulSoup(response.text, 'html.parser')
                # Extract all paragraph text
                paragraphs = soup.find_all('p')
                page_content = " ".join([para.get_text() for para in paragraphs])
                context += page_content + " "
            except requests.exceptions.RequestException as e:
                st.error(f"Error fetching URL {url}: {e}")
        self.data = context

    def set_pdf(self, pdf_file):
        context = ""
        pdf_reader = PdfReader(pdf_file)
        for page in pdf_reader.pages:
            context += page.extract_text() + " "
        self.data += context

    def set_question(self, question):
        self.question = question

    def process(self):
        input_text = f'''Act as an expert question-answering model and text summarizer. 
        Your task is to provide the most accurate and concise answers to the question based on the provided context. 
        Use the context to form a well-informed, clear, and precise response.
        Question: {self.question} 
        Context: {self.data}'''
        
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(input_text)
        return response.text 

def main():
    # Page configuration
    st.set_page_config(page_title="Equity Research Tool", layout="wide")

    # Sidebar for URLs and instructions
    with st.sidebar:
        
        # Article URLs
        st.header("📁 Article URLs")
        urls = []
        for i in range(3):
            url = st.text_input(f"URL {i+1}", placeholder=f"Enter URL {i+1}")
            if url:
                urls.append(url)

        # PDF Upload
        st.header("📄 Upload PDF")
        uploaded_pdf = st.file_uploader("Upload a PDF document", type=["pdf"], help="Alternatively, you can upload a PDF file for analysis.")

        # Proceed Button
        if st.button("➡ Proceed", help="Click to confirm URLs or upload before proceeding."):
            if not urls and not uploaded_pdf:
                st.error("❌ Please provide at least one URL or upload a PDF file.")
            elif uploaded_pdf:
                st.success("✅ PDF uploaded successfully. Processing...")
            else:
                st.success("✅ URLs confirmed. Ready to proceed.")

    # Main panel for question and answers
    st.markdown(
        """
        <h1 style='text-align: center; color: #4CAF50;'>
            📊 Equity Research Question Answering Tool
        </h1>
        """,
        unsafe_allow_html=True
    )
    st.markdown(
        """
        <p style='font-size: 18px; text-align: center;'>
            This tool uses AI to extract relevant answers from news articles or uploaded PDFs.
        </p>
        """,
        unsafe_allow_html=True
    )

    # Question input
    question_input = st.text_input(
        "🔎 Enter your question:", placeholder="Type your question here...", key="question_input",
        help="Ask any question related to the articles or PDF you provided."
    )

    # Get Answer button
    if st.button("🧠 Get Answer", help="Click to process your question and find answers."):
        if not urls and not uploaded_pdf:
            st.error("❌ Please provide at least one URL or upload a PDF file.")
        elif not question_input:
            st.error("❌ Please enter a question.")
        else:
            with st.spinner("⏳ Processing... Please wait."):
                # Use GeminiAPI methods
                model = GeminiAPI()
                if urls:
                    model.set_urls(urls)
                if uploaded_pdf:
                    model.set_pdf(uploaded_pdf)
                model.set_question(question_input)

                # Process and get the answer
                answer = model.process()

            # Display the result
            if answer is not None:
                st.success("✅ Answer found!")
                st.markdown(
                    f"<div style='color: green; font-size: 18px;'><b>Answer:</b> {answer}</div>",
                    unsafe_allow_html=True
                )
            else:
                st.warning("⚠️ No relevant context found.")

    # Footer
    st.sidebar.markdown("---")
    with st.sidebar.expander("ℹ Tips for Effective Usage", expanded=False):
        st.markdown(
            """
            <ul style='font-size: 14px; color: #6c757d;'>
            <li>Provide accurate and accessible URLs for better results.</li>
            <li>Frame your questions clearly to maximize relevance.</li>
            <li>Visit our <a href='https://example.com/guide' target='_blank'>User Guide</a> for detailed help.</li>
            </ul>
            """,
            unsafe_allow_html=True
        )

if __name__ == "__main__":
    main()
