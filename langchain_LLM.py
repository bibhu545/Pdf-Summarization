import os
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import fitz  # PyMuPDF for PDF processing
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
import io
import uvicorn
import openai

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

openai_api_key = os.environ.get("OPENAI_API_KEY")  

if not openai_api_key:
    raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")

openai.api_key = openai_api_key

def extract_text_from_pdf(file_bytes):
    """
    Extract text from a PDF file using PyMuPDF.
    :param file_bytes: PDF content in bytes
    :return: Extracted text as a string
    """
    try:
        pdf_stream = io.BytesIO(file_bytes)
        pdf_document = fitz.open(stream=pdf_stream, filetype="pdf")
        text = ""
        for page in pdf_document:
            text += page.get_text()
        if not text.strip():
            raise ValueError("The PDF contains no readable text.")
        return text
    except Exception as e:
        raise ValueError(f"Error extracting text from PDF: {e}")


def summarize_text_with_openai(text, max_summary_length):
    """
    Summarize the extracted text using OpenAI's API.
    :param text: Text to summarize
    :param max_summary_length: Maximum length of the summary in words
    :return: Summarized text
    """
    prompt_template = f"Summarize the following text in {max_summary_length} words:\n\n{text}"

    # Use OpenAI's ChatCompletion API
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",  # or gpt-4 or other models
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt_template},
            ],
            max_tokens=150,  # Limit the summary length
            temperature=0.7
        )
        summary = response['choices'][0]['message']['content'].strip()
        return summary
    except Exception as e:
        raise ValueError(f"Error generating summary: {e}")


@app.post("/summarize")
async def summarize_pdf(
    file: UploadFile = File(...), max_summary_length: int = Form(200)
):
    """
    API endpoint for summarizing a PDF file.
    :param file: Uploaded PDF file
    :param max_summary_length: Desired maximum length of the summary
    :return: Summary as JSON
    """
    try:
        # Read PDF content
        pdf_content = await file.read()

        # Extract text from PDF
        extracted_text = extract_text_from_pdf(pdf_content)

        # Summarize the extracted text
        summary = summarize_text_with_openai(extracted_text, max_summary_length)

        return {"summary": summary}
    except ValueError as ve:
        return JSONResponse(content={"error": str(ve)}, status_code=400)
    except Exception as e:
        return JSONResponse(content={"error": f"Internal server error: {e}"}, status_code=500)

# Run the application
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=4050)
