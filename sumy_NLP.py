from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import fitz  # PyMuPDF for PDF text extraction
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
import io
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def extract_text_from_pdf(file_bytes):
    """Extract text from a PDF file."""
    try:
        file_stream = io.BytesIO(file_bytes)
        pdf_document = fitz.open(stream=file_stream, filetype="pdf")

        # Extract text from each page
        text = ""
        for page_number in range(len(pdf_document)):
            page = pdf_document[page_number]
            text += page.get_text() + "\n"  # Append text from each page

        if not text.strip():
            raise ValueError("The PDF contains no readable text.")
        return text
    except Exception as e:
        raise ValueError(f"Error processing PDF: {e}")

def summarize_text_with_sumy(text, sentence_count=5):
    """Summarize text using Sumy."""
    try:
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizer = LsaSummarizer()
        summary = summarizer(parser.document, sentence_count)

        # Combine summarized sentences into a single string
        summarized_text = " ".join(str(sentence) for sentence in summary)
        return summarized_text
    except Exception as e:
        raise ValueError(f"Error summarizing text with Sumy: {e}")

@app.post("/summarize")
async def summarize_pdf(file: UploadFile = File(...), sentence_count: int = Form(5)):
    """
    Endpoint to process a PDF and summarize it.
    :param file: PDF file uploaded by the user
    :param sentence_count: Number of sentences for the summary
    :return: Summary of the PDF
    """
    try:
        # Read the file content as bytes
        pdf_content = await file.read()

        # Extract text from the PDF
        extracted_text = extract_text_from_pdf(pdf_content)

        # Summarize the extracted text
        summary = summarize_text_with_sumy(extracted_text, sentence_count=sentence_count)
        return {"summary": summary}
    except ValueError as ve:
        return JSONResponse(content={"error": str(ve)}, status_code=400)
    except Exception as e:
        return JSONResponse(content={"error": f"Internal server error: {e}"}, status_code=500)



# Run the application
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=4050)