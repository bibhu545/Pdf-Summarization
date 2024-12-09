import io
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
import PyPDF2
from transformers import pipeline
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

# Load summarization model (BART from Hugging Face)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

class SummaryRequest(BaseModel):
    word_limit: int

def extract_text_from_pdf(file_bytes):
    """Extracts text from a PDF file given its bytes."""
    text = ""
    try:
        # Wrap bytes in a BytesIO object
        file_stream = io.BytesIO(file_bytes)
        reader = PyPDF2.PdfReader(file_stream)
        for page in reader.pages:
            text += page.extract_text()
    except Exception as e:
        raise ValueError(f"Error extracting text from PDF: {e}")
    return text

def summarize_text(text, max_words):
    """Summarizes the text to fit within the specified word limit."""
    max_length = int(max_words * 1.5)  # Adjust for token count
    min_length = int(max_words * 0.8)
    summary = summarizer(text, max_length=130, min_length=30, do_sample=False)
    print(summary)
    return summary[0]['summary_text']

@app.post("/summarize")
async def summarize_pdf(file: UploadFile = File(...), word_limit: int = Form(...)):
    try:
        # Read the file content
        pdf_content = await file.read()
        
        # Extract text from PDF
        extracted_text = extract_text_from_pdf(pdf_content)
        
        # Summarize the extracted text
        if len(extracted_text.strip()) == 0:
            return JSONResponse(content={"error": "The uploaded PDF has no readable text."}, status_code=400)
        
        summary = summarize_text(extracted_text, word_limit)
        return {"summary": summary}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


# Run the application
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=4050)
