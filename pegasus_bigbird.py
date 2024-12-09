from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import fitz  # PyMuPDF for PDF text extraction
from transformers import BigBirdPegasusForConditionalGeneration, AutoTokenizer
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

# Load BigBird-Pegasus model and tokenizer
MODEL_NAME = "google/bigbird-pegasus-large-arxiv"  # Pretrained for long-text summarization
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = BigBirdPegasusForConditionalGeneration.from_pretrained(MODEL_NAME)

def extract_text_from_pdf(file_bytes):
    """
    Extract text from a PDF file.
    :param file_bytes: PDF content as bytes
    :return: Extracted text
    """
    try:
        pdf_stream = io.BytesIO(file_bytes)
        pdf_document = fitz.open(stream=pdf_stream, filetype="pdf")
        text = ""
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            text += page.get_text()
        if not text.strip():
            raise ValueError("The PDF contains no readable text.")
        return text
    except Exception as e:
        raise ValueError(f"Error extracting text from PDF: {e}")

def summarize_text(text, max_summary_length):
    """
    Summarize text using BigBird-Pegasus.
    :param text: Input text to summarize
    :param max_summary_length: Maximum length of the summary
    :return: Summary text
    """
    try:
        inputs = tokenizer(
            text, return_tensors="pt", max_length=4096, truncation=True
        )  # BigBird-Pegasus handles up to 4096 tokens
        summary_ids = model.generate(
            inputs["input_ids"],
            max_length=max_summary_length,
            min_length=10,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True,
        )
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary
    except Exception as e:
        raise ValueError(f"Error generating summary: {e}")

@app.post("/summarize")
async def summarize_pdf(
    file: UploadFile = File(...), max_summary_length: int = Form(200)
):
    """
    Endpoint to process a PDF and summarize it.
    :param file: PDF file uploaded by the user
    :param max_summary_length: Maximum length of the summary
    :return: Summary of the PDF
    """
    try:
        # Read the file content
        pdf_content = await file.read()

        # Extract text from the PDF
        extracted_text = extract_text_from_pdf(pdf_content)

        # Summarize the text
        summary = summarize_text(extracted_text, max_summary_length)

        return {"summary": summary}
    except ValueError as ve:
        return JSONResponse(content={"error": str(ve)}, status_code=400)
    except Exception as e:
        return JSONResponse(content={"error": f"Internal server error: {e}"}, status_code=500)
    


# Run the application
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=4050)
