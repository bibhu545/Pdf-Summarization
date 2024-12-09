from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import fitz  # pymupdf
import io
from transformers import BartForConditionalGeneration, BartTokenizer
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the BART model and tokenizer
model_name = "facebook/bart-large-cnn"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

def extract_text_from_large_pdf(file_bytes):
    """Extract text from a large PDF using pymupdf."""
    try:
        file_stream = io.BytesIO(file_bytes)
        pdf_document = fitz.open(stream=file_stream, filetype="pdf")

        # Extract text from each page
        text = ""
        for page_number in range(len(pdf_document)):
            page = pdf_document[page_number]
            text += page.get_text() + "\n"  # Concatenate page text

        if not text.strip():
            raise ValueError("The PDF contains no readable text.")
        return text
    except Exception as e:
        raise ValueError(f"Error processing PDF: {e}")

def split_text_into_chunks(text, max_chunk_size=1024):
    """Split text into chunks to handle token limits for BART."""
    words = text.split()
    for i in range(0, len(words), max_chunk_size):
        yield " ".join(words[i:i + max_chunk_size])

def summarize_text_with_bart(text, max_length, min_length):
    """Summarize text using BART."""
    try:
        inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
        summary_ids = model.generate(inputs, max_length=max_length, min_length=min_length, length_penalty=2.0, num_beams=4, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary
    except Exception as e:
        raise ValueError(f"Error summarizing text with BART: {e}")

def summarize_large_text_with_bart(text, max_length, min_length):
    """Summarize large text by splitting it into chunks."""
    chunk_summaries = []
    for chunk in split_text_into_chunks(text):
        summary = summarize_text_with_bart(chunk, max_length=max_length, min_length=min_length)
        chunk_summaries.append(summary)
    # Combine chunk summaries
    final_summary = " ".join(chunk_summaries)
    return summarize_text_with_bart(final_summary, max_length=max_length, min_length=min_length)  # Final concise summary

@app.post("/summarize")
async def summarize_pdf(file: UploadFile = File(...), word_limit: int = Form(...)):
    """Endpoint to process and summarize a PDF."""
    try:
        # Read the file content as bytes
        pdf_content = await file.read()

        # Extract text from the large PDF
        extracted_text = extract_text_from_large_pdf(pdf_content)

        # Summarize the extracted text with BART
        max_length = int(word_limit * 1.5)  # Adjust for token count
        min_length = int(word_limit * 0.8)
        summary = summarize_large_text_with_bart(extracted_text, max_length=max_length, min_length=min_length)
        return {"summary": summary}
    except ValueError as ve:
        return JSONResponse(content={"error": str(ve)}, status_code=400)
    except Exception as e:
        return JSONResponse(content={"error": f"Internal server error: {e}"}, status_code=500)

# Run the application
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=4050)