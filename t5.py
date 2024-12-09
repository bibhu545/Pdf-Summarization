from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import fitz  # For extracting text from PDFs
from transformers import T5Tokenizer, T5ForConditionalGeneration
import io
import uvicorn

app = FastAPI()

# Enable CORS for Angular or other frontend frameworks
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],  # Change to your frontend's domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load T5 model and tokenizer
MODEL_NAME = "t5-small"  # You can use "t5-base" or "t5-large" for better results
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

def extract_text_from_pdf(file_bytes):
    """
    Extract text from a PDF file.
    :param file_bytes: PDF content as bytes
    :return: Extracted text
    """
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

def summarize_with_t5(text, max_summary_length):
    """
    Summarize text using the T5 model.
    :param text: Input text to summarize
    :param max_summary_length: Maximum length of the summary
    :return: Summary text
    """
    try:
        # Prepare the input for T5
        input_text = "summarize: " + text
        input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)

        # Generate the summary
        summary_ids = model.generate(
            input_ids,
            max_length=max_summary_length,
            min_length=10,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary
    except Exception as e:
        raise ValueError(f"Error summarizing text with T5: {e}")

@app.post("/summarize")
async def summarize_pdf(file: UploadFile = File(...), word_limit: int = Form(...)):
    """
    Endpoint to process a PDF and summarize it.
    :param file: PDF file uploaded by the user
    :param max_summary_length: Maximum length of the summary
    :return: Summary of the PDF
    """
    try:
        # Read the file content as bytes
        pdf_content = await file.read()

        # Extract text from the PDF
        extracted_text = extract_text_from_pdf(pdf_content)

        # Summarize the extracted text
        summary = summarize_with_t5(extracted_text, word_limit)
        return {"summary": summary}
    except ValueError as ve:
        return JSONResponse(content={"error": str(ve)}, status_code=400)
    except Exception as e:
        return JSONResponse(content={"error": f"Internal server error: {e}"}, status_code=500)


# Run the application
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=4050)