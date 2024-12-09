from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import fitz  # pymupdf
import openai
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

# OpenAI API key (Set your API key here)

openai.api_key = "key here"

def extract_text_from_large_pdf(file_bytes):
    """Efficiently extracts text from a large PDF using pymupdf."""
    try:
        # Wrap bytes in a BytesIO object
        file_stream = io.BytesIO(file_bytes)
        pdf_document = fitz.open(stream=file_stream, filetype="pdf")

        # Extract text from each page
        text = ""
        for page_number in range(len(pdf_document)):
            page = pdf_document[page_number]
            text += page.get_text() + "\n"  # Concatenate page text

        # Ensure extracted text is not empty
        if not text.strip():
            raise ValueError("The PDF contains no readable text.")
        return text
    except Exception as e:
        raise ValueError(f"Error processing PDF: {e}")

def summarize_text(text, word_limit):
    """Summarizes the given text using the OpenAI ChatCompletion API."""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5",  # You can use "gpt-3.5-turbo" for a cheaper option
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes text."},
                {"role": "user", "content": f"Summarize the following text in {word_limit} words:\n\n{text}"}
            ],
            max_tokens=word_limit * 2,
            temperature=0.5,
        )
        summary = response.choices[0].message['content'].strip()
        return summary
    except Exception as e:
        raise ValueError(f"Error generating summary: {e}")

@app.post("/summarize")
async def summarize_pdf(file: UploadFile = File(...), word_limit: int = Form(...)):
    """Endpoint to process and summarize a PDF."""
    try:
        # Read the file content as bytes
        pdf_content = await file.read()

        # Extract text from the large PDF
        extracted_text = extract_text_from_large_pdf(pdf_content)

        # Summarize the extracted text
        summary = summarize_text(extracted_text, word_limit)
        return {"summary": summary}
    except ValueError as ve:
        return JSONResponse(content={"error": str(ve)}, status_code=400)
    except Exception as e:
        return JSONResponse(content={"error": f"Internal server error: {e}"}, status_code=500)


# Run the application
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=4050)