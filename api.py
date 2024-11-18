from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from utils import get_model_output
from generate_summary import generate_summary
import shutil
import os

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/summarize/")
async def summarize_image(image: UploadFile = File(...), model_name: str = "mrm8488/bert2bert_shared-spanish-finetuned-summarization", lang: str = 'spa'):
    # Save the image to a temporary file
    temp_image_path = f"./temp/{image.filename}"
    os.makedirs(os.path.dirname(temp_image_path), exist_ok=True)
    with open(temp_image_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    # Result    
    extracted_text = get_model_output(temp_image_path, lang)
    summary = generate_summary(extracted_text, model_name)
    
    # Remove the temporary image file
    os.remove(temp_image_path)
    
    # Return the extracted text, summary, and markdown content
    content = {"extracted_text": extracted_text, "summary": summary}
    print(content)
    return JSONResponse(content=content)

@app.post("/summarize-text/")
async def summarize_text(request: Request, model_name: str = "mrm8488/bert2bert_shared-spanish-finetuned-summarization", lang: str = 'spa'):
    payload = await request.json()
    text = payload.get("text")
    
    # Result    
    summary = generate_summary(text, model_name)
    
    # Return the summary and markdown content
    content = {"summary": summary}
    print(content)
    return JSONResponse(content=content)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)