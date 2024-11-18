from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from utils import get_model_output
from generate_summary import generate_summary
import shutil
import os

app = FastAPI()

@app.post("/summarize/")
async def summarize_image(image: UploadFile = File(...), model_name: str = "mrm8488/bert2bert_shared-spanish-finetuned-summarization"):
    # Save the image to a temporary file
    temp_image_path = f"./temp/{image.filename}"
    os.makedirs(os.path.dirname(temp_image_path), exist_ok=True)
    with open(temp_image_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    # Result    
    extracted_text = get_model_output(temp_image_path)
    summary = generate_summary(extracted_text, model_name)
    
    # Remove the temporary image file
    os.remove(temp_image_path)
    
    # Return the extracted text and summary
    return JSONResponse(content={"extracted_text": extracted_text, "summary": summary})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)