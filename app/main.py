from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import os
import base64

from .models import Base64ImageRequest, OcrResultResponse, ChequeOcrResponse, CostInfo
from .services import initialize_agent

# FastAPI application setup
app = FastAPI(
    title="Check OCR API",
    description="API for extracting information from bank checks",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this with specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/ocr/check/base64", response_model=OcrResultResponse)
async def process_check_base64(request: Base64ImageRequest):
    """
    Extract information from a base64-encoded bank check image.

    - **image**: Base64-encoded image string
    - **model**: The OpenAI model to use (default: gpt-4o)
    """
    try:
        # Decode base64 string
        image_data = base64.b64decode(request.image)

        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(image_data)
            temp_path = temp_file.name

        # Process image
        result, cost = initialize_agent(temp_path, request.model)

        # Create response
        response = OcrResultResponse(
            chequeOcrResponse=ChequeOcrResponse(**result.get("chequeOcrResponse", {})),
            cost_info=CostInfo(**cost),
        )
        return response

    except base64.binascii.Error:
        raise HTTPException(status_code=400, detail="Invalid base64 image string")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing check: {str(e)}")

    finally:
        if "temp_path" in locals() and os.path.exists(temp_path):
            os.unlink(temp_path)


@app.post("/ocr/check", response_model=OcrResultResponse)
async def process_check_file(file: UploadFile = File(...), model: str = "gpt-4o-mini"):
    """
    Extract information from an uploaded bank check image file.

    - **file**: The image file to process
    - **model**: The OpenAI model to use (default: gpt-4o)
    """
    try:
        # Save uploaded file to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name

        # Process image
        result, cost = initialize_agent(temp_path, model)

        # Create response
        response = OcrResultResponse(
            chequeOcrResponse=ChequeOcrResponse(**result.get("chequeOcrResponse", {})),
            cost_info=CostInfo(**cost),
        )
        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing check: {str(e)}")

    finally:
        if "temp_path" in locals() and os.path.exists(temp_path):
            os.unlink(temp_path)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}
