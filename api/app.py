"""
FastAPI Web Application for DocStudio
Provides REST API endpoints for document summarization
"""

import os
import tempfile
from typing import Optional, List
from pathlib import Path
import logging

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.main import DocStudio, create_summarization_config
from src.summarizer import SummarizationConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="DocStudio API",
    description="Powerful abstractive document summarization API",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize DocStudio
doc_studio = DocStudio()


# Pydantic models for request/response
class SummarizationRequest(BaseModel):
    text: str = Field(..., description="Text to summarize")
    use_openai: bool = Field(False, description="Whether to use OpenAI for summarization")
    max_length: int = Field(512, description="Maximum length of summary")
    min_length: int = Field(50, description="Minimum length of summary")
    temperature: float = Field(1.0, description="Sampling temperature")
    num_beams: int = Field(4, description="Number of beams for beam search")


class SummarizationResponse(BaseModel):
    success: bool
    summary: str
    text_stats: dict
    summary_metadata: dict
    processing_time_seconds: float
    error: Optional[str] = None


class DocumentResponse(BaseModel):
    success: bool
    summary: str
    file_info: dict
    document_stats: dict
    summary_metadata: dict
    processing_time_seconds: float
    error: Optional[str] = None


class SystemInfoResponse(BaseModel):
    supported_formats: List[str]
    available_models: dict
    version: str


@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information."""
    return {
        "message": "DocStudio API - Document Summarization Service",
        "version": "1.0.0",
        "endpoints": {
            "/docs": "API documentation",
            "/summarize/text": "Summarize text input",
            "/summarize/document": "Summarize uploaded document",
            "/system/info": "Get system information"
        }
    }


@app.post("/summarize/text", response_model=SummarizationResponse)
async def summarize_text(request: SummarizationRequest):
    """
    Summarize text input.
    
    Args:
        request: Summarization request with text and parameters
        
    Returns:
        Summarization result with summary and metadata
    """
    try:
        # Create configuration
        config = create_summarization_config(
            max_length=request.max_length,
            min_length=request.min_length,
            temperature=request.temperature,
            num_beams=request.num_beams
        )
        
        # Process summarization
        result = doc_studio.summarize_text(
            request.text, 
            use_openai=request.use_openai, 
            config=config
        )
        
        return SummarizationResponse(**result)
        
    except Exception as e:
        logger.error(f"Text summarization error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/summarize/document", response_model=DocumentResponse)
async def summarize_document(
    file: UploadFile = File(...),
    use_openai: bool = Form(False),
    max_length: int = Form(512),
    min_length: int = Form(50),
    temperature: float = Form(1.0),
    num_beams: int = Form(4)
):
    """
    Summarize uploaded document.
    
    Args:
        file: Uploaded document file
        use_openai: Whether to use OpenAI for summarization
        max_length: Maximum length of summary
        min_length: Minimum length of summary
        temperature: Sampling temperature
        num_beams: Number of beams for beam search
        
    Returns:
        Document summarization result
    """
    try:
        # Validate file format
        file_extension = Path(file.filename).suffix.lower()
        supported_formats = doc_studio.get_supported_formats()
        
        if file_extension not in supported_formats:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file format. Supported: {supported_formats}"
            )
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            # Create configuration
            config = create_summarization_config(
                max_length=max_length,
                min_length=min_length,
                temperature=temperature,
                num_beams=num_beams
            )
            
            # Process document
            result = doc_studio.process_document(
                temp_file_path, 
                use_openai=use_openai, 
                config=config
            )
            
            return DocumentResponse(**result)
            
        finally:
            # Clean up temporary file
            os.unlink(temp_file_path)
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document summarization error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/system/info", response_model=SystemInfoResponse)
async def get_system_info():
    """
    Get system information including supported formats and available models.
    
    Returns:
        System information
    """
    try:
        return SystemInfoResponse(
            supported_formats=doc_studio.get_supported_formats(),
            available_models=doc_studio.get_available_models(),
            version="1.0.0"
        )
    except Exception as e:
        logger.error(f"System info error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "DocStudio API"}


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 