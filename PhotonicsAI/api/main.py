from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

from PhotonicsAI.Photon import llm_api

# Add API metadata
app = FastAPI(
    title="PhotonicsAI API",
    description="API for photonic circuit design and optimization",
    version="1.0.0"
)

class IntentRequest(BaseModel):
    prompt: str

class IntentResponse(BaseModel):
    category_id: int
    response: str

@app.get("/")
async def root():
    """Root endpoint that provides basic API information"""
    return {
        "name": "PhotonicsAI API",
        "version": "1.0.0",
        "description": "API for photonic circuit design and optimization",
        "docs_url": "/docs",
        "redoc_url": "/redoc"
    }

@app.post("/api/v1/intent", response_model=IntentResponse)
async def classify_intent(request: IntentRequest):
    """
    Classifies the input prompt to determine if it's a photonic layout request.
    """
    try:
        # result = llm_api.intent_classification(request.prompt)
        result = llm_api.combined_pipeline(request.prompt)

        return IntentResponse(
            category_id=result.category_id,
            response=result.response
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
