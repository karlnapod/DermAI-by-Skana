from dotenv import load_dotenv

# Load environment variables from .env before any module that reads them is imported.
load_dotenv()

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from cancer_classifier  import CancerClassifier
from disease_classifier import DiseaseClassifier
from llm_descriptor     import generate_description
from llm_chatbot        import get_chat_reply


# ---------------------------------------------------------------------------
# Model paths
# ---------------------------------------------------------------------------

CANCER_MODEL_PATH  = os.path.join(os.path.dirname(__file__), "models", "best_model.pth")
DISEASE_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "best_model.pt")

# Global classifier instances — loaded once at startup and reused per request.
cancer_classifier:  CancerClassifier  | None = None
disease_classifier: DiseaseClassifier | None = None


# ---------------------------------------------------------------------------
# Request body schemas
# ---------------------------------------------------------------------------

class DescriptionRequest(BaseModel):
    predicted_class:       str
    predicted_description: str
    confidence:            float
    severity:              int
    probabilities:         dict[str, float]


class ChatMessage(BaseModel):
    role:    str
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]


# ---------------------------------------------------------------------------
# Startup / shutdown
# ---------------------------------------------------------------------------

# The following context manager is responsible for loading both classifiers at
# server startup so inference calls do not pay the model load penalty on the
# first request.
@asynccontextmanager
async def lifespan(app: FastAPI):

    global cancer_classifier, disease_classifier

    if os.path.exists(CANCER_MODEL_PATH):
        try:
            cancer_classifier = CancerClassifier(CANCER_MODEL_PATH)
            print(f"Cancer classifier loaded from {CANCER_MODEL_PATH}")
        except Exception as exc:
            print(f"Failed to load cancer classifier: {exc}")
    else:
        print(
            f"Warning: cancer model not found at {CANCER_MODEL_PATH}. "
            "Place best_model.pth inside the backend/models/ folder."
        )

    if os.path.exists(DISEASE_MODEL_PATH):
        try:
            disease_classifier = DiseaseClassifier(DISEASE_MODEL_PATH)
            print(f"Disease classifier loaded from {DISEASE_MODEL_PATH}")
        except Exception as exc:
            print(f"Failed to load disease classifier: {exc}")
    else:
        print(
            f"Warning: disease model not found at {DISEASE_MODEL_PATH}. "
            "Place best_model.pt inside the backend/models/ folder."
        )

    yield


# ---------------------------------------------------------------------------
# App and middleware
# ---------------------------------------------------------------------------

app = FastAPI(
    title="DermAI by Skana - Backend API",
    description=(
        "Inference API for skin disease classification, "
        "skin cancer detection, and AI-powered descriptions."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

# The following route is responsible for returning a health status response
# so the frontend and monitoring tools can verify the API is reachable.
@app.get("/health")
async def health_check():

    return {
        "status":                "ok",
        "cancer_model_loaded":   cancer_classifier  is not None,
        "disease_model_loaded":  disease_classifier is not None,
    }


# ---------------------------------------------------------------------------
# Inference endpoints
# ---------------------------------------------------------------------------

# The following route is responsible for accepting a dermoscopic image upload,
# running it through the ResNet-50 cancer classifier, and returning the
# prediction, confidence score, and raw malignancy probability.
#
# RATE LIMITING — not yet implemented, enable before any public deployment.
# Recommended: 10 requests per minute per IP via slowapi + Redis/in-memory store.
@app.post("/predict/skin-cancer")
async def predict_skin_cancer(file: UploadFile = File(...)):

    if cancer_classifier is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "The cancer detection model is not loaded. "
                "Place best_model.pth inside backend/models/ and restart."
            ),
        )

    if file.content_type not in ("image/jpeg", "image/png"):
        raise HTTPException(status_code=400, detail="Only JPG and PNG images are accepted.")

    image_bytes = await file.read()
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        return cancer_classifier.predict(image_bytes)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}")


# The following route is responsible for accepting a dermoscopic image upload,
# running it through the ResNet-152 skin disease classifier, and returning the
# predicted class, all 8 class probabilities, and the severity score.
#
# RATE LIMITING — not yet implemented, enable before any public deployment.
# Recommended: 10 requests per minute per IP via slowapi + Redis/in-memory store.
@app.post("/predict/skin-disease")
async def predict_skin_disease(file: UploadFile = File(...)):

    if disease_classifier is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "The skin disease classification model is not loaded. "
                "Place best_model.pt inside backend/models/ and restart."
            ),
        )

    if file.content_type not in ("image/jpeg", "image/png"):
        raise HTTPException(status_code=400, detail="Only JPG and PNG images are accepted.")

    image_bytes = await file.read()
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        return disease_classifier.predict(image_bytes)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}")


# ---------------------------------------------------------------------------
# LLM endpoints
# ---------------------------------------------------------------------------

# The following route is responsible for accepting a disease classification
# result payload and returning an AI-generated plain-language description of
# the predicted condition, generated via Google AI Studio (Gemini).
# The primary API key is tried first; if it is rate-limited (429), the backup
# key is used automatically.
#
# RATE LIMITING — client-side limit of 5 requests per 2 minutes is enforced
# in the frontend. Server-side rate limiting via slowapi should be added
# before any public deployment.
@app.post("/llm/describe-disease")
async def llm_describe_disease(body: DescriptionRequest):

    try:
        description = generate_description(
            predicted_class=       body.predicted_class,
            predicted_description= body.predicted_description,
            confidence=            body.confidence,
            severity=              body.severity,
            probabilities=         body.probabilities,
        )
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Description generation failed: {exc}",
        )

    return {
        "description": description,
        "model":       "gemini-3.1-flash-lite-preview",
    }


# The following route is responsible for receiving a conversation message
# array from the chatbot frontend and returning an AI-generated reply via
# Google AI Studio (Gemma). The full message history is sent on every call
# so the model retains context across turns. The system prompt is embedded
# in the first user message of the history.
#
# RATE LIMITING — client-side limit of 15 messages per session is enforced
# in the frontend. Server-side rate limiting via slowapi should be added
# before any public deployment.
@app.post("/llm/chat")
async def llm_chat(body: ChatRequest):

    if not body.messages:
        raise HTTPException(status_code=400, detail="messages list cannot be empty.")

    messages = [{"role": m.role, "content": m.content} for m in body.messages]

    try:
        reply = get_chat_reply(messages)
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Chat response failed: {exc}",
        )

    return {"reply": reply}
