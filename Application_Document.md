# DermAI by Skana — Application Document

---

## Table of Contents

1. [Codebase Infrastructure Implementation](#1-codebase-infrastructure-implementation)
2. [AI Subsystems Pipeline Design](#2-ai-subsystems-pipeline-design)
3. [The Goal](#3-the-goal)
4. [The Design](#4-the-design)
5. [The Integration](#5-the-integration)
6. [The Features](#6-the-features)
7. [The Stakeholders](#7-the-stakeholders)
8. [The Deployment](#8-the-deployment)
9. [Future Prospects and Improvements](#9-future-prospects-and-improvements)

---

## 1. Codebase Infrastructure Implementation

### Overview

DermAI by Skana is a full-stack web application built with a clear separation between the frontend (what the user sees) and the backend (where AI inference and API calls happen). The two sides communicate over HTTP through a REST API.

### Folder Structure

```
DermAI by Skana/
│
├── backend/                         # Python API server
│   ├── main.py                      # FastAPI app — route definitions, startup
│   ├── cancer_classifier.py         # ResNet-50 cancer inference wrapper
│   ├── disease_classifier.py        # ResNet-152 disease inference wrapper
│   ├── llm_descriptor.py            # Gemini-based AI description generator
│   ├── llm_chatbot.py               # Gemma-based chatbot handler
│   ├── requirements.txt             # All Python package dependencies
│   ├── .env                         # Secret API keys (never shared)
│   ├── .env.example                 # Safe template for new deployments
│   ├── models/
│   │   ├── best_model.pth           # Trained cancer model weights
│   │   └── best_model.pt            # Trained disease model weights
│   └── disease_model/               # Disease model training artefacts
│       ├── config.json              # Full training configuration
│       ├── summary.json             # Final metrics and run summary
│       ├── history.csv              # Per-epoch training/validation metrics
│       ├── classification_report.csv
│       └── confusion_matrix.csv
│
├── frontend/                        # React web application
│   ├── src/
│   │   ├── App.jsx                  # Root component, routing setup
│   │   ├── main.jsx                 # React entry point
│   │   ├── components/
│   │   │   ├── Navbar.jsx           # Persistent navigation bar
│   │   │   └── ChatBot.jsx          # Floating AI chatbot overlay
│   │   ├── pages/
│   │   │   ├── LandingPage.jsx      # Hero / home page
│   │   │   ├── DisclaimersPage.jsx  # Medical disclaimer cards
│   │   │   ├── SkinCancerPage.jsx   # Cancer detection UI
│   │   │   ├── SkinDiseasePage.jsx  # Disease classification UI
│   │   │   └── AboutPage.jsx        # Project and team page
│   │   └── styles/
│   │       └── main.css             # Single global stylesheet
│   ├── public/assets/images/        # Static assets (logo)
│   ├── index.html                   # HTML shell
│   ├── package.json                 # Node dependencies and scripts
│   └── vite.config.js               # Build config and API proxy
│
└── Documentation/
    ├── README.md
    ├── Application_Document.md
    └── User_Manual.md
```

### Environment

| Component | Technology | Version |
|---|---|---|
| Python runtime | conda environment | 3.12.7 |
| Web framework | FastAPI | ≥ 0.111.0 |
| ASGI server | Uvicorn | ≥ 0.30.0 |
| Deep learning | PyTorch (CPU) | ≥ 2.4.0 |
| Image processing | Pillow | ≥ 10.3.0 |
| LLM SDK | google-genai | ≥ 1.0.0 |
| Frontend bundler | Vite | 6.x |
| UI library | React | 18.x |
| Client routing | React Router DOM | 6.x |
| OS (developed on) | Windows 10/11 | — |

### Key Design Principles

The codebase follows a small set of rules that were set at the start and held throughout:

- **Single CSS file.** All styles live in `main.css`. CSS Custom Properties (design tokens) are defined in `:root` and reused throughout. No inline styles, no CSS modules.
- **One concern per file.** Each Python module handles exactly one responsibility (`cancer_classifier.py` only does cancer inference, `llm_descriptor.py` only does description generation, and so on).
- **Strict environment variable management.** No API key ever appears in source code. All secrets are read from `.env` via `python-dotenv`, loaded before any dependent module is imported.
- **Models loaded once at startup.** Both PyTorch models are instantiated during the FastAPI lifespan event and held in memory as global instances. This prevents the heavy load penalty from hitting on the first request.

---

## 2. AI Subsystems Pipeline Design

### 2.1 Skin Cancer Detection — ResNet-50

**The Model**

| Property | Detail |
|---|---|
| Architecture | ResNet-50 (pretrained on ImageNet, fine-tuned) |
| Task | Binary classification — benign vs. malignant |
| Parameters | ~25.6 million |
| Classifier head | Dropout(0.4) → Linear(2048 → 256) → ReLU → Dropout(0.2) → Linear(256 → 1) |
| Output | Single logit, passed through sigmoid to get malignancy probability |

The standard ResNet-50 fully-connected head (a single linear layer to 1000 ImageNet classes) is replaced with a custom two-layer head that narrows to a single output neuron. The sigmoid activation converts the raw logit to a probability between 0 and 1, where values above 0.50 indicate malignancy.

**The Pipeline**

```
User uploads image (JPG/PNG)
        ↓
FastAPI receives UploadFile, reads raw bytes
        ↓
PIL opens bytes as RGB image
        ↓
Inference transforms applied:
  • Resize to 224 × 224
  • ToTensor (normalises pixel values to [0, 1])
  • Normalize with ImageNet mean/std
        ↓
Tensor passed through ResNet-50 (no gradient, eval mode)
        ↓
sigmoid(logit) → prob_malignant
        ↓
Thresholding:
  • prob_malignant >= 0.50 → raw = "malignant", confidence = prob_malignant
  • prob_malignant <  0.50 → raw = "benign",    confidence = 1 - prob_malignant
        ↓
Confidence gate:
  • confidence >= 0.40 → return raw prediction
  • confidence <  0.40 → return "inconclusive"
        ↓
JSON response: { prediction, confidence, prob_malignant }
        ↓
Frontend renders result card and confidence ring
```

**Key thresholds:**

| Threshold | Value | Purpose |
|---|---|---|
| `CLASSIFICATION_THRESHOLD` | 0.50 | Malignant vs. benign decision boundary |
| `CONFIDENCE_THRESHOLD` | 0.40 | Minimum confidence to report a result |

---

### 2.2 Skin Disease Classification — ResNet-152

**The Model**

| Property | Detail |
|---|---|
| Architecture | ResNet-152 (pretrained on ImageNet, fine-tuned) |
| Task | 8-class classification (ISIC 2019 categories) |
| Parameters | ~60.2 million |
| Classifier head | Single Linear layer (2048 → 8) |
| Output | 8 logits, passed through softmax for class probabilities |
| Dataset | ISIC 2019 skin lesion dataset |

The 8 disease classes and their clinical severity scores (0–100):

| Code | Full Name | Severity |
|---|---|---|
| MEL | Melanoma | 95 |
| SCC | Squamous Cell Carcinoma | 78 |
| BCC | Basal Cell Carcinoma | 65 |
| AK | Actinic Keratosis | 52 |
| VASC | Vascular Lesion | 38 |
| BKL | Benign Keratosis | 18 |
| NV | Melanocytic Nevus | 15 |
| DF | Dermatofibroma | 10 |

**Training Configuration (sourced from `disease_model/config.json`):**

| Parameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 0.0005 |
| Weight decay | 0.0001 |
| Batch size | 32 |
| Max epochs | 100 |
| Early stopping patience | 5 |
| LR scheduler | ReduceLROnPlateau (factor 0.5, patience 2) |
| Loss function | Cross-entropy |
| Class sampler | Weighted (to handle class imbalance) |
| Input size | 384 × 384 px |

**Data augmentation used during training:**

- Horizontal flip (p = 0.50)
- Vertical flip (p = 0.20)
- Random rotation ±15°
- Color jitter (brightness, contrast, saturation: ±0.10; hue: ±0.02)

**Model Performance (from `disease_model/summary.json`):**

| Metric | Value |
|---|---|
| Best epoch | 27 of 100 |
| Test accuracy | 86.2% |
| Macro F1 | 81.1% |
| Macro precision | 86.5% |
| Macro recall | 77.1% |
| Total training time | ~17.5 hours |

**The Pipeline**

```
User uploads image (JPG/PNG)
        ↓
FastAPI receives UploadFile, reads raw bytes
        ↓
PIL opens bytes as RGB image
        ↓
Inference transforms applied:
  • Resize to 384 × 384
  • ToTensor
  • Normalize with ImageNet mean/std
        ↓
Tensor passed through ResNet-152 (no gradient, eval mode)
        ↓
softmax(logits) → 8 class probabilities
        ↓
argmax → predicted class index → CLASS_ORDER lookup
        ↓
SEVERITY_SCORES lookup for predicted class
        ↓
JSON response: {
  predicted_class, predicted_description,
  confidence, severity, probabilities (all 8)
}
        ↓
Frontend renders probability bars, result pill, severity bar
```

---

### 2.3 LLM Descriptor — Gemini

**The Model**

| Property | Detail |
|---|---|
| Model | gemini-3.1-flash-lite-preview |
| Provider | Google AI Studio |
| Temperature | 0.7 |
| Max output tokens | 600 |
| Output format | Plain-text paragraph |

**The Pipeline**

```
User clicks "Generate Description" on disease results page
        ↓
Frontend checks rate limit (5 per 2 minutes, tracked client-side)
        ↓
POST /api/llm/describe-disease with:
  { predicted_class, predicted_description, confidence, severity, probabilities }
        ↓
Backend builds prompt:
  • System instructions (domain scope, output rules)
  • Prediction context (condition name, confidence %, severity, all 8 class probabilities)
        ↓
Calls Gemini API with primary key (GEMINI_PRIMARY_API_KEY)
        ↓
If 429 rate limit error → retries automatically with backup key (GEMINI_BACKUP_API_KEY)
        ↓
Returns { description, model }
        ↓
Frontend displays description text in the AI Description box
```

---

### 2.4 LLM Chatbot — Gemma

**The Model**

| Property | Detail |
|---|---|
| Model | gemma-3-1b-it |
| Provider | Google AI Studio |
| Temperature | 0.7 |
| Max output tokens | 500 |
| Scope | Dermatology-only with injection/security guardrails |

**The Pipeline**

```
User types message and presses Enter or Send
        ↓
Frontend validates: not empty, ≤ 500 chars, session < 15 messages
        ↓
POST /api/llm/chat with full message history:
  [
    { role: "user",      content: "..." },
    { role: "assistant", content: "..." },
    { role: "user",      content: "new message" }
  ]
        ↓
Backend builds contents array:
  • First user message: system prompt embedded at the top
  • All previous turns passed as alternating user/model Content objects
  • Final user message appended last
        ↓
Gemma API call with full conversation history
        ↓
Returns { reply }
        ↓
Frontend appends AI bubble to message list, auto-scrolls
```

**Context strategy:**
The full conversation history is sent with every request. The system prompt is embedded only in the first user message turn. This means the model always has context from the entire session while keeping the prompt structure clean.

---

## 3. The Goal

### Purpose

DermAI by Skana aims to make AI-driven skin health awareness more accessible. The application provides a structured, easy-to-use interface for users to upload images of skin lesions and receive:

- A classification result (benign, malignant, or one of 8 specific disease categories)
- A confidence score indicating how certain the model is
- A clinical severity score for prioritising follow-up action
- An AI-generated plain-language explanation of the result
- A conversational assistant for follow-up dermatology questions

### The Problem Being Addressed

Skin conditions — ranging from harmless moles to life-threatening melanoma — are extremely common, yet access to specialist dermatology care is limited in many parts of the world. Long waiting times and the cost of consultation mean many people go without timely advice. DermAI acts as a first-pass screening tool that helps users understand whether their skin condition may warrant urgent professional attention.

### Business and Research Prospects

From a research perspective, the system demonstrates that consumer-grade hardware (CPU inference) can serve practical deep learning predictions through a clean, user-facing interface. From a business perspective, the application has potential for use as:

- A pre-consultation screening tool for telemedicine platforms
- An educational resource for medical students and the general public
- A foundation for a premium diagnostic-assist subscription service

---

## 4. The Design

### User Interface

The application uses a minimalistic, professional aesthetic based on a warm cream-white background (`#F6F8F8`) with teal accent tones (`#2AADA8`, `#5EC9C3`) and muted text. Every page shares the same persistent navigation bar and floating chatbot button so the user always has access to both navigation and Karl.

**Color Palette:**

| Token | Hex | Usage |
|---|---|---|
| `--color-bg-page` | `#F6F8F8` | Main background |
| `--color-bg-card` | `#EDF3F3` | Cards and panels |
| `--color-teal-deep` | `#2AADA8` | Primary accent, buttons |
| `--color-teal-mid` | `#5EC9C3` | Secondary accent |
| `--color-text-hero` | `#3A3D40` | Headings |
| `--color-text-sub` | `#8A9299` | Labels and captions |
| `--color-accent-red` | `#C8706E` | Warnings, risk indicators |

**Pages:**

| Page | Route | Purpose |
|---|---|---|
| Landing | `/` | Hero title with gradient text, subtitle, entry point |
| Disclaimers | `/disclaimers` | 5 medical disclaimer cards with warning icons |
| Skin Cancer | `/skin-cancer` | Cancer detection upload and results |
| Skin Disease | `/skin-disease` | Disease classification upload, probability bars, AI description |
| About | `/about` | Project background and team cards |

### Frontend Architecture

The frontend is a Single Page Application (SPA) built with React and Vite. Client-side routing is handled by React Router DOM so the page never hard-reloads.

**Component tree:**
```
App.jsx
  ├── BackgroundBlobs        (decorative teal gradient blobs)
  ├── Navbar                 (persistent across all routes)
  ├── Routes
  │   ├── /              → LandingPage
  │   ├── /disclaimers   → DisclaimersPage
  │   ├── /skin-cancer   → SkinCancerPage
  │   ├── /skin-disease  → SkinDiseasePage
  │   └── /about         → AboutPage
  └── ChatBot                (floating overlay, always rendered)
```

**State management:** All state is local to components using React's `useState` and `useRef` hooks. There is no global state store — the app is small enough that component-level state is sufficient.

**Data flow (frontend → backend):**
```
fetch('/api/...')
    ↓
Vite dev proxy (/api → http://localhost:8000)
    ↓
FastAPI backend
    ↓
JSON response
    ↓
Component state update → React re-render
```

The Vite proxy (`vite.config.js`) rewrites `/api` requests to `localhost:8000` during development so the frontend and backend can run on separate ports without CORS issues.

### Backend Architecture

The backend is a FastAPI application with a linear structure. All routes are defined in `main.py`. Inference and LLM logic are isolated in separate modules.

**Startup sequence:**
```python
# main.py
load_dotenv()             # Load API keys from .env
# FastAPI lifespan event:
CancerClassifier.load()   # ResNet-50 loaded to memory
DiseaseClassifier.load()  # ResNet-152 loaded to memory
# Server begins accepting requests
```

**API routes:**

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Returns server status and model load state |
| POST | `/predict/skin-cancer` | Accepts image, returns binary classification |
| POST | `/predict/skin-disease` | Accepts image, returns 8-class classification |
| POST | `/llm/describe-disease` | Accepts result payload, returns AI description |
| POST | `/llm/chat` | Accepts message history, returns chatbot reply |

### Overall Data Flow

```
User action (upload / button / message)
        ↓
React component updates state + triggers fetch
        ↓
HTTP POST to /api/* (proxied by Vite to FastAPI)
        ↓
FastAPI route validates request (Pydantic model or UploadFile)
        ↓
Inference or LLM module called
        ↓
JSON response returned to frontend
        ↓
React state updated → component re-renders with results
```

---

## 5. The Integration

### Deep Learning Models → FastAPI

Both classifiers are Python classes that wrap the PyTorch model. They expose a single `predict(image_bytes: bytes) -> dict` method. FastAPI reads the raw bytes from the uploaded file and passes them directly to the classifier.

```python
# FastAPI route
image_bytes = await file.read()
result = cancer_classifier.predict(image_bytes)
return result  # FastAPI auto-serialises dict to JSON
```

Inside the classifier, PIL converts the bytes to an RGB image, the inference transforms are applied, and the tensor is passed through the model. No file is ever written to disk — everything is processed in memory.

```python
# Inside cancer_classifier.py
img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
tensor = INFERENCE_TRANSFORMS(img).unsqueeze(0)
logit = self.model(tensor)
```

### LLM Modules → FastAPI

The LLM modules are similarly isolated. `main.py` imports `generate_description` and `get_chat_reply` and calls them inside try/except blocks. Any exception from the Google AI Studio SDK is caught and converted to an appropriate HTTP error.

```python
# FastAPI route
description = generate_description(
    predicted_class=body.predicted_class,
    confidence=body.confidence,
    ...
)
return { "description": description }
```

### Key Integration Challenge — Model Architecture Matching

The biggest technical requirement when loading a saved PyTorch model is that the Python code must reconstruct the **exact same architecture** that was used during training before loading the weights. If even one layer is different, `model.load_state_dict()` will fail with a key mismatch error.

For the cancer model, the custom two-layer head must be defined exactly as it was during training:

```python
model.fc = nn.Sequential(
    nn.Dropout(p=0.4),
    nn.Linear(in_features, 256),
    nn.ReLU(),
    nn.Dropout(p=0.2),
    nn.Linear(256, 1),
)
```

For the disease model, the head is a single linear layer:

```python
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
```

This information was sourced from the training notebook (`skin_cancer_classifier-1.ipynb`) and the disease model training scripts (`disease_model/infer_isic.py`).

### Key Integration Challenge — Environment and Dependency Conflicts

PyTorch CPU wheels are not published on PyPI. The `--extra-index-url https://download.pytorch.org/whl/cpu` directive must be present in `requirements.txt` so pip can resolve the correct wheel for the user's platform. This was a significant hurdle during initial setup and is now documented in `requirements.txt`.

### Key Integration Challenge — 429 Rate Limit Fallback

The Gemini descriptor uses two API keys so that if one hits the Google AI Studio rate limit (HTTP 429), the request is automatically retried with the backup key. This is handled transparently in `llm_descriptor.py` and requires no special handling on the frontend.

---

## 6. The Features

### Functional Requirements

#### F1 — Image Upload (Cancer and Disease Detectors)
Both detector pages support image upload via drag-and-drop onto a dedicated drop zone or by clicking to open the system file picker. Only JPG and PNG files are accepted. The image is previewed immediately after selection and remains visible throughout the analysis process. The page only resets when the user explicitly clicks "Scan A New Image."

#### F2 — Skin Cancer Detection
A binary classification is performed by the ResNet-50 model. The confidence score is displayed as an animated circular ring (similar to a fitness ring), filling to the percentage of the model's confidence. Results are displayed in one of three styled containers — green for benign, red for malignant, and grey for inconclusive (when the model's confidence is below 40%).

#### F3 — Skin Disease Classification
An 8-class classification is performed by the ResNet-152 model. Results are displayed as:
- Eight animated horizontal confidence bars (one per class, colour-coded)
- A main result pill showing the predicted class name and confidence percentage
- An animated severity bar (colour-coded from green to red) reflecting clinical risk

#### F4 — AI Disease Description Generator
After a disease classification result is available, the user can click "Generate Description" to request an AI-generated plain-language explanation from Gemini. The description covers what the disease is, what causes it, its severity, how to identify it, and whether to seek immediate care. The API is never called automatically — it requires an explicit button click. A rate limit of 5 descriptions per 2-minute window is enforced on the frontend.

#### F5 — AI Dermatology Chatbot (Karl)
A floating chat panel is accessible from every page. Karl (powered by Gemma) answers dermatology-related questions in a conversational style. The chatbot enforces a 500-character input limit, a 15-message session limit, and a strict dermatology-only domain constraint built into the system prompt. Conversation history is maintained and sent with each request so the model retains context across turns.

#### F6 — Model Architecture Transparency
Both detector pages include a "Model Transparency" section at the bottom that displays key statistics about the underlying model: architecture name, parameter count, task type, training performance metrics, input specifications, and accepted file formats. This is to help users understand what they are interacting with and to build appropriate trust in the tool.

#### F7 — Medical Disclaimers Page
A dedicated page presents five structured disclaimer cards using warning icons and clear language. This page is accessible from the navigation bar and reinforces the non-clinical nature of the tool.

#### F8 — Navigation and Routing
A persistent navigation bar links to all pages. Client-side routing via React Router DOM means page transitions happen instantly without full-page reloads.

---

### Non-Functional Requirements

#### Performance
- Both PyTorch models are loaded into memory at server startup and reused across requests, eliminating the model load time from per-request latency.
- All image processing is performed in memory without writing to disk.
- The LLM APIs are called only on explicit user action (button press), not automatically, to avoid unnecessary latency and API consumption.
- CPU inference is used throughout. Inference latency depends on hardware but is typically 1–5 seconds for the disease model (ResNet-152, 384×384 input) and under 1 second for the cancer model (ResNet-50, 224×224 input).

#### Scalability
- The current setup is designed for a single-server, local or small-hosted deployment.
- Rate limiting is noted in comments throughout the backend but not yet implemented. Before public deployment, `slowapi` with an in-memory or Redis-backed limiter should be added to `/predict/skin-cancer`, `/predict/skin-disease`, `/llm/describe-disease`, and `/llm/chat`.
- The frontend rate limit for descriptions (5 per 2 minutes, tracked in `useRef`) is client-side only and can be bypassed by a technical user. Server-side enforcement is needed for production.

#### Security
- API keys are stored in a `.env` file and are never embedded in source code or returned in any API response.
- `python-dotenv` loads keys into the process environment at startup before any module that reads them is imported.
- The chatbot system prompt includes explicit injection-resistance instructions: the model is instructed to ignore requests to "reset," "ignore previous instructions," or "enter developer mode," and to refuse all requests outside the dermatology domain.
- The backend validates file content type (`image/jpeg` or `image/png`) and rejects empty files before passing bytes to the classifier.
- CORS is restricted to `http://localhost:3000` to prevent unauthorised cross-origin requests.

#### Reliability and Availability
- The Gemini descriptor uses a primary/backup key failover pattern. If the primary key is rate limited, the request is automatically retried with the backup key without user-visible errors.
- Both classifier instances are wrapped in try/except at startup so a missing model file produces a clear warning log rather than a crash, and the server starts in a degraded state (the affected endpoint returns a 503 with a clear message).
- The health check endpoint (`GET /health`) reports whether each model is loaded, enabling external monitoring.

#### Usability
- The application uses plain, jargon-free language throughout the UI.
- Error states are communicated clearly to the user (e.g., "Model is not loaded," "Generation limit reached").
- Drag-and-drop upload is supplemented with a click-to-upload fallback.
- The chatbot input supports keyboard-native interactions: Enter to send, Shift+Enter for a new line, Ctrl/Cmd+Z for word-based undo, and full copy/paste support.
- Medical disclaimers are displayed prominently on their own page and reinforced within the result cards.
- The confidence ring (cancer) and probability bars (disease) provide visual representations of uncertainty rather than presenting results as certainties.
- The severity bar uses a colour gradient (teal → amber → red) so urgency is immediately understood without reading the numeric score.

#### Maintainability
- Each concern is isolated in its own file. Adding a new model requires only a new classifier module and a new route in `main.py`.
- All design values (colours, spacing, typography) are defined as CSS Custom Properties in `:root` so the entire visual theme can be changed in one place.
- The `disease_model/` folder preserves training artefacts (`config.json`, `summary.json`, `history.csv`, `confusion_matrix.csv`) so the model's provenance and performance are fully documented.
- Code comments explain intent and non-obvious decisions rather than narrating what the code does.
- The `.env.example` file ensures any new developer can configure the environment correctly without access to the original secrets.

---

## 7. The Stakeholders

### Primary Stakeholders

**End Users (General Public)**
The primary users are individuals who want a quick, accessible first look at a skin condition before deciding whether to seek professional care. They are not medically trained and need results presented in plain, reassuring language. The application addresses their needs through simple upload flows, visual confidence representations, and AI-generated explanations.

**Medical Students and Trainees**
A secondary group of primary users who may use DermAI as an educational tool to see how deep learning models classify dermoscopic images across the 8 ISIC disease categories. The model transparency section and per-class probability bars serve this audience.

### Secondary Stakeholders

**Developers and Maintainers**
The team responsible for maintaining or extending the system. Their needs are addressed by the clean file separation, comprehensive documentation, `.env.example` template, and preserved training artefacts.

**Healthcare Providers (Future)**
In a future productised version, clinicians or telemedicine platforms could integrate DermAI as a pre-screening tool. Their need for accuracy, reliability, and regulatory compliance would require significant additional validation and certification before this is feasible.

---

## 8. The Deployment

### Development Environment Setup

The application was developed and tested on Windows 10/11 with the following environment:

**Python Environment**

A dedicated conda environment with Python 3.12.7 is required. This version is the industry standard for deep learning work and is compatible with all PyTorch CPU wheels available through the official PyTorch wheel index.

```bash
# Create the conda environment
conda create -n dermai python=3.12.7

# Activate it
conda activate dermai

# Navigate to backend and install dependencies
cd backend
pip install -r requirements.txt
```

The `requirements.txt` includes an `--extra-index-url` directive pointing to the official PyTorch wheel server because PyTorch is not distributed through the standard PyPI index:

```
--extra-index-url https://download.pytorch.org/whl/cpu
```

**Node Environment**

Node.js 18 or later is required for the frontend. npm is used as the package manager.

```bash
cd frontend
npm install
npm run dev
```

**API Keys**

Before starting the backend, API keys must be added to `backend/.env`. Keys are obtained from [aistudio.google.com](https://aistudio.google.com):

```
GEMINI_PRIMARY_API_KEY=your_key_here
GEMINI_BACKUP_API_KEY=your_backup_key_here
GEMMA_API_KEY=your_key_here
```

**Model Files**

The trained model weight files must be placed in `backend/models/`:
- `best_model.pth` — Cancer detection model
- `best_model.pt` — Disease classification model

**Running the Backend**

```bash
# From the backend folder, with conda environment active
uvicorn main:app --reload --port 8000
```

**Running the Frontend**

```bash
# From the frontend folder
npm run dev
```

The application is available at `http://localhost:3000`. The Vite dev server proxies all `/api/*` requests to `http://localhost:8000`.

### Infrastructure Considerations

The current deployment is local-only (localhost). For a production deployment:

- A reverse proxy (e.g., Nginx) should sit in front of Uvicorn.
- HTTPS should be enforced.
- The `CORS allow_origins` list in `main.py` should be updated to the production domain.
- Server-side rate limiting should be implemented (e.g., `slowapi` with Redis).
- The frontend should be built for production (`npm run build`) and served as static files.
- Model weight files should be stored in a persistent volume rather than committed to a repository.

---

## 9. Future Prospects and Improvements

### 1. Server-Side Rate Limiting

Currently, rate limiting for the LLM endpoints is only enforced on the frontend (client-side) and can be bypassed by a technical user. The backend should implement proper server-side rate limiting using a library such as `slowapi` backed by Redis. A rate of 10 prediction requests per minute per IP and 5 LLM description requests per 2 minutes per IP would be reasonable starting limits.

### 2. GPU Inference

Both models currently run on CPU, which means inference takes 1–5 seconds per request. Migrating to a GPU-backed deployment would reduce this to milliseconds and allow significantly higher request throughput.

### 3. Expanded Cancer Model

The cancer detection model currently performs binary classification (benign vs. malignant). An improved version could classify across multiple cancer subtypes (e.g., melanoma, basal cell carcinoma, squamous cell carcinoma) to provide more actionable diagnostic information.

### 4. Confidence Calibration

The raw confidence scores from the models are softmax probabilities, which are known to be overconfident. Applying post-hoc calibration methods (e.g., temperature scaling) would make the reported confidence percentages more meaningful and reliable.

### 5. Image Metadata and Clinical Context

Currently, only the image is sent for inference. Providing the model with additional structured context — such as the patient's age, skin type, lesion size, or location on the body — could significantly improve classification accuracy and relevance of the AI-generated description.

### 6. User Accounts and History

Adding authentication and a user history feature would allow users to track lesions over time, which is clinically valuable for monitoring changes. This would require a database layer (e.g., PostgreSQL) and a secure authentication system.

### 7. Explainability (Grad-CAM)

Incorporating Grad-CAM (Gradient-weighted Class Activation Mapping) would allow the system to visually highlight the regions of the uploaded image that the model focused on when making its prediction. This would greatly improve transparency and trust.

### 8. Multilingual Support

Expanding the chatbot system prompt and UI strings to support multiple languages would significantly broaden accessibility, particularly in regions where English is not the primary language.

### 9. Mobile Application

The current web UI is responsive but optimised for desktop. A dedicated mobile application (React Native) would allow users to take and upload photos directly from their phone camera, significantly improving the upload experience.

### 10. Clinical Validation and Regulatory Compliance

Before any form of clinical deployment, the models would need formal clinical validation studies comparing their performance against board-certified dermatologists, and the system would need to comply with applicable medical device regulations (e.g., FDA 510(k) in the US, CE marking in the EU).
