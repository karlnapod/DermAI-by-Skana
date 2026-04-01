import os

from google import genai
from google.genai import types


# ---------------------------------------------------------------------------
# Model and API configuration
# ---------------------------------------------------------------------------

DESCRIPTOR_MODEL = "gemini-3.1-flash-lite-preview"

# API keys are loaded from .env by main.py before this module is used.
# Keys are read lazily (inside the function) so load_dotenv() in main.py
# runs before these values are resolved.
def _primary_key() -> str:
    return os.environ.get("GEMINI_PRIMARY_API_KEY", "")

def _backup_key() -> str:
    return os.environ.get("GEMINI_BACKUP_API_KEY", "")


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

DESCRIPTOR_SYSTEM_PROMPT = """\
You are an AI dermatologist specialist. You will use the information provided \
from the deep learning model in the form of class label prediction and \
confidence scores to generate a description of what those mean. Your knowledge \
will be completely limited to the following disease categories: Melanoma, \
Melanocytic Nevus, Basal Cell Carcinoma, Actinic Keratosis, Benign Keratosis, \
Dermatofibroma, Vascular Lesion, Squamous Cell Carcinoma. Furthermore, you \
will explain in around 250 words the following:

- What is the disease that got predicted?
- What can cause it?
- How severe is this?
- What are some ways to identify it?
- Should the patient seek immediate medical attention?

Some rules for your description:

- Ensure you use simple English that is easily understood.
- Make your description in a simple well-formatted paragraph.
- Do not create any content outside the boundary of the information given to you.
- Do not generate any content that is unrelated to dermatology.
- Do not generate any content from a different subject.
- Do not generate any content from the internet.
- Do not generate any content that does not discuss the information given to you.
- If the information given to you is inconclusive or confidence scores are lower \
than 40%, you will explain to the user that the model could not properly \
identify the disease. Reasons: Image may not be clear. Image may not show skin \
conditions. Image is outside the boundary of knowledge for the model. The skin \
condition may not fall under the categories the model specialises in.
- Reference the provided confidence score directly in your description to \
communicate the certainty level of the prediction.
- Reference the provided severity score to clearly communicate the level of \
medical urgency.
- Base your response solely on the prediction data provided in this prompt. \
Do not fabricate or extrapolate beyond it.
- Do not recommend specific medications, drug names, or treatment procedures. \
Only recommend consulting a qualified dermatologist or medical professional.
- Keep your description concise and focused, approximately 200-250 words.\
"""

# Full names for abbreviations used when building the prediction context.
_CLASS_NAMES = {
    "MEL":  "Melanoma",
    "NV":   "Melanocytic Nevus",
    "BCC":  "Basal Cell Carcinoma",
    "AK":   "Actinic Keratosis",
    "BKL":  "Benign Keratosis",
    "DF":   "Dermatofibroma",
    "VASC": "Vascular Lesion",
    "SCC":  "Squamous Cell Carcinoma",
}


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

# The following function is responsible for assembling the full prompt that
# is sent to Gemini. It embeds the system instructions and all prediction
# data so the model has complete context without needing to fetch anything
# from external sources.
def _build_prompt(
    predicted_class: str,
    predicted_description: str,
    confidence: float,
    severity: int,
    probabilities: dict,
) -> str:

    sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)

    probs_lines = "\n".join(
        f"  - {abbrev} ({_CLASS_NAMES.get(abbrev, abbrev)}): {pct:.1f}%"
        for abbrev, pct in sorted_probs
    )

    context = (
        f"Prediction data from the deep learning model:\n"
        f"- Predicted condition: {predicted_description} ({predicted_class})\n"
        f"- Model confidence: {confidence:.1f}%\n"
        f"- Severity score: {severity} / 100\n"
        f"- All class probabilities (highest to lowest):\n{probs_lines}\n\n"
        f"Generate your description based strictly on the above prediction data."
    )

    return f"{DESCRIPTOR_SYSTEM_PROMPT}\n\n{context}"


# ---------------------------------------------------------------------------
# API call helpers
# ---------------------------------------------------------------------------

# The following function is responsible for calling the Gemini API with a
# specific API key and returning the generated description text.
def _call_gemini(api_key: str, prompt: str) -> str:

    client = genai.Client(api_key=api_key)

    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=prompt)],
        ),
    ]

    config = types.GenerateContentConfig(
        temperature=0.7,
        max_output_tokens=600,
    )

    response = client.models.generate_content(
        model=DESCRIPTOR_MODEL,
        contents=contents,
        config=config,
    )

    return response.text.strip()


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

# The following function is responsible for generating an AI description for
# a disease classification result. It tries the primary API key first and
# automatically falls back to the backup key if a 429 rate limit error occurs.
def generate_description(
    predicted_class: str,
    predicted_description: str,
    confidence: float,
    severity: int,
    probabilities: dict,
) -> str:

    primary = _primary_key()
    backup  = _backup_key()

    if not primary:
        raise ValueError(
            "GEMINI_PRIMARY_API_KEY is not set. "
            "Add your key to the backend/.env file."
        )

    prompt = _build_prompt(
        predicted_class,
        predicted_description,
        confidence,
        severity,
        probabilities,
    )

    try:
        print("[LLM Descriptor] Calling Gemini with primary key...")
        return _call_gemini(primary, prompt)

    except Exception as exc:
        error_str = str(exc).lower()
        is_rate_limit = any(
            code in error_str
            for code in ["429", "quota", "resource_exhausted", "rate limit", "exhausted"]
        )

        if is_rate_limit and backup:
            print(f"[LLM Descriptor] Primary key rate limited ({exc}). Retrying with backup key...")
            return _call_gemini(backup, prompt)

        raise
