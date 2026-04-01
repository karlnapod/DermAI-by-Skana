import io

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFile
from torchvision import models, transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True


# Model configuration — sourced directly from disease_model/config.json.
MODEL_NAME  = "resnet152"
IMAGE_SIZE  = 384
NUM_CLASSES = 8

# Class order must match the class_order list from the training config exactly.
# Index 0 = MEL, Index 1 = NV, ..., Index 7 = SCC.
CLASS_ORDER = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]

# Full descriptive names for each abbreviated class label.
CLASS_DESCRIPTIONS = {
    "MEL":  "Melanoma",
    "NV":   "Melanocytic Nevus",
    "BCC":  "Basal Cell Carcinoma",
    "AK":   "Actinic Keratosis",
    "BKL":  "Benign Keratosis",
    "DF":   "Dermatofibroma",
    "VASC": "Vascular Lesion",
    "SCC":  "Squamous Cell Carcinoma",
}

# Severity scores (0–100) represent medical urgency and dangerousness for
# each class. Scores are derived from clinical literature on skin lesion risk.
SEVERITY_SCORES = {
    "MEL":  95,   # Melanoma — life-threatening malignancy, urgent referral needed
    "SCC":  78,   # Squamous cell carcinoma — can metastasise, treat promptly
    "BCC":  65,   # Basal cell carcinoma — locally destructive, rarely metastasises
    "AK":   52,   # Actinic keratosis — precancerous lesion, monitor and treat
    "VASC": 38,   # Vascular lesion — usually benign, clinical assessment needed
    "BKL":  18,   # Benign keratosis — benign, routine monitoring sufficient
    "NV":   15,   # Melanocytic nevus — benign mole, periodic review recommended
    "DF":   10,   # Dermatofibroma — very benign, rarely requires intervention
}

# Inference transforms mirror the val/test transforms used during training.
# Resize to 384×384 then apply ImageNet mean and standard deviation normalisation.
INFERENCE_TRANSFORMS = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# The following function is responsible for constructing the ResNet-152 model
# with the exact classifier head used during training: a single linear layer
# that maps the 2048-dimensional feature vector to 8 class logits.
def build_disease_model() -> nn.Module:

    model = models.resnet152(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

    return model


class DiseaseClassifier:
    """
    Wraps the trained ResNet-152 disease classifier and exposes a single
    predict() method. The model is loaded once at construction time and
    reused for every subsequent inference request.
    """

    # The following method is responsible for loading the saved state_dict
    # into the reconstructed ResNet-152 architecture and setting it to eval mode.
    def __init__(self, model_path: str) -> None:

        self.device = torch.device("cpu")
        self.model  = build_disease_model()

        state_dict = torch.load(
            model_path,
            map_location=self.device,
            weights_only=True,
        )

        self.model.load_state_dict(state_dict)
        self.model.eval()

        print(f"[DiseaseClassifier] Loaded from {model_path}")


    # The following method is responsible for running a full inference pass
    # on raw image bytes and returning a structured prediction dictionary
    # containing the top predicted class, all 8 class probabilities expressed
    # as percentages, and the clinical severity score for the top class.
    def predict(self, image_bytes: bytes) -> dict:

        img    = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = INFERENCE_TRANSFORMS(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)

        probs = F.softmax(logits, dim=1)[0]

        predicted_index = int(probs.argmax().item())
        predicted_class = CLASS_ORDER[predicted_index]
        confidence      = float(probs[predicted_index].item())

        probabilities = {
            CLASS_ORDER[i]: round(float(probs[i].item()) * 100, 1)
            for i in range(NUM_CLASSES)
        }

        print(
            f"[Inference] predicted={predicted_class} "
            f"({CLASS_DESCRIPTIONS[predicted_class]}) | "
            f"confidence={confidence * 100:.1f}% | "
            f"severity={SEVERITY_SCORES[predicted_class]}"
        )

        return {
            "predicted_class":       predicted_class,
            "predicted_description": CLASS_DESCRIPTIONS[predicted_class],
            "confidence":            round(confidence * 100, 1),
            "severity":              SEVERITY_SCORES[predicted_class],
            "probabilities":         probabilities,
        }
