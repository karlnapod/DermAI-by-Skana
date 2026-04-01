import io
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


# Classification threshold: if sigmoid(logit) >= this value the sample is
# predicted as malignant.
CLASSIFICATION_THRESHOLD = 0.50

# Confidence threshold: if the confidence in the prediction falls below this
# value the result is treated as inconclusive and no diagnosis is reported.
CONFIDENCE_THRESHOLD = 0.40

# Class labels must match the alphabetical order used by ImageFolder during
# training: index 0 = benign, index 1 = malignant.
CLASSES = ["benign", "malignant"]

# Inference transforms mirror the val_transforms used during training.
# No augmentation is applied — only deterministic resize and normalisation.
INFERENCE_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# The following function is responsible for constructing the ResNet-50 model
# with the same custom classifier head that was used during training.
# The architecture must match exactly so that the saved state_dict can be
# loaded without key mismatches.
def build_cancer_model() -> nn.Module:

    model = models.resnet50(weights=None)
    in_features = model.fc.in_features

    model.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(256, 1),
    )

    return model


class CancerClassifier:
    """
    Wraps the trained ResNet-50 model and exposes a single predict() method.
    The model is loaded once at construction time and reused for every request.
    """

    # The following method is responsible for loading the saved state_dict
    # into the reconstructed model architecture and setting it to eval mode.
    def __init__(self, model_path: str) -> None:

        self.device = torch.device("cpu")
        self.model = build_cancer_model()

        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        print(f"[CancerClassifier] Loaded. CONFIDENCE_THRESHOLD={CONFIDENCE_THRESHOLD}, CLASSIFICATION_THRESHOLD={CLASSIFICATION_THRESHOLD}")


    # The following method is responsible for running a full inference pass
    # on raw image bytes and returning a structured prediction dictionary.
    def predict(self, image_bytes: bytes) -> dict:

        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        tensor = INFERENCE_TRANSFORMS(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logit = self.model(tensor)

        prob_malignant = torch.sigmoid(logit).item()

        if prob_malignant >= CLASSIFICATION_THRESHOLD:
            raw_prediction = "malignant"
            confidence     = prob_malignant
        else:
            raw_prediction = "benign"
            confidence     = 1.0 - prob_malignant

        # Suppress the diagnosis when the model is not confident enough.
        if confidence < CONFIDENCE_THRESHOLD:
            prediction = "inconclusive"
        else:
            prediction = raw_prediction

        print(
            f"[Inference] prob_malignant={prob_malignant:.4f} | "
            f"raw={raw_prediction} | confidence={confidence:.4f} | "
            f"threshold={CONFIDENCE_THRESHOLD} | final={prediction}"
        )

        return {
            "prediction":     prediction,
            "confidence":     round(confidence * 100, 1),
            "prob_malignant": round(prob_malignant * 100, 1),
        }
