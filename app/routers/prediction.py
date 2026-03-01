"""
Disease Prediction API Router
File: routers/prediction.py

Include in main.py:
    from routers.prediction import router as prediction_router
    app.include_router(prediction_router)
"""

import os
import io
import json
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel
from fastapi import APIRouter, HTTPException, Depends, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Any, Dict

from core.security import get_current_user

router = APIRouter(prefix="/api/predict", tags=["Disease Prediction"])

# ─── Model directories ────────────────────────────────────────────────────────
MODELS_DIR = r"D:\Data_Science\Projects\AI_Healthcare_System\model_training_new\models"

# ─── DermNet DINOv2 config ────────────────────────────────────────────────────
DERM_MODEL_PATH       = r"D:\Data_Science\Projects\AI_Healthcare_System\model_training_new\models\dermnet_dinov2\derm_model.pt"
DERM_CLASS_NAMES_PATH = r"D:\Data_Science\Projects\AI_Healthcare_System\model_training_new\models\dermnet_dinov2\derm_class_names.json"
DERM_IMG_SIZE         = 224
DERM_BACKBONE_NAME    = "Jayanth2002/dinov2-base-finetuned-SkinDisease"

# ─── MURA Orthopedics config ──────────────────────────────────────────────────
ORTHO_MODEL_PATH      = r"D:\Data_Science\Projects\AI_Healthcare_System\model_training_new\models\mura_efficientnetv2\mura_best.pt"
ORTHO_CONFIG_PATH     = r"D:\Data_Science\Projects\AI_Healthcare_System\model_training_new\models\mura_efficientnetv2\config.json"
ORTHO_IMG_SIZE        = 320
ORTHO_BODY_PARTS      = ['XR_ELBOW', 'XR_FINGER', 'XR_FOREARM',
                          'XR_HAND', 'XR_HUMERUS', 'XR_SHOULDER', 'XR_WRIST']

_loaded_models: Dict[str, Any] = {}
_derm_model   = None
_derm_classes = None
_ortho_model  = None
_ortho_config = None


# ─── DermNet model definition (must match training) ───────────────────────────
class DermNetDINOv2(nn.Module):
    def __init__(self, backbone, num_classes, hidden_size):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(0.4),
            nn.Linear(hidden_size, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        outputs   = self.backbone(x)
        cls_token = outputs.last_hidden_state[:, 0, :]
        return self.head(cls_token)


# ─── MURA EfficientNetV2 model definition (must match training) ───────────────
class MURAModel(nn.Module):
    def __init__(self, num_classes=2, num_parts=7):
        super().__init__()
        import timm
        self.backbone = timm.create_model(
            'tf_efficientnetv2_m.in21k_ft_in1k',
            pretrained=False,       # weights loaded from checkpoint
            num_classes=0,
            global_pool='avg',
        )
        feat_dim = self.backbone.num_features  # 1280

        self.main_head = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )
        self.aux_head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_parts),
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.main_head(features)


# ─── Lazy loaders ─────────────────────────────────────────────────────────────
def _get_derm_model():
    global _derm_model, _derm_classes
    if _derm_model is not None:
        return _derm_model, _derm_classes

    if not os.path.exists(DERM_MODEL_PATH):
        raise HTTPException(status_code=404,
            detail="Dermatology model not found. Place best_model.pt in models/dermnet_dinov2/.")
    if not os.path.exists(DERM_CLASS_NAMES_PATH):
        raise HTTPException(status_code=404,
            detail="class_names.json not found in models/dermnet_dinov2/.")

    with open(DERM_CLASS_NAMES_PATH) as f:
        _derm_classes = json.load(f)

    device   = "cuda" if torch.cuda.is_available() else "cpu"
    backbone = AutoModel.from_pretrained(DERM_BACKBONE_NAME)
    model    = DermNetDINOv2(backbone, len(_derm_classes), backbone.config.hidden_size)
    model.load_state_dict(torch.load(DERM_MODEL_PATH, map_location=device))
    model.to(device).eval()
    _derm_model = model
    return _derm_model, _derm_classes


def _get_ortho_model():
    global _ortho_model, _ortho_config
    if _ortho_model is not None:
        return _ortho_model, _ortho_config

    if not os.path.exists(ORTHO_MODEL_PATH):
        raise HTTPException(status_code=404,
            detail="Orthopedics model not found. Place mura_best.pt in models/mura_efficientnetv2/.")

    # Load config (optional — for kappa/accuracy metrics)
    if os.path.exists(ORTHO_CONFIG_PATH):
        with open(ORTHO_CONFIG_PATH) as f:
            _ortho_config = json.load(f)
    else:
        _ortho_config = {"best_kappa": 0.6507, "acc_tta": 0.8265}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = MURAModel(num_classes=2, num_parts=7)
    model.load_state_dict(torch.load(ORTHO_MODEL_PATH, map_location=device))
    model.to(device).eval()

    # Enable Dropout for TTA stochasticity while keeping BatchNorm in eval mode
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()

    _ortho_model = model
    return _ortho_model, _ortho_config


# ─── Image preprocessors ──────────────────────────────────────────────────────
def _preprocess_derm_image(image_bytes: bytes) -> torch.Tensor:
    from torchvision import transforms
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(DERM_IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return val_transform(img).unsqueeze(0)


def _preprocess_ortho_image(image_bytes: bytes) -> torch.Tensor:
    """Preprocess X-ray for EfficientNetV2-M inference (matches val_transform in notebook)."""
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    val_transform = A.Compose([
        A.Resize(ORTHO_IMG_SIZE, ORTHO_IMG_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    img    = np.array(Image.open(io.BytesIO(image_bytes)).convert("RGB"))
    tensor = val_transform(image=img)["image"]
    return tensor.unsqueeze(0)


# ─── Standard PKL model helpers ───────────────────────────────────────────────
def _load_model(department: str) -> dict:
    dept = department.lower()
    if dept not in _loaded_models:
        path = os.path.join(MODELS_DIR, f"{dept}_model.pkl")
        if not os.path.exists(path):
            raise HTTPException(status_code=404,
                detail=f"Model for '{dept}' not found. Run train_models.py first.")
        with open(path, "rb") as f:
            _loaded_models[dept] = pickle.load(f)
    return _loaded_models[dept]


def _safe_metrics(metrics: dict) -> dict:
    return {k: float(v) if hasattr(v, "item") else v for k, v in metrics.items()}


def _risk_level(probability: float) -> str:
    if probability < 30:   return "Low"
    elif probability < 60: return "Moderate"
    else:                  return "High"


# ─── Department metadata ───────────────────────────────────────────────────────
DEPT_META = {
    "general_practice": {
        "label":       "General Practice",
        "condition":   "Common Disease",
        "description": "Predicts one of 41 common diseases from your symptoms.",
    },
    "cardiology": {
        "label":       "Cardiology",
        "condition":   "Heart Disease",
        "description": "Predicts heart disease risk based on symptoms and risk factors.",
    },
    "endocrinology": {
        "label":       "Endocrinology",
        "condition":   "Diabetes",
        "description": "Predicts early-stage diabetes risk from reported symptoms.",
    },
    "psychiatry": {
        "label":       "Psychiatry",
        "condition":   "Mental Health Disorder",
        "description": "Predicts mental health condition from behavioural symptoms.",
    },
    "gastroenterology": {
        "label":       "Gastroenterology",
        "condition":   "GI Disease",
        "description": "Predicts gastrointestinal condition from symptoms and lifestyle factors.",
    },
    "pediatrics": {
        "label":       "Pediatrics",
        "condition":   "Pediatric Disease",
        "description": "Predicts one of 13 common childhood diseases from symptom severity scores.",
    },
    "alzheimer": {
        "label":       "Neurology — Alzheimer's",
        "condition":   "Alzheimer's Disease",
        "description": "Predicts Alzheimer's disease risk from cognitive assessments, lifestyle, and clinical factors.",
    },
    "stroke": {
        "label":       "Neurology — Stroke",
        "condition":   "Stroke",
        "description": "Predicts stroke risk from age, glucose, BMI, hypertension, and lifestyle factors.",
    },
    "cancer": {
        "label":       "Oncology — Cancer Risk",
        "condition":   "Cancer",
        "description": "Predicts cancer risk from genetic background, BMI, smoking, alcohol, and personal history.",
    },
}


# ─── Schemas ──────────────────────────────────────────────────────────────────
class PredictionRequest(BaseModel):
    department: str
    features: Dict[str, float]


# ─── Endpoints ────────────────────────────────────────────────────────────────

@router.get("/departments")
async def list_departments(current_user: dict = Depends(get_current_user)):
    available = []

    for dept, meta in DEPT_META.items():
        path = os.path.join(MODELS_DIR, f"{dept}_model.pkl")
        if not os.path.exists(path):
            continue
        try:
            model_data = _load_model(dept)
            available.append({
                "id":            dept,
                "label":         meta["label"],
                "condition":     meta["condition"],
                "description":   meta["description"],
                "is_multiclass": model_data["metrics"].get("is_multiclass", False),
                "class_names":   model_data.get("class_names", []),
                "feature_names": model_data["feature_names"],
                "feature_info":  model_data["feature_info"],
                "metrics":       _safe_metrics(model_data["metrics"]),
            })
        except Exception:
            pass

    # Dermatology card
    if os.path.exists(DERM_MODEL_PATH) and os.path.exists(DERM_CLASS_NAMES_PATH):
        with open(DERM_CLASS_NAMES_PATH) as f:
            derm_classes = json.load(f)
        available.append({
            "id":            "dermatology",
            "label":         "Dermatology",
            "condition":     "Skin Disease",
            "description":   f"AI-powered skin disease classifier across {len(derm_classes)} conditions. Upload a photo of the affected area.",
            "is_multiclass": True,
            "is_image":      True,
            "class_names":   derm_classes,
            "feature_names": [],
            "feature_info":  {},
            "metrics": {
                "accuracy":   69.2,
                "best_model": "DINOv2-Base",
            },
        })

    # Orthopedics (MURA) card
    if os.path.exists(ORTHO_MODEL_PATH):
        ortho_cfg = {}
        if os.path.exists(ORTHO_CONFIG_PATH):
            with open(ORTHO_CONFIG_PATH) as f:
                ortho_cfg = json.load(f)
        available.append({
            "id":            "orthopedics",
            "label":         "Orthopedics",
            "condition":     "Musculoskeletal Abnormality",
            "description":   "Detects abnormalities in musculoskeletal X-rays across 7 body parts (elbow, finger, forearm, hand, humerus, shoulder, wrist).",
            "is_multiclass": False,
            "is_image":      True,
            "class_names":   ["Normal", "Abnormal"],
            "feature_names": [],
            "feature_info":  {},
            "metrics": {
                "accuracy":   round(ortho_cfg.get("acc_tta",   0.8265) * 100, 1),
                "kappa":      round(ortho_cfg.get("kappa_tta", 0.6507), 4),
                "best_model": "EfficientNetV2-M",
            },
        })

    return {"departments": available}


@router.post("/")
async def predict(
    body: PredictionRequest,
    current_user: dict = Depends(get_current_user),
):
    """Standard (non-image) prediction endpoint."""
    dept = body.department.lower()

    if dept == "dermatology":
        raise HTTPException(status_code=400,
            detail="Dermatology predictions require an image. Use POST /api/predict/dermatology.")
    if dept == "orthopedics":
        raise HTTPException(status_code=400,
            detail="Orthopedics predictions require an X-ray image. Use POST /api/predict/orthopedics.")
    if dept not in DEPT_META:
        raise HTTPException(status_code=400, detail=f"Unknown department: '{dept}'")

    model_data    = _load_model(dept)
    pipeline      = model_data["pipeline"]
    feat_names    = model_data["feature_names"]
    condition     = model_data["condition"]
    is_multiclass = model_data["metrics"].get("is_multiclass", False)
    class_names   = model_data.get("class_names", [])

    try:
        X = pd.DataFrame(
            [[float(body.features.get(f, 0)) for f in feat_names]],
            columns=feat_names,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid input values: {e}")

    raw_prediction = pipeline.predict(X)[0]
    probabilities  = pipeline.predict_proba(X)[0]

    pipeline_classes = None
    if hasattr(pipeline, "classes_"):
        pipeline_classes = pipeline.classes_.tolist()
    elif hasattr(pipeline[-1], "classes_"):
        pipeline_classes = pipeline[-1].classes_.tolist()

    if pipeline_classes and isinstance(pipeline_classes[0], str):
        class_names = pipeline_classes
    elif class_names and isinstance(class_names[0], str):
        pass
    else:
        class_names = [str(c) for c in (pipeline_classes or [])]

    base_response = {
        "department":     dept,
        "condition":      condition,
        "model_used":     model_data["metrics"].get("best_model", "unknown"),
        "model_accuracy": float(model_data["metrics"].get("accuracy", 0)),
        "disclaimer": (
            "This prediction is generated by a machine learning model for "
            "informational purposes only. It is NOT a medical diagnosis. "
            "Please consult a qualified healthcare professional."
        ),
    }

    if is_multiclass:
        if isinstance(raw_prediction, str):
            predicted_class = raw_prediction
            try:
                pred_idx = class_names.index(predicted_class)
            except (ValueError, AttributeError):
                pred_idx = int(np.argmax(probabilities))
        else:
            pred_idx        = int(raw_prediction)
            predicted_class = (
                class_names[pred_idx]
                if class_names and pred_idx < len(class_names)
                else str(pred_idx)
            )

        confidence = round(float(probabilities[pred_idx]) * 100, 1)
        top3_idx   = np.argsort(probabilities)[::-1][:3]
        top3 = [
            {
                "disease":     class_names[i] if class_names and i < len(class_names) else str(i),
                "probability": round(float(probabilities[i]) * 100, 1),
            }
            for i in top3_idx
        ]
        return {
            **base_response,
            "is_multiclass":   True,
            "predicted_class": predicted_class,
            "confidence":      confidence,
            "top_predictions": top3,
            "message":         f"Most likely diagnosis: {predicted_class} ({confidence}% confidence)",
        }
    else:
        probability    = round(float(probabilities[1]) * 100, 1)
        risk           = _risk_level(probability)
        prediction     = int(raw_prediction)
        positive_label = class_names[1] if len(class_names) > 1 else "Positive"
        negative_label = class_names[0] if len(class_names) > 0 else "Negative"
        return {
            **base_response,
            "is_multiclass": False,
            "prediction":    prediction,
            "probability":   probability,
            "risk_level":    risk,
            "result":        positive_label if prediction == 1 else negative_label,
            "message": (
                f"Indicators suggest potential {condition} risk."
                if prediction == 1
                else f"No strong indicators of {condition} detected."
            ),
        }


@router.post("/dermatology")
async def predict_dermatology(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
):
    """Image-based skin disease prediction using DINOv2."""
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg", "image/webp"):
        raise HTTPException(status_code=400,
            detail="Invalid file type. Please upload a JPEG or PNG image.")

    image_bytes = await file.read()
    if len(image_bytes) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Image too large. Maximum size is 10 MB.")

    model, class_names = _get_derm_model()
    device = next(model.parameters()).device
    tensor = _preprocess_derm_image(image_bytes).to(device)

    with torch.no_grad():
        logits = model(tensor)[0]
        probs  = torch.softmax(logits, dim=0).cpu().numpy()

    top5_idx = probs.argsort()[::-1][:5]
    top5 = [
        {"disease": class_names[i], "probability": round(float(probs[i]) * 100, 1)}
        for i in top5_idx
    ]
    pred_idx        = int(top5_idx[0])
    predicted_class = class_names[pred_idx]
    confidence      = round(float(probs[pred_idx]) * 100, 1)

    return {
        "department":      "dermatology",
        "condition":       "Skin Disease",
        "is_multiclass":   True,
        "is_image":        True,
        "predicted_class": predicted_class,
        "confidence":      confidence,
        "top_predictions": top5,
        "message":         f"Most likely condition: {predicted_class} ({confidence}% confidence)",
        "model_used":      "DINOv2-Base",
        "model_accuracy":  69.2,
        "disclaimer": (
            "This prediction is generated by a deep learning model for "
            "informational purposes only. It is NOT a medical diagnosis. "
            "Please consult a qualified dermatologist."
        ),
    }


@router.post("/orthopedics")
async def predict_orthopedics(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
):
    """
    X-ray based musculoskeletal abnormality detection using EfficientNetV2-M (MURA).
    Accepts a JPEG/PNG X-ray image of any MURA-supported body part.
    Returns Normal vs Abnormal prediction with probabilities.
    """
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg", "image/webp"):
        raise HTTPException(status_code=400,
            detail="Invalid file type. Please upload a JPEG or PNG image.")

    image_bytes = await file.read()
    if len(image_bytes) > 15 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Image too large. Maximum size is 15 MB.")

    model, config = _get_ortho_model()
    device = next(model.parameters()).device

    tensor = _preprocess_ortho_image(image_bytes).to(device)

    # TTA: 5 forward passes with stochastic dropout, average probabilities
    TTA_STEPS = 5
    prob_abnormal_list = []

    with torch.no_grad():
        for _ in range(TTA_STEPS):
            logits = model(tensor)
            prob   = F.softmax(logits, dim=1)[0, 1].cpu().item()
            prob_abnormal_list.append(prob)

    prob_abnormal = float(np.mean(prob_abnormal_list))
    prob_normal   = 1.0 - prob_abnormal
    prediction    = 1 if prob_abnormal >= 0.5 else 0

    kappa    = config.get("kappa_tta", config.get("best_kappa", 0.6507))
    accuracy = round(config.get("acc_tta", 0.8265) * 100, 1)

    return {
        "department":    "orthopedics",
        "condition":     "Musculoskeletal Abnormality",
        "is_ortho":      True,          # signals the frontend to use the ortho result card
        "is_image":      True,
        "prediction":    prediction,
        "result":        "ABNORMAL" if prediction == 1 else "NORMAL",
        "prob_abnormal": round(prob_abnormal * 100, 1),
        "prob_normal":   round(prob_normal   * 100, 1),
        "body_part":     "Unknown",     # body part detection not included in single-image inference
        "message": (
            "Potential abnormality detected. Please consult an orthopaedic specialist."
            if prediction == 1
            else "No significant abnormality detected in this X-ray."
        ),
        "model_used":    "EfficientNetV2-M",
        "model_kappa":   round(kappa, 4),
        "model_accuracy": accuracy,
        "disclaimer": (
            "This prediction is generated by a deep learning model trained on the Stanford MURA dataset "
            "for informational purposes only. It is NOT a medical diagnosis. "
            "Please consult a qualified radiologist or orthopaedic specialist."
        ),
    }