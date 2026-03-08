"""
Disease Prediction API Router
File: routers/prediction.py
"""
import os
import io
import gc
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
from typing import Any, Dict, Optional
from pathlib import Path
from huggingface_hub import hf_hub_download

from app.core.security import get_current_user

router = APIRouter(prefix="/api/predict", tags=["Disease Prediction"])

HF_REPO_ID = "NalluriShreya/ai-healthcare-models"

# ─── DermNet DINOv2 config ────────────────────────────────────────────────────
DERM_IMG_SIZE         = 224
DERM_BACKBONE_NAME    = "Jayanth2002/dinov2-base-finetuned-SkinDisease"

# ─── MURA Orthopedics config ──────────────────────────────────────────────────
ORTHO_IMG_SIZE        = 320
ORTHO_BODY_PARTS      = ['XR_ELBOW', 'XR_FINGER', 'XR_FOREARM',
                          'XR_HAND', 'XR_HUMERUS', 'XR_SHOULDER', 'XR_WRIST']

_loaded_models: Dict[str, Any] = {}
_derm_model   = None
_derm_classes = None
_ortho_model  = None
_ortho_config = None

# Cache for /departments — built once on first call, reused forever
_departments_cache: Optional[list] = None


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
            pretrained=False,
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

    DERM_MODEL_PATH = hf_hub_download(repo_id=HF_REPO_ID, filename="derm_model.pt")
    DERM_CLASS_NAMES_PATH = hf_hub_download(repo_id=HF_REPO_ID, filename="derm_class_names.json")

    with open(DERM_CLASS_NAMES_PATH) as f:
        _derm_classes = json.load(f)

    device = "cpu"
    backbone = AutoModel.from_pretrained(DERM_BACKBONE_NAME)
    model = DermNetDINOv2(backbone, len(_derm_classes), backbone.config.hidden_size)
    state_dict = torch.load(str(DERM_MODEL_PATH), map_location=device)
    model.load_state_dict(state_dict)
    model = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
    model.to(device).eval()

    _derm_model = model
    return _derm_model, _derm_classes


def _unload_derm_model():
    """Unload DINOv2 after inference to free ~400MB RAM."""
    global _derm_model, _derm_classes
    _derm_model = None
    _derm_classes = None
    gc.collect()


def _get_ortho_model():
    global _ortho_model, _ortho_config
    if _ortho_model is not None:
        return _ortho_model, _ortho_config

    ortho_model_path = hf_hub_download(repo_id=HF_REPO_ID, filename="mura_best.pt")
    ortho_config_path = hf_hub_download(repo_id=HF_REPO_ID, filename="config.json")

    with open(ortho_config_path) as f:
        _ortho_config = json.load(f)

    device = "cpu"
    model = MURAModel(num_classes=2, num_parts=7)
    state_dict = torch.load(ortho_model_path, map_location=device)
    model.load_state_dict(state_dict)
    model = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
    model.to(device).eval()

    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()

    _ortho_model = model
    return _ortho_model, _ortho_config


def _unload_ortho_model():
    """Unload EfficientNetV2 after inference to free ~300MB RAM."""
    global _ortho_model, _ortho_config
    _ortho_model = None
    _ortho_config = None
    gc.collect()


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
        model_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=f"{dept}_model.pkl"
        )
        with open(model_path, "rb") as f:
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
    """
    Loads all 9 pkl models (small scikit-learn pipelines, ~10MB each = ~100MB total)
    and caches the result. Runs ONCE per server lifetime — every subsequent call
    returns instantly from cache with zero HuggingFace downloads or model loading.

    The heavy PyTorch image models (DINOv2 ~400MB, EfficientNetV2 ~300MB) are
    NOT loaded here — only on actual image prediction requests, and unloaded
    immediately after to stay within Render's 512MB free tier limit.
    """
    global _departments_cache
    if _departments_cache is not None:
        return {"departments": _departments_cache}

    available = []

    # Load all 9 pkl models — safe, they are small scikit-learn pipelines
    for dept, meta in DEPT_META.items():
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

    # Dermatology card — downloads only the small JSON, NOT the DINOv2 model
    try:
        class_path = hf_hub_download(repo_id=HF_REPO_ID, filename="derm_class_names.json")
        with open(class_path) as f:
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
    except Exception:
        pass

    # Orthopedics card — downloads only the small config JSON, NOT the EfficientNetV2 model
    try:
        config_path = hf_hub_download(repo_id=HF_REPO_ID, filename="config.json")
        with open(config_path) as f:
            ortho_cfg = json.load(f)

        available.append({
            "id":            "orthopedics",
            "label":         "Orthopedics",
            "condition":     "Musculoskeletal Abnormality",
            "description":   "Detects abnormalities in musculoskeletal X-rays across 7 body parts.",
            "is_multiclass": False,
            "is_image":      True,
            "class_names":   ["Normal", "Abnormal"],
            "feature_names": [],
            "feature_info":  {},
            "metrics": {
                "accuracy":   round(ortho_cfg.get("acc_tta", 0.8265) * 100, 1),
                "kappa":      round(ortho_cfg.get("kappa_tta", 0.6507), 4),
                "best_model": "EfficientNetV2-M",
            },
        })
    except Exception:
        pass

    # Store in cache — this block never runs again until server restarts
    _departments_cache = available
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

    try:
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

        result = {
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
    finally:
        # Always unload DINOv2 after inference to free ~400MB RAM
        _unload_derm_model()

    return result


@router.post("/orthopedics")
async def predict_orthopedics(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
):
    """X-ray based musculoskeletal abnormality detection using EfficientNetV2-M (MURA)."""
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg", "image/webp"):
        raise HTTPException(status_code=400,
            detail="Invalid file type. Please upload a JPEG or PNG image.")

    image_bytes = await file.read()
    if len(image_bytes) > 15 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Image too large. Maximum size is 15 MB.")

    try:
        model, config = _get_ortho_model()
        device = next(model.parameters()).device
        tensor = _preprocess_ortho_image(image_bytes).to(device)

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

        result = {
            "department":     "orthopedics",
            "condition":      "Musculoskeletal Abnormality",
            "is_ortho":       True,
            "is_image":       True,
            "prediction":     prediction,
            "result":         "ABNORMAL" if prediction == 1 else "NORMAL",
            "prob_abnormal":  round(prob_abnormal * 100, 1),
            "prob_normal":    round(prob_normal   * 100, 1),
            "body_part":      "Unknown",
            "message": (
                "Potential abnormality detected. Please consult an orthopaedic specialist."
                if prediction == 1
                else "No significant abnormality detected in this X-ray."
            ),
            "model_used":     "EfficientNetV2-M",
            "model_kappa":    round(kappa, 4),
            "model_accuracy": accuracy,
            "disclaimer": (
                "This prediction is generated by a deep learning model trained on the Stanford MURA dataset "
                "for informational purposes only. It is NOT a medical diagnosis. "
                "Please consult a qualified radiologist or orthopaedic specialist."
            ),
        }
    finally:
        # Always unload EfficientNetV2 after inference to free ~300MB RAM
        _unload_ortho_model()

    return result