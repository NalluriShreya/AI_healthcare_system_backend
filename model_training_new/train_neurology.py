"""
Neurology Department — Model Training Script
Trains TWO models and saves them as separate .pkl files:
  1. alzheimer  → binary (Alzheimer's Disease yes/no)
  2. stroke     → binary (Stroke risk yes/no)

Datasets required (download from Kaggle):
  • Alzheimer's : kaggle.com/datasets/rabieelkharoua/alzheimers-disease-dataset
                  file: alzheimers_disease_data.csv
  • Stroke      : kaggle.com/datasets/fedesoriano/stroke-prediction-dataset
                  file: healthcare-dataset-stroke-data.csv

Place both CSV files in the same folder as this script, then run:
    python train_neurology.py

Output (saved to ./models/):
    alzheimer_model.pkl
    stroke_model.pkl
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

warnings.filterwarnings("ignore")

MODELS_DIR = "./models"
os.makedirs(MODELS_DIR, exist_ok=True)

# ─── Helpers ──────────────────────────────────────────────────────────────────

def evaluate_models(X, y, models: dict) -> tuple[str, object, dict]:
    """Cross-validate several models; return best name, estimator, and metrics."""
    skf     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=skf, scoring="accuracy", n_jobs=-1)
        results[name] = scores.mean()
        print(f"  {name:<25} CV acc = {scores.mean():.4f} ± {scores.std():.4f}")
    best_name = max(results, key=results.get)
    print(f"\n  ✓ Best model: {best_name} ({results[best_name]:.4f})")
    return best_name, models[best_name], results


def save_model(pipeline, feature_names, feature_info, metrics, condition,
               class_names, path, is_multiclass=False):
    payload = {
        "pipeline":      pipeline,
        "feature_names": feature_names,
        "feature_info":  feature_info,
        "class_names":   class_names,
        "condition":     condition,
        "metrics":       metrics,
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    print(f"  Saved → {path}\n")


# ══════════════════════════════════════════════════════════════════════════════
# 1. ALZHEIMER'S DISEASE
# ══════════════════════════════════════════════════════════════════════════════
def train_alzheimer():
    print("=" * 60)
    print("ALZHEIMER'S DISEASE MODEL")
    print("=" * 60)

    df = pd.read_csv("alzheimers_disease_data.csv")
    print(f"Loaded {len(df)} rows, {df.shape[1]} columns")

    # Drop identifier / confidential columns
    drop_cols = ["PatientID", "DoctorInCharge"]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    target_col = "Diagnosis"
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)

    print(f"Class distribution:\n{y.value_counts().to_string()}\n")

    # ── Feature info for the frontend form ───────────────────────────────────
    feature_info = {
        "Age":                      {"type": "number",   "label": "Age (years)",              "min": 60,  "max": 90},
        "Gender":                   {"type": "select",   "label": "Gender",                   "options": [{"value": 0, "label": "Male"}, {"value": 1, "label": "Female"}]},
        "Ethnicity":                {"type": "select",   "label": "Ethnicity",                "options": [{"value": 0, "label": "Caucasian"}, {"value": 1, "label": "African American"}, {"value": 2, "label": "Asian"}, {"value": 3, "label": "Other"}]},
        "EducationLevel":           {"type": "select",   "label": "Education Level",          "options": [{"value": 0, "label": "None"}, {"value": 1, "label": "High School"}, {"value": 2, "label": "Bachelor's"}, {"value": 3, "label": "Higher"}]},
        "BMI":                      {"type": "number",   "label": "BMI",                      "min": 15,  "max": 40},
        "Smoking":                  {"type": "checkbox", "label": "Smoking"},
        "AlcoholConsumption":       {"type": "number",   "label": "Alcohol (units/week)",     "min": 0,   "max": 20},
        "PhysicalActivity":         {"type": "number",   "label": "Physical Activity (hrs/week)", "min": 0, "max": 10},
        "DietQuality":              {"type": "number",   "label": "Diet Quality Score (0-10)","min": 0,   "max": 10},
        "SleepQuality":             {"type": "number",   "label": "Sleep Quality Score (4-10)","min": 4,  "max": 10},
        "FamilyHistoryAlzheimers":  {"type": "checkbox", "label": "Family History of Alzheimer's"},
        "CardiovascularDisease":    {"type": "checkbox", "label": "Cardiovascular Disease"},
        "Diabetes":                 {"type": "checkbox", "label": "Diabetes"},
        "Depression":               {"type": "checkbox", "label": "Depression"},
        "HeadInjury":               {"type": "checkbox", "label": "History of Head Injury"},
        "Hypertension":             {"type": "checkbox", "label": "Hypertension"},
        "SystolicBP":               {"type": "number",   "label": "Systolic BP (mmHg)",       "min": 90,  "max": 180},
        "DiastolicBP":              {"type": "number",   "label": "Diastolic BP (mmHg)",      "min": 60,  "max": 120},
        "CholesterolTotal":         {"type": "number",   "label": "Total Cholesterol (mg/dL)","min": 150, "max": 300},
        "CholesterolLDL":           {"type": "number",   "label": "LDL Cholesterol (mg/dL)", "min": 50,  "max": 200},
        "CholesterolHDL":           {"type": "number",   "label": "HDL Cholesterol (mg/dL)", "min": 20,  "max": 100},
        "CholesterolTriglycerides": {"type": "number",   "label": "Triglycerides (mg/dL)",   "min": 50,  "max": 400},
        "MMSE":                     {"type": "number",   "label": "MMSE Score (0-30)",        "min": 0,   "max": 30},
        "FunctionalAssessment":     {"type": "number",   "label": "Functional Assessment (0-10)", "min": 0, "max": 10},
        "MemoryComplaints":         {"type": "checkbox", "label": "Memory Complaints"},
        "BehavioralProblems":       {"type": "checkbox", "label": "Behavioral Problems"},
        "ADL":                      {"type": "number",   "label": "ADL Score (0-10)",         "min": 0,   "max": 10},
        "Confusion":                {"type": "checkbox", "label": "Confusion"},
        "Disorientation":           {"type": "checkbox", "label": "Disorientation"},
        "PersonalityChanges":       {"type": "checkbox", "label": "Personality Changes"},
        "DifficultyCompletingTasks":{"type": "checkbox", "label": "Difficulty Completing Tasks"},
        "Forgetfulness":            {"type": "checkbox", "label": "Forgetfulness"},
    }

    # Keep only columns that exist in the CSV
    feature_names = [f for f in feature_info.keys() if f in X.columns]
    X = X[feature_names]

    # ── Models ────────────────────────────────────────────────────────────────
    base_models = {
        "GradientBoosting":   Pipeline([("scaler", StandardScaler()), ("clf", GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42))]),
        "RandomForest":       Pipeline([("scaler", StandardScaler()), ("clf", RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1))]),
        "LogisticRegression": Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000, C=1.0, random_state=42))]),
    }

    print("Cross-validation results:")
    best_name, best_model, cv_results = evaluate_models(X, y, base_models)

    # Final fit on full data
    best_model.fit(X, y)
    y_pred = best_model.predict(X)
    y_prob = best_model.predict_proba(X)[:, 1]

    acc = round(accuracy_score(y, y_pred) * 100, 2)
    auc = round(roc_auc_score(y, y_prob) * 100, 2)
    print(f"\nTrain accuracy : {acc}%")
    print(f"Train AUC      : {auc}%")
    print(classification_report(y, y_pred, target_names=["No Alzheimer's", "Alzheimer's"]))

    metrics = {
        "accuracy":     acc,
        "roc_auc":      auc,
        "best_model":   best_name,
        "is_multiclass":False,
    }

    save_model(
        pipeline=best_model,
        feature_names=feature_names,
        feature_info={k: v for k, v in feature_info.items() if k in feature_names},
        metrics=metrics,
        condition="Alzheimer's Disease",
        class_names=["No Alzheimer's", "Alzheimer's"],
        path=os.path.join(MODELS_DIR, "alzheimer_model.pkl"),
    )


# ══════════════════════════════════════════════════════════════════════════════
# 2. STROKE PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
def train_stroke():
    print("=" * 60)
    print("STROKE PREDICTION MODEL")
    print("=" * 60)

    df = pd.read_csv("healthcare-dataset-stroke-data.csv")
    print(f"Loaded {len(df)} rows, {df.shape[1]} columns")

    # Drop ID
    df.drop(columns=["id"], inplace=True, errors="ignore")

    # Encode categoricals
    le_gender = LabelEncoder()
    le_married = LabelEncoder()
    le_work = LabelEncoder()
    le_residence = LabelEncoder()
    le_smoking = LabelEncoder()

    df["gender"]           = le_gender.fit_transform(df["gender"].astype(str))
    df["ever_married"]     = le_married.fit_transform(df["ever_married"].astype(str))
    df["work_type"]        = le_work.fit_transform(df["work_type"].astype(str))
    df["Residence_type"]   = le_residence.fit_transform(df["Residence_type"].astype(str))
    df["smoking_status"]   = le_smoking.fit_transform(df["smoking_status"].astype(str))

    # Impute missing BMI
    df["bmi"] = df["bmi"].fillna(df["bmi"].median())

    target_col = "stroke"
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)

    print(f"Class distribution:\n{y.value_counts().to_string()}")
    print(f"Note: Heavy class imbalance — using SMOTE\n")

    feature_info = {
        "gender":         {"type": "select",   "label": "Gender",              "options": [{"value": v, "label": l} for v, l in zip(le_gender.transform(le_gender.classes_).tolist(), le_gender.classes_.tolist())]},
        "age":            {"type": "number",   "label": "Age (years)",         "min": 0,   "max": 82},
        "hypertension":   {"type": "checkbox", "label": "Hypertension"},
        "heart_disease":  {"type": "checkbox", "label": "Heart Disease"},
        "ever_married":   {"type": "select",   "label": "Ever Married",        "options": [{"value": v, "label": l} for v, l in zip(le_married.transform(le_married.classes_).tolist(), le_married.classes_.tolist())]},
        "work_type":      {"type": "select",   "label": "Work Type",           "options": [{"value": v, "label": l} for v, l in zip(le_work.transform(le_work.classes_).tolist(), le_work.classes_.tolist())]},
        "Residence_type": {"type": "select",   "label": "Residence Type",      "options": [{"value": v, "label": l} for v, l in zip(le_residence.transform(le_residence.classes_).tolist(), le_residence.classes_.tolist())]},
        "avg_glucose_level": {"type": "number","label": "Avg Glucose (mg/dL)", "min": 50,  "max": 280},
        "bmi":            {"type": "number",   "label": "BMI",                 "min": 10,  "max": 98},
        "smoking_status": {"type": "select",   "label": "Smoking Status",      "options": [{"value": v, "label": l} for v, l in zip(le_smoking.transform(le_smoking.classes_).tolist(), le_smoking.classes_.tolist())]},
    }

    feature_names = list(X.columns)

    # ── SMOTE + models ────────────────────────────────────────────────────────
    smote = SMOTE(random_state=42)

    base_models = {
        "GradientBoosting":   ImbPipeline([("smote", smote), ("scaler", StandardScaler()), ("clf", GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42))]),
        "RandomForest":       ImbPipeline([("smote", smote), ("scaler", StandardScaler()), ("clf", RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1))]),
        "LogisticRegression": ImbPipeline([("smote", smote), ("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000, C=1.0, random_state=42))]),
    }

    print("Cross-validation results:")
    best_name, best_model, cv_results = evaluate_models(X, y, base_models)

    best_model.fit(X, y)
    y_pred = best_model.predict(X)
    y_prob = best_model.predict_proba(X)[:, 1]

    acc = round(accuracy_score(y, y_pred) * 100, 2)
    auc = round(roc_auc_score(y, y_prob) * 100, 2)
    print(f"\nTrain accuracy : {acc}%")
    print(f"Train AUC      : {auc}%")
    print(classification_report(y, y_pred, target_names=["No Stroke", "Stroke"]))

    metrics = {
        "accuracy":     acc,
        "roc_auc":      auc,
        "best_model":   best_name,
        "is_multiclass":False,
    }

    save_model(
        pipeline=best_model,
        feature_names=feature_names,
        feature_info={k: v for k, v in feature_info.items() if k in feature_names},
        metrics=metrics,
        condition="Stroke",
        class_names=["No Stroke", "Stroke"],
        path=os.path.join(MODELS_DIR, "stroke_model.pkl"),
    )


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    train_alzheimer()
    train_stroke()
    print("✅ All neurology models trained and saved to ./models/")