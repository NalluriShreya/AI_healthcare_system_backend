"""
Neurology & Oncology — Model Training Script
============================================
Trains THREE models and saves them as .pkl files:

  1. alzheimer_model.pkl  — Alzheimer's Disease (binary)
  2. stroke_model.pkl     — Stroke Risk (binary)
  3. cancer_model.pkl     — Cancer Risk (binary)

Datasets (place in the same folder as this script):
  • alzheimers_disease_data.csv          kaggle.com/datasets/rabieelkharoua/alzheimers-disease-dataset
  • healthcare-dataset-stroke-data.csv   kaggle.com/datasets/fedesoriano/stroke-prediction-dataset
  • cancer_dataset.csv                   kaggle.com/datasets/rabieelkharoua/cancer-prediction-dataset

Requirements:
    pip install scikit-learn pandas numpy

Usage:
    python train_neurology_oncology.py

Output saved to ./models/

Results (verified):
  Alzheimer  — RandomForest       CV AUC 0.9504  Train Acc 98.28%
  Stroke     — LogisticRegression CV AUC 0.8366  Train Acc 73.82%  (heavy imbalance 4861/249)
  Cancer     — GradientBoosting   CV AUC 0.9511  Train Acc 98.47%
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

warnings.filterwarnings("ignore")
MODELS_DIR = r'D:\Data_Science\Projects\AI_Healthcare_System\model_training_new\models'
os.makedirs(MODELS_DIR, exist_ok=True)


# ─── Shared helpers ───────────────────────────────────────────────────────────

def build_models():
    """Three pipelines with Imputer + Scaler + Classifier.
    SimpleImputer is included so NaN values (e.g. stroke BMI) are handled
    correctly inside cross-validation folds — not leaked from the full dataset.
    """
    return {
        "GradientBoosting": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc",  StandardScaler()),
            ("clf", GradientBoostingClassifier(
                n_estimators=300, learning_rate=0.05,
                max_depth=4, subsample=0.8, random_state=42)),
        ]),
        "RandomForest": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc",  StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=300, max_depth=10,
                class_weight="balanced", random_state=42, n_jobs=-1)),
        ]),
        "LogisticRegression": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc",  StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=1000, C=0.5,
                class_weight="balanced", random_state=42)),
        ]),
    }


def evaluate_models(X, y, models):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=skf, scoring="roc_auc", n_jobs=-1)
        results[name] = scores.mean()
        print(f"  {name:<25} CV AUC = {scores.mean():.4f} +/- {scores.std():.4f}")
    best_name = max(results, key=results.get)
    print(f"\n  Best: {best_name}  (AUC = {results[best_name]:.4f})")
    return best_name, models[best_name]


def final_fit_and_report(model, X, y, class_names):
    model.fit(X, y)
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    acc = round(accuracy_score(y, y_pred) * 100, 2)
    auc = round(roc_auc_score(y, y_prob) * 100, 2)
    print(f"\n  Train Accuracy: {acc}%  |  Train AUC: {auc}%")
    print(classification_report(y, y_pred, target_names=class_names))
    return acc, auc


def save_pkl(pipeline, feature_names, feature_info, metrics, condition, class_names, path):
    with open(path, "wb") as f:
        pickle.dump({
            "pipeline":      pipeline,
            "feature_names": feature_names,
            "feature_info":  feature_info,
            "class_names":   class_names,
            "condition":     condition,
            "metrics":       metrics,
        }, f)
    print(f"  Saved -> {path}\n")


# ══════════════════════════════════════════════════════════════════════════════
# 1. ALZHEIMER'S DISEASE
#    File   : alzheimers_disease_data.csv
#    Rows   : 2149  |  Features: 32  |  No nulls  |  All numeric
#    Target : Diagnosis  (0=No: 1389,  1=Yes: 760)
# ══════════════════════════════════════════════════════════════════════════════
def train_alzheimer():
    print("=" * 60)
    print("1. ALZHEIMER'S DISEASE")
    print("=" * 60)

    df = pd.read_csv(r"datasets\alzheimers_disease_data.csv")
    df.drop(columns=["PatientID", "DoctorInCharge"], inplace=True, errors="ignore")

    y = df["Diagnosis"].astype(int)
    X = df.drop(columns=["Diagnosis"])
    feature_names = list(X.columns)

    print(f"Samples: {len(df)}  |  Features: {X.shape[1]}")
    print(f"Class  : 0={(y==0).sum()}  1={(y==1).sum()}\n")

    feature_info = {
        "Age":                       {"type": "number",   "label": "Age (years)",                      "min": 60,  "max": 90},
        "Gender":                    {"type": "select",   "label": "Gender",                           "options": [{"value": 0, "label": "Male"}, {"value": 1, "label": "Female"}]},
        "Ethnicity":                 {"type": "select",   "label": "Ethnicity",                        "options": [{"value": 0, "label": "Caucasian"}, {"value": 1, "label": "African American"}, {"value": 2, "label": "Asian"}, {"value": 3, "label": "Other"}]},
        "EducationLevel":            {"type": "select",   "label": "Education Level",                  "options": [{"value": 0, "label": "None"}, {"value": 1, "label": "High School"}, {"value": 2, "label": "Bachelor's"}, {"value": 3, "label": "Higher"}]},
        "BMI":                       {"type": "number",   "label": "BMI",                              "min": 15,  "max": 40},
        "Smoking":                   {"type": "checkbox", "label": "Smoking"},
        "AlcoholConsumption":        {"type": "number",   "label": "Alcohol Consumption (units/week)",  "min": 0,  "max": 20},
        "PhysicalActivity":          {"type": "number",   "label": "Physical Activity (hrs/week)",     "min": 0,   "max": 10},
        "DietQuality":               {"type": "number",   "label": "Diet Quality Score (0-10)",        "min": 0,   "max": 10},
        "SleepQuality":              {"type": "number",   "label": "Sleep Quality Score (4-10)",       "min": 4,   "max": 10},
        "FamilyHistoryAlzheimers":   {"type": "checkbox", "label": "Family History of Alzheimer's"},
        "CardiovascularDisease":     {"type": "checkbox", "label": "Cardiovascular Disease"},
        "Diabetes":                  {"type": "checkbox", "label": "Diabetes"},
        "Depression":                {"type": "checkbox", "label": "Depression"},
        "HeadInjury":                {"type": "checkbox", "label": "History of Head Injury"},
        "Hypertension":              {"type": "checkbox", "label": "Hypertension"},
        "SystolicBP":                {"type": "number",   "label": "Systolic BP (mmHg)",               "min": 90,  "max": 180},
        "DiastolicBP":               {"type": "number",   "label": "Diastolic BP (mmHg)",              "min": 60,  "max": 120},
        "CholesterolTotal":          {"type": "number",   "label": "Total Cholesterol (mg/dL)",        "min": 150, "max": 300},
        "CholesterolLDL":            {"type": "number",   "label": "LDL Cholesterol (mg/dL)",          "min": 50,  "max": 200},
        "CholesterolHDL":            {"type": "number",   "label": "HDL Cholesterol (mg/dL)",          "min": 20,  "max": 100},
        "CholesterolTriglycerides":  {"type": "number",   "label": "Triglycerides (mg/dL)",            "min": 50,  "max": 400},
        "MMSE":                      {"type": "number",   "label": "MMSE Score (0-30)",                "min": 0,   "max": 30},
        "FunctionalAssessment":      {"type": "number",   "label": "Functional Assessment (0-10)",     "min": 0,   "max": 10},
        "MemoryComplaints":          {"type": "checkbox", "label": "Memory Complaints"},
        "BehavioralProblems":        {"type": "checkbox", "label": "Behavioral Problems"},
        "ADL":                       {"type": "number",   "label": "ADL Score (0-10)",                 "min": 0,   "max": 10},
        "Confusion":                 {"type": "checkbox", "label": "Confusion"},
        "Disorientation":            {"type": "checkbox", "label": "Disorientation"},
        "PersonalityChanges":        {"type": "checkbox", "label": "Personality Changes"},
        "DifficultyCompletingTasks": {"type": "checkbox", "label": "Difficulty Completing Tasks"},
        "Forgetfulness":             {"type": "checkbox", "label": "Forgetfulness"},
    }

    print("Cross-validation (AUC):")
    best_name, best_model = evaluate_models(X, y, build_models())
    acc, auc = final_fit_and_report(best_model, X, y, ["No Alzheimer's", "Alzheimer's"])

    save_pkl(
        pipeline=best_model,
        feature_names=feature_names,
        feature_info={k: v for k, v in feature_info.items() if k in feature_names},
        metrics={"accuracy": acc, "roc_auc": auc, "best_model": best_name, "is_multiclass": False},
        condition="Alzheimer's Disease",
        class_names=["No Alzheimer's", "Alzheimer's"],
        path=os.path.join(MODELS_DIR, "alzheimer_model.pkl"),
    )


# ══════════════════════════════════════════════════════════════════════════════
# 2. STROKE
#    File   : healthcare-dataset-stroke-data.csv
#    Rows   : 5110  |  Features: 10  |  BMI has 201 NaN (handled in pipeline)
#    Target : stroke  (0=No: 4861,  1=Yes: 249)  — heavy imbalance
#    Encoding: gender, ever_married, work_type, Residence_type, smoking_status
# ══════════════════════════════════════════════════════════════════════════════
def train_stroke():
    print("=" * 60)
    print("2. STROKE PREDICTION")
    print("=" * 60)

    df = pd.read_csv(r"datasets\healthcare-dataset-stroke-data.csv")
    df.drop(columns=["id"], inplace=True, errors="ignore")

    # Convert bmi to numeric — keeps NaN values (imputer handles inside pipeline)
    df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce")

    # Encode categoricals
    df["gender"]         = df["gender"].map({"Male": 1, "Female": 0, "Other": 0})
    df["ever_married"]   = df["ever_married"].map({"Yes": 1, "No": 0})
    df["Residence_type"] = df["Residence_type"].map({"Urban": 1, "Rural": 0})
    df["work_type"]      = df["work_type"].map({
        "Private": 0, "Self-employed": 1, "Govt_job": 2, "children": 3, "Never_worked": 4
    })
    df["smoking_status"] = df["smoking_status"].map({
        "never smoked": 0, "formerly smoked": 1, "smokes": 2, "Unknown": 3
    })

    y = df["stroke"].astype(int)
    X = df.drop(columns=["stroke"])
    feature_names = list(X.columns)

    print(f"Samples: {len(df)}  |  Features: {X.shape[1]}")
    print(f"Class  : 0={(y==0).sum()}  1={(y==1).sum()}  (heavy imbalance -> class_weight='balanced')")
    print(f"BMI NaN: 201 rows -> handled by SimpleImputer(median) inside pipeline\n")

    feature_info = {
        "gender":            {"type": "select",   "label": "Gender",               "options": [{"value": 0, "label": "Female"}, {"value": 1, "label": "Male"}]},
        "age":               {"type": "number",   "label": "Age (years)",          "min": 0,   "max": 82},
        "hypertension":      {"type": "checkbox", "label": "Hypertension"},
        "heart_disease":     {"type": "checkbox", "label": "Heart Disease"},
        "ever_married":      {"type": "select",   "label": "Ever Married",         "options": [{"value": 0, "label": "No"}, {"value": 1, "label": "Yes"}]},
        "work_type":         {"type": "select",   "label": "Work Type",            "options": [{"value": 0, "label": "Private"}, {"value": 1, "label": "Self-employed"}, {"value": 2, "label": "Govt Job"}, {"value": 3, "label": "Children"}, {"value": 4, "label": "Never Worked"}]},
        "Residence_type":    {"type": "select",   "label": "Residence Type",       "options": [{"value": 0, "label": "Rural"}, {"value": 1, "label": "Urban"}]},
        "avg_glucose_level": {"type": "number",   "label": "Avg Glucose (mg/dL)",  "min": 50,  "max": 280},
        "bmi":               {"type": "number",   "label": "BMI",                  "min": 10,  "max": 98},
        "smoking_status":    {"type": "select",   "label": "Smoking Status",       "options": [{"value": 0, "label": "Never Smoked"}, {"value": 1, "label": "Formerly Smoked"}, {"value": 2, "label": "Smokes"}, {"value": 3, "label": "Unknown"}]},
    }

    print("Cross-validation (AUC):")
    best_name, best_model = evaluate_models(X, y, build_models())
    acc, auc = final_fit_and_report(best_model, X, y, ["No Stroke", "Stroke"])

    save_pkl(
        pipeline=best_model,
        feature_names=feature_names,
        feature_info={k: v for k, v in feature_info.items() if k in feature_names},
        metrics={"accuracy": acc, "roc_auc": auc, "best_model": best_name, "is_multiclass": False},
        condition="Stroke",
        class_names=["No Stroke", "Stroke"],
        path=os.path.join(MODELS_DIR, "stroke_model.pkl"),
    )


# ══════════════════════════════════════════════════════════════════════════════
# 3. CANCER RISK
#    File   : cancer_dataset.csv
#    Rows   : 1500  |  Features: 8  |  No nulls  |  All numeric
#    Target : Diagnosis  (0=No: 943,  1=Yes: 557)
#    Features: Age, Gender, BMI, Smoking, GeneticRisk,
#              PhysicalActivity, AlcoholIntake, CancerHistory
# ══════════════════════════════════════════════════════════════════════════════
def train_cancer():
    print("=" * 60)
    print("3. CANCER RISK PREDICTION")
    print("=" * 60)

    df = pd.read_csv(r"datasets\cancer_dataset.csv")

    y = df["Diagnosis"].astype(int)
    X = df.drop(columns=["Diagnosis"])
    feature_names = list(X.columns)

    print(f"Samples: {len(df)}  |  Features: {X.shape[1]}")
    print(f"Class  : 0={(y==0).sum()}  1={(y==1).sum()}\n")

    feature_info = {
        "Age":             {"type": "number",   "label": "Age (years)",                  "min": 20,  "max": 80},
        "Gender":          {"type": "select",   "label": "Gender",                       "options": [{"value": 0, "label": "Female"}, {"value": 1, "label": "Male"}]},
        "BMI":             {"type": "number",   "label": "BMI",                          "min": 15,  "max": 40},
        "Smoking":         {"type": "checkbox", "label": "Smoking"},
        "GeneticRisk":     {"type": "select",   "label": "Genetic Risk",                 "options": [{"value": 0, "label": "Low"}, {"value": 1, "label": "Medium"}, {"value": 2, "label": "High"}]},
        "PhysicalActivity":{"type": "number",   "label": "Physical Activity (hrs/week)", "min": 0,   "max": 10},
        "AlcoholIntake":   {"type": "number",   "label": "Alcohol Intake (units/week)",  "min": 0,   "max": 5},
        "CancerHistory":   {"type": "checkbox", "label": "Personal History of Cancer"},
    }

    print("Cross-validation (AUC):")
    best_name, best_model = evaluate_models(X, y, build_models())
    acc, auc = final_fit_and_report(best_model, X, y, ["No Cancer", "Cancer"])

    save_pkl(
        pipeline=best_model,
        feature_names=feature_names,
        feature_info={k: v for k, v in feature_info.items() if k in feature_names},
        metrics={"accuracy": acc, "roc_auc": auc, "best_model": best_name, "is_multiclass": False},
        condition="Cancer",
        class_names=["No Cancer", "Cancer"],
        path=os.path.join(MODELS_DIR, "cancer_model.pkl"),
    )


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    train_alzheimer()
    train_stroke()
    train_cancer()
    print("=" * 60)
    print("All models saved to ./models/")
    print("  alzheimer_model.pkl")
    print("  stroke_model.pkl")
    print("  cancer_model.pkl")