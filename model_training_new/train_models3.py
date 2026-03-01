"""
Disease Prediction Model Training — Symptom Based
Departments: General Practice, Cardiology, Endocrinology, Psychiatry,
             Gastroenterology, Pediatrics

Run:  python train_models.py
Output: models/  (one .pkl per department)
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score,
)

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
# UPDATE THESE PATHS
# ─────────────────────────────────────────────────────────────
GENERAL_PRACTICE_PATH   = 'datasets\dataset.csv'
CARDIOLOGY_PATH         = 'datasets\heart_disease_risk_dataset_earlymed.csv'
ENDOCRINOLOGY_PATH      = 'datasets\diabetes_data_upload.csv'
PSYCHIATRY_PATH         = 'datasets\Dataset-Mental-Disorders.csv'
# GASTROENTEROLOGY_PATH   = 'datasets\gastrointestinal_disease_dataset.csv'
GASTROENTEROLOGY_PATH   = 'datasets\gastroenterology_symptoms.csv'
PEDIATRICS_PATH         = 'datasets\pediatric_symptom_data.csv'

OUTPUT_DIR = r'D:\Data_Science\Projects\AI_Healthcare_System\model_training_new\models'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# CONSOLE HELPERS
# ─────────────────────────────────────────────────────────────
GREEN  = '\033[92m'
YELLOW = '\033[93m'
CYAN   = '\033[96m'
BOLD   = '\033[1m'
RESET  = '\033[0m'

def section(title):
    print(f'\n{BOLD}{CYAN}{"="*60}{RESET}')
    print(f'{BOLD}{CYAN}  {title}{RESET}')
    print(f'{BOLD}{CYAN}{"="*60}{RESET}')

def ok(msg):   print(f'  {GREEN}checkmark{RESET} {msg}')
def info(msg): print(f'  {YELLOW}->{RESET} {msg}')
def err(msg):  print(f'  \033[91mX\033[0m {msg}')


# ─────────────────────────────────────────────────────────────
# 1. GENERAL PRACTICE
#    dataset.csv — Disease, Symptom_1 ... Symptom_17
#    131 unique symptoms across 17 columns (sparse wide format)
#    41 disease classes (multiclass)
#    We convert to binary symptom matrix: one column per symptom
# ─────────────────────────────────────────────────────────────
def load_general_practice(path):
    df = pd.read_csv(path)
    info(f'Raw shape: {df.shape}')

    symptom_cols = [c for c in df.columns if c.startswith('Symptom_')]

    # Collect all unique symptom names
    all_symptoms = set()
    for col in symptom_cols:
        all_symptoms.update(df[col].dropna().str.strip().unique())
    all_symptoms = sorted(all_symptoms)
    info(f'Unique symptoms: {len(all_symptoms)}')

    # Build binary matrix row by row
    binary_rows = []
    for _, row in df.iterrows():
        present = {str(row[c]).strip() for c in symptom_cols if pd.notna(row[c])}
        binary_rows.append({s: int(s in present) for s in all_symptoms})

    X = pd.DataFrame(binary_rows, columns=all_symptoms)
    y = df['Disease'].str.strip()

    ok(f'After pivot: X={X.shape} | {y.nunique()} diseases')

    feature_info = {
        sym: {'label': sym.replace('_', ' ').title(), 'type': 'checkbox', 'default': 0}
        for sym in all_symptoms
    }

    return X, y, all_symptoms, feature_info, None


# ─────────────────────────────────────────────────────────────
# 2. CARDIOLOGY
#    heart_disease_risk_dataset_earlymed.csv
#    All binary (0/1) symptom + risk factor columns
#    Age is numeric, Gender is 0/1
#    Target: Heart_Risk (0 = no risk, 1 = at risk)
# ─────────────────────────────────────────────────────────────
def load_cardiology(path):
    df = pd.read_csv(path).dropna()
    info(f'Shape after dropna: {df.shape}')

    y = df['Heart_Risk'].astype(int)
    X = df.drop(columns=['Heart_Risk'])

    ok(f'Features: {X.shape[1]} | Class balance: {y.value_counts().to_dict()}')

    feature_names = list(X.columns)
    feature_info = {
        'Chest_Pain':          {'label': 'Chest Pain',                      'type': 'checkbox', 'default': 0},
        'Shortness_of_Breath': {'label': 'Shortness of Breath',             'type': 'checkbox', 'default': 0},
        'Fatigue':             {'label': 'Fatigue / Tiredness',             'type': 'checkbox', 'default': 0},
        'Palpitations':        {'label': 'Heart Palpitations',              'type': 'checkbox', 'default': 0},
        'Dizziness':           {'label': 'Dizziness',                       'type': 'checkbox', 'default': 0},
        'Swelling':            {'label': 'Swelling in Legs / Ankles',       'type': 'checkbox', 'default': 0},
        'Pain_Arms_Jaw_Back':  {'label': 'Pain in Arms, Jaw or Back',       'type': 'checkbox', 'default': 0},
        'Cold_Sweats_Nausea':  {'label': 'Cold Sweats / Nausea',            'type': 'checkbox', 'default': 0},
        'High_BP':             {'label': 'Known High Blood Pressure',       'type': 'checkbox', 'default': 0},
        'High_Cholesterol':    {'label': 'Known High Cholesterol',          'type': 'checkbox', 'default': 0},
        'Diabetes':            {'label': 'Diabetes',                        'type': 'checkbox', 'default': 0},
        'Smoking':             {'label': 'Smoking',                         'type': 'checkbox', 'default': 0},
        'Obesity':             {'label': 'Obesity',                         'type': 'checkbox', 'default': 0},
        'Sedentary_Lifestyle': {'label': 'Sedentary / Inactive Lifestyle',  'type': 'checkbox', 'default': 0},
        'Family_History':      {'label': 'Family History of Heart Disease', 'type': 'checkbox', 'default': 0},
        'Chronic_Stress':      {'label': 'Chronic Stress',                  'type': 'checkbox', 'default': 0},
        'Gender':              {'label': 'Gender', 'type': 'select',
                                'options': [{'value': 0, 'label': 'Female'}, {'value': 1, 'label': 'Male'}]},
        'Age':                 {'label': 'Age', 'type': 'number', 'min': 1, 'max': 100, 'unit': 'years'},
    }

    return X, y, feature_names, feature_info, None


# ─────────────────────────────────────────────────────────────
# 3. ENDOCRINOLOGY
#    diabetes_data_upload.csv
#    Symptom cols: Yes/No strings
#    Gender: Male/Female string
#    Target: class — 'Positive' / 'Negative'
# ─────────────────────────────────────────────────────────────
def load_endocrinology(path):
    df = pd.read_csv(path)
    info(f'Raw shape: {df.shape}')

    yes_no_cols = [c for c in df.columns if c not in ['Age', 'Gender', 'class']]
    for col in yes_no_cols:
        df[col] = df[col].map({'Yes': 1, 'No': 0})

    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
    df['class']  = df['class'].map({'Positive': 1, 'Negative': 0})
    df = df.dropna()

    y = df['class'].astype(int)
    X = df.drop(columns=['class'])

    ok(f'Shape: {X.shape} | Class balance: {y.value_counts().to_dict()}')

    feature_names = list(X.columns)
    feature_info = {
        'Age':                {'label': 'Age',                            'type': 'number',   'min': 1, 'max': 100, 'unit': 'years'},
        'Gender':             {'label': 'Gender',                         'type': 'select',
                               'options': [{'value': 0, 'label': 'Female'}, {'value': 1, 'label': 'Male'}]},
        'Polyuria':           {'label': 'Excessive Urination (Polyuria)', 'type': 'checkbox', 'default': 0},
        'Polydipsia':         {'label': 'Excessive Thirst (Polydipsia)',  'type': 'checkbox', 'default': 0},
        'sudden weight loss': {'label': 'Sudden Weight Loss',             'type': 'checkbox', 'default': 0},
        'weakness':           {'label': 'Weakness',                       'type': 'checkbox', 'default': 0},
        'Polyphagia':         {'label': 'Excessive Hunger (Polyphagia)',  'type': 'checkbox', 'default': 0},
        'Genital thrush':     {'label': 'Genital Thrush / Infection',     'type': 'checkbox', 'default': 0},
        'visual blurring':    {'label': 'Blurred Vision',                 'type': 'checkbox', 'default': 0},
        'Itching':            {'label': 'Itching',                        'type': 'checkbox', 'default': 0},
        'Irritability':       {'label': 'Irritability',                   'type': 'checkbox', 'default': 0},
        'delayed healing':    {'label': 'Delayed Healing of Wounds',      'type': 'checkbox', 'default': 0},
        'partial paresis':    {'label': 'Partial Weakness / Paresis',     'type': 'checkbox', 'default': 0},
        'muscle stiffness':   {'label': 'Muscle Stiffness',               'type': 'checkbox', 'default': 0},
        'Alopecia':           {'label': 'Hair Loss (Alopecia)',           'type': 'checkbox', 'default': 0},
        'Obesity':            {'label': 'Obesity',                        'type': 'checkbox', 'default': 0},
    }

    return X, y, feature_names, feature_info, None


# ─────────────────────────────────────────────────────────────
# 4. PSYCHIATRY
#    Dataset-Mental-Disorders.csv
#    Frequency cols: Seldom/Sometimes/Usually/Most-Often → 1/2/3/4
#    YES/NO cols → 1/0  (has trailing space 'YES ' — stripped)
#    Scale cols: "X From 10" → extract integer X
#    Target: Expert Diagnose — Bipolar Type-1, Bipolar Type-2,
#                               Depression, Normal  (multiclass)
# ─────────────────────────────────────────────────────────────
def load_psychiatry(path):
    df = pd.read_csv(path)
    info(f'Raw shape: {df.shape}')

    df = df.drop(columns=['Patient Number'], errors='ignore')

    freq_map  = {'Seldom': 1, 'Sometimes': 2, 'Usually': 3, 'Most-Often': 4}
    for col in ['Sadness', 'Euphoric', 'Exhausted', 'Sleep dissorder']:
        df[col] = df[col].map(freq_map)

    yes_no_cols = [
        'Mood Swing', 'Suicidal thoughts', 'Anorxia', 'Authority Respect',
        'Try-Explanation', 'Aggressive Response', 'Ignore & Move-On',
        'Nervous Break-down', 'Admit Mistakes', 'Overthinking'
    ]
    for col in yes_no_cols:
        df[col] = df[col].str.strip().map({'YES': 1, 'NO': 0})

    for col in ['Sexual Activity', 'Concentration', 'Optimisim']:
        df[col] = df[col].str.extract(r'(\d+)').astype(float)

    le = LabelEncoder()
    df['Expert Diagnose'] = le.fit_transform(df['Expert Diagnose'])

    df = df.dropna()
    y  = df['Expert Diagnose'].astype(int)
    X  = df.drop(columns=['Expert Diagnose'])

    ok(f'Shape: {X.shape} | Classes: {list(le.classes_)}')
    ok(f'Class balance: {y.value_counts().to_dict()}')

    feature_names = list(X.columns)
    feature_info = {
        'Sadness':             {'label': 'Sadness', 'type': 'select',
                                'options': [{'value':1,'label':'Seldom'},{'value':2,'label':'Sometimes'},{'value':3,'label':'Usually'},{'value':4,'label':'Most Often'}]},
        'Euphoric':            {'label': 'Euphoric / Overly Happy', 'type': 'select',
                                'options': [{'value':1,'label':'Seldom'},{'value':2,'label':'Sometimes'},{'value':3,'label':'Usually'},{'value':4,'label':'Most Often'}]},
        'Exhausted':           {'label': 'Exhaustion / Tiredness', 'type': 'select',
                                'options': [{'value':1,'label':'Seldom'},{'value':2,'label':'Sometimes'},{'value':3,'label':'Usually'},{'value':4,'label':'Most Often'}]},
        'Sleep dissorder':     {'label': 'Sleep Disorder', 'type': 'select',
                                'options': [{'value':1,'label':'Seldom'},{'value':2,'label':'Sometimes'},{'value':3,'label':'Usually'},{'value':4,'label':'Most Often'}]},
        'Mood Swing':          {'label': 'Mood Swings',                     'type': 'checkbox', 'default': 0},
        'Suicidal thoughts':   {'label': 'Suicidal Thoughts',               'type': 'checkbox', 'default': 0},
        'Anorxia':             {'label': 'Anorexia / Loss of Appetite',     'type': 'checkbox', 'default': 0},
        'Authority Respect':   {'label': 'Difficulty Respecting Authority', 'type': 'checkbox', 'default': 0},
        'Try-Explanation':     {'label': 'Tries to Over-Explain',           'type': 'checkbox', 'default': 0},
        'Aggressive Response': {'label': 'Aggressive Responses',            'type': 'checkbox', 'default': 0},
        'Ignore & Move-On':    {'label': 'Tends to Ignore and Move On',     'type': 'checkbox', 'default': 0},
        'Nervous Break-down':  {'label': 'Nervous Breakdowns',              'type': 'checkbox', 'default': 0},
        'Admit Mistakes':      {'label': 'Admits Mistakes Easily',          'type': 'checkbox', 'default': 0},
        'Overthinking':        {'label': 'Overthinking',                    'type': 'checkbox', 'default': 0},
        'Sexual Activity':     {'label': 'Sexual Activity Level',           'type': 'number', 'min': 1, 'max': 10, 'unit': '/10'},
        'Concentration':       {'label': 'Concentration Level',             'type': 'number', 'min': 1, 'max': 10, 'unit': '/10'},
        'Optimisim':           {'label': 'Optimism Level',                  'type': 'number', 'min': 1, 'max': 10, 'unit': '/10'},
    }

    return X, y, feature_names, feature_info, le


# ─────────────────────────────────────────────────────────────
# 5. GASTROENTEROLOGY
#    gastrointestinal_disease_dataset.csv
#    Mix of numeric, binary, and categorical columns.
#    Categorical cols encoded via LabelEncoder.
#    Target: Disease_Class (6 classes — multiclass)
#
#    Columns
#    -------
#    Numeric  : Age, BMI, Body_Weight, Microbiome_Index, CRP_ESR,
#               Genetic_Markers, Fecal_Calprotectin, Stress_Level,
#               Physical_Activity, Bowel_Movement_Frequency
#    Binary   : Family_History, Autoimmune_Disorders, H_Pylori_Status,
#               Occult_Blood_Test, Endoscopy_Result, Colonoscopy_Result,
#               Stool_Culture, Food_Intolerance, Smoking_Status,
#               Alcohol_Use, Abdominal_Pain, Bloating, Diarrhea,
#               Constipation, Rectal_Bleeding, Appetite_Loss, Weight_Loss,
#               NSAID_Use, Antibiotic_Use, PPI_Use, Medications
#    Categorical: Gender, Obesity_Status, Ethnicity, Diet_Type, Bowel_Habits
# ─────────────────────────────────────────────────────────────
def load_gastroenterology(path):
    df = pd.read_csv(path)
    info(f'Raw shape: {df.shape}')

    # Target encoding
    le = LabelEncoder()
    df['disease'] = le.fit_transform(df['disease'])

    df = df.dropna()
    y = df['disease'].astype(int)
    X = df.drop(columns=['disease'])

    ok(f'Shape: {X.shape} | Classes: {list(le.classes_)}')
    ok(f'Class balance: {y.value_counts().to_dict()}')

    feature_names = list(X.columns)

    feature_info = {
        # -- Demographics & Numeric --
        'age': {'label': 'Age', 'type': 'number', 'min': 1, 'max': 120, 'unit': 'years'},
        'gender': {'label': 'Gender', 'type': 'select', 
                   'options': [{'value': 0, 'label': 'Female'}, {'value': 1, 'label': 'Male'}]},
        'bmi': {'label': 'BMI', 'type': 'number', 'min': 10, 'max': 60, 'unit': 'kg/m²'},
        'symptom_duration_days': {'label': 'Symptom Duration', 'type': 'number', 'min': 0, 'max': 5000, 'unit': 'days'},

        # -- Symptoms (Binary) --
        'heartburn':      {'label': 'Heartburn',      'type': 'checkbox', 'default': 0},
        'regurgitation':  {'label': 'Regurgitation',  'type': 'checkbox', 'default': 0},
        'chest_pain':     {'label': 'Chest Pain',     'type': 'checkbox', 'default': 0},
        'hoarseness':     {'label': 'Hoarseness',     'type': 'checkbox', 'default': 0},
        'chronic_cough':  {'label': 'Chronic Cough',  'type': 'checkbox', 'default': 0},
        'dysphagia':      {'label': 'Difficulty Swallowing (Dysphagia)', 'type': 'checkbox', 'default': 0},
        'bloating':       {'label': 'Bloating',       'type': 'checkbox', 'default': 0},
        'nausea':         {'label': 'Nausea',         'type': 'checkbox', 'default': 0},
        'vomiting':       {'label': 'Vomiting',       'type': 'checkbox', 'default': 0},
        'abdominal_pain': {'label': 'Abdominal Pain', 'type': 'checkbox', 'default': 0},
        'diarrhea':       {'label': 'Diarrhea',       'type': 'checkbox', 'default': 0},
        'constipation':   {'label': 'Constipation',   'type': 'checkbox', 'default': 0},
        'rectal_bleeding':{'label': 'Rectal Bleeding','type': 'checkbox', 'default': 0},
        'weight_loss':    {'label': 'Weight Loss',    'type': 'checkbox', 'default': 0},
        'fever':          {'label': 'Fever',          'type': 'checkbox', 'default': 0},
        'fatigue':        {'label': 'Fatigue',        'type': 'checkbox', 'default': 0},

        # -- Lifestyle & History (Binary) --
        'smoking':        {'label': 'Smoking Status', 'type': 'checkbox', 'default': 0},
        'alcohol_use':    {'label': 'Alcohol Use',    'type': 'checkbox', 'default': 0},
        'family_history': {'label': 'Family History', 'type': 'checkbox', 'default': 0},
    }

    return X, y, feature_names, feature_info, le

# ─────────────────────────────────────────────────────────────
# 6. PEDIATRICS
#    pediatric_symptom_data.csv
#    Symptom severity scores: 0 = absent, 1 = mild,
#                             2 = moderate, 3 = severe
#    23 symptom columns, all numeric (int64)
#    Target: CONDITION (13 pediatric disease classes — multiclass)
# ─────────────────────────────────────────────────────────────
def load_pediatrics(path):
    df = pd.read_csv(path)
    info(f'Raw shape: {df.shape}')

    le = LabelEncoder()
    df['CONDITION'] = le.fit_transform(df['CONDITION'])

    df = df.dropna()
    y = df['CONDITION'].astype(int)
    X = df.drop(columns=['CONDITION'])

    ok(f'Shape: {X.shape} | Classes: {list(le.classes_)}')
    ok(f'Class balance: {y.value_counts().to_dict()}')

    feature_names = list(X.columns)

    # Severity scale shared by all symptoms
    severity_options = [
        {'value': 0, 'label': 'Absent'},
        {'value': 1, 'label': 'Mild'},
        {'value': 2, 'label': 'Moderate'},
        {'value': 3, 'label': 'Severe'},
    ]

    feature_info = {
        'ABDOMINAL_PAIN':       {'label': 'Abdominal Pain',           'type': 'select', 'options': severity_options},
        'CHEST_PAIN':           {'label': 'Chest Pain',               'type': 'select', 'options': severity_options},
        'COUGH':                {'label': 'Cough',                    'type': 'select', 'options': severity_options},
        'DEHYDRATION':          {'label': 'Dehydration',              'type': 'select', 'options': severity_options},
        'DIARRHEA':             {'label': 'Diarrhea',                 'type': 'select', 'options': severity_options},
        'FEVER':                {'label': 'Fever',                    'type': 'select', 'options': severity_options},
        'HEADACHE':             {'label': 'Headache',                 'type': 'select', 'options': severity_options},
        'ITCHING':              {'label': 'Itching / Rash Sensation', 'type': 'select', 'options': severity_options},
        'MUSCLE_ACHES':         {'label': 'Muscle Aches',             'type': 'select', 'options': severity_options},
        'NAUSEA':               {'label': 'Nausea',                   'type': 'select', 'options': severity_options},
        'NECK_STIFFNESS':       {'label': 'Neck Stiffness',           'type': 'select', 'options': severity_options},
        'PHOTOPHOBIA':          {'label': 'Photophobia (Light Sensitivity)', 'type': 'select', 'options': severity_options},
        'POLYDIPSIA':           {'label': 'Excessive Thirst (Polydipsia)',   'type': 'select', 'options': severity_options},
        'POLYURIA':             {'label': 'Excessive Urination (Polyuria)',  'type': 'select', 'options': severity_options},
        'RASH':                 {'label': 'Rash',                     'type': 'select', 'options': severity_options},
        'RESPIRATORY_DISTRESS': {'label': 'Respiratory Distress',     'type': 'select', 'options': severity_options},
        'RUNNY_NOSE':           {'label': 'Runny Nose',               'type': 'select', 'options': severity_options},
        'SNEEZING':             {'label': 'Sneezing',                 'type': 'select', 'options': severity_options},
        'SORE_THROAT':          {'label': 'Sore Throat',              'type': 'select', 'options': severity_options},
        'STRIDOR':              {'label': 'Stridor (Noisy Breathing)', 'type': 'select', 'options': severity_options},
        'VOMITING':             {'label': 'Vomiting',                 'type': 'select', 'options': severity_options},
        'WEIGHT_LOSS':          {'label': 'Weight Loss',              'type': 'select', 'options': severity_options},
        'WHEEZING':             {'label': 'Wheezing',                 'type': 'select', 'options': severity_options},
    }

    return X, y, feature_names, feature_info, le


# ─────────────────────────────────────────────────────────────
# TRAIN & EVALUATE
# ─────────────────────────────────────────────────────────────
def train_and_evaluate(X, y, is_multiclass=False):
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    candidates = {
        'GradientBoosting': Pipeline([
            ('scaler', StandardScaler()),
            ('model', GradientBoostingClassifier(
                n_estimators=200, learning_rate=0.1, max_depth=4, random_state=42
            ))
        ]),
        'RandomForest': Pipeline([
            ('scaler', StandardScaler()),
            ('model', RandomForestClassifier(
                n_estimators=200, max_depth=10, min_samples_split=4, random_state=42
            ))
        ]),
        'LogisticRegression': Pipeline([
            ('scaler', StandardScaler()),
            ('model', LogisticRegression(
                max_iter=1000, random_state=42,
                multi_class='multinomial' if is_multiclass else 'auto'
            ))
        ]),
    }

    scoring = 'roc_auc_ovr_weighted' if is_multiclass else 'roc_auc'
    cv      = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    for name, pipe in candidates.items():
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring=scoring)
        results[name] = scores.mean()
        info(f'{name:22s}  CV AUC: {scores.mean():.4f} +/- {scores.std():.4f}')

    best_name = max(results, key=results.get)
    best_pipe = candidates[best_name]
    best_pipe.fit(X_train, y_train)

    y_pred  = best_pipe.predict(X_test)
    y_proba = best_pipe.predict_proba(X_test)
    avg     = 'weighted' if is_multiclass else 'binary'
    if is_multiclass:
        auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
    else:
        auc = roc_auc_score(y_test, y_proba[:, 1])

    metrics = {
        'best_model':    best_name,
        'accuracy':      round(float(accuracy_score(y_test, y_pred)) * 100, 2),
        'precision':     round(float(precision_score(y_test, y_pred, average=avg, zero_division=0)) * 100, 2),
        'recall':        round(float(recall_score(y_test, y_pred, average=avg, zero_division=0)) * 100, 2),
        'f1':            round(float(f1_score(y_test, y_pred, average=avg, zero_division=0)) * 100, 2),
        'roc_auc':       round(float(auc) * 100, 2),
        'cv_auc':        round(float(results[best_name]) * 100, 2),
        'train_size':    int(len(X_train)),
        'test_size':     int(len(X_test)),
        'is_multiclass': is_multiclass,
    }

    print(f'\n  {BOLD}Best model : {GREEN}{best_name}{RESET}')
    print(f'  {"Accuracy":<12} {metrics["accuracy"]}%')
    print(f'  {"Precision":<12} {metrics["precision"]}%')
    print(f'  {"Recall":<12} {metrics["recall"]}%')
    print(f'  {"F1 Score":<12} {metrics["f1"]}%')
    print(f'  {"ROC AUC":<12} {metrics["roc_auc"]}%')
    print(f'  {"CV AUC":<12} {metrics["cv_auc"]}%  (5-fold on train set)')

    return best_pipe, metrics


# ─────────────────────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────────────────────
def save_model(pipeline, feature_names, feature_info, metrics,
               department, condition, label_encoder=None, class_names=None):
    payload = {
        'pipeline':      pipeline,
        'feature_names': feature_names,
        'feature_info':  feature_info,
        'metrics':       metrics,
        'department':    department,
        'condition':     condition,
        'label_encoder': label_encoder,
        'class_names':   class_names,
    }
    path = os.path.join(OUTPUT_DIR, f'{department.lower()}_model.pkl')
    with open(path, 'wb') as f:
        pickle.dump(payload, f)
    ok(f'Saved -> {path}')
    return path


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    print(f'\n{BOLD}Symptom-Based Disease Prediction - Model Training{RESET}')
    print('Departments: General Practice | Cardiology | Endocrinology | Psychiatry | Gastroenterology | Pediatrics\n')

    saved_paths = []
    skipped     = []

    jobs = [
        # title                                      path                      loader                  dept                 condition                    is_multi  class_names
        # ('1 / 6  GENERAL PRACTICE (41 Diseases)',    GENERAL_PRACTICE_PATH,    load_general_practice,   'general_practice',  '41 Common Diseases',        True,     None),
        # ('2 / 6  CARDIOLOGY (Heart Disease Risk)',   CARDIOLOGY_PATH,          load_cardiology,         'cardiology',        'Heart Disease',             False,    ['No Risk', 'At Risk']),
        # ('3 / 6  ENDOCRINOLOGY (Diabetes Risk)',     ENDOCRINOLOGY_PATH,       load_endocrinology,      'endocrinology',     'Diabetes',                  False,    ['Negative', 'Positive']),
        # ('4 / 6  PSYCHIATRY (Mental Disorder)',      PSYCHIATRY_PATH,          load_psychiatry,         'psychiatry',        'Mental Health Disorder',    True,     None),
        ('5 / 6  GASTROENTEROLOGY (GI Diseases)',    GASTROENTEROLOGY_PATH,    load_gastroenterology,   'gastroenterology',  'GI Disease',                True,     None),
        ('6 / 6  PEDIATRICS (Pediatric Diseases)',   PEDIATRICS_PATH,          load_pediatrics,         'pediatrics',        'Pediatric Disease',         True,     None),
    ]

    for title, path, loader, dept, condition, is_multi, class_names in jobs:
        section(title)
        if not os.path.exists(path):
            err(f'File not found: {path}')
            err('Update the path at the top of this script and re-run.')
            skipped.append(dept)
            continue

        info(f'Loading {path}')
        X, y, feat_names, feat_info, le = loader(path)
        pipeline, metrics = train_and_evaluate(X, y, is_multiclass=is_multi)

        # For multiclass departments le holds the LabelEncoder with class names
        resolved_class_names = list(le.classes_) if le is not None else class_names
        p = save_model(pipeline, feat_names, feat_info, metrics,
                       dept, condition, label_encoder=le,
                       class_names=resolved_class_names)
        saved_paths.append(p)

    # ── Summary ──────────────────────────────────────────────
    section('SUMMARY')
    if saved_paths:
        print(f'\n  {"Department":<22} {"Condition":<28} {"Model":<22} {"Accuracy":>10} {"AUC":>8}')
        print(f'  {"-"*95}')
        for p in saved_paths:
            with open(p, 'rb') as f:
                d = pickle.load(f)
            m = d['metrics']
            print(f'  {d["department"]:<22} {d["condition"]:<28} {m["best_model"]:<22} {m["accuracy"]:>9}% {m["roc_auc"]:>7}%')

    if skipped:
        print(f'\n  {YELLOW}Skipped: {", ".join(skipped)}{RESET}')

    if saved_paths:
        print(f'\n  Models saved to: {OUTPUT_DIR}')
    print()


if __name__ == '__main__':
    main()