"""
generate_gastro_dataset.py
──────────────────────────
Generates a clinically realistic synthetic gastroenterology dataset
with real symptom-to-disease signal.

8 GI conditions × ~3 750 patients = 30 000 rows
16 symptom columns (binary) + 6 patient profile columns

Run:  python generate_gastro_dataset.py
Output: datasets/gastroenterology_symptoms.csv
"""

import os
import numpy as np
import pandas as pd

np.random.seed(42)

# ─────────────────────────────────────────────────────────────
# SYMPTOM PROBABILITY TABLE
# Each value is P(symptom present | disease).
# Baseline noise for unlisted symptoms = 0.05.
# ─────────────────────────────────────────────────────────────
DISEASES = {
    'GERD': {
        'heartburn':       0.92,
        'regurgitation':   0.88,
        'chest_pain':      0.62,
        'hoarseness':      0.40,
        'chronic_cough':   0.42,
        'dysphagia':       0.32,
        'bloating':        0.55,
        'nausea':          0.45,
        'vomiting':        0.28,
        'abdominal_pain':  0.30,
        'diarrhea':        0.08,
        'constipation':    0.12,
        'rectal_bleeding': 0.02,
        'weight_loss':     0.12,
        'fever':           0.03,
        'fatigue':         0.30,
    },
    'Irritable Bowel Syndrome': {
        'abdominal_pain':  0.92,
        'bloating':        0.88,
        'diarrhea':        0.72,
        'constipation':    0.68,
        'nausea':          0.42,
        'vomiting':        0.28,
        'fatigue':         0.45,
        'heartburn':       0.22,
        'regurgitation':   0.12,
        'chest_pain':      0.10,
        'rectal_bleeding': 0.04,
        'weight_loss':     0.12,
        'fever':           0.05,
        'dysphagia':       0.05,
        'hoarseness':      0.04,
        'chronic_cough':   0.06,
    },
    'Peptic Ulcer Disease': {
        'abdominal_pain':  0.94,
        'nausea':          0.68,
        'vomiting':        0.58,
        'heartburn':       0.62,
        'regurgitation':   0.32,
        'weight_loss':     0.42,
        'rectal_bleeding': 0.35,
        'bloating':        0.48,
        'diarrhea':        0.22,
        'constipation':    0.18,
        'fever':           0.18,
        'fatigue':         0.50,
        'chest_pain':      0.22,
        'dysphagia':       0.22,
        'hoarseness':      0.06,
        'chronic_cough':   0.06,
    },
    "Crohn's Disease": {
        'diarrhea':        0.90,
        'abdominal_pain':  0.88,
        'weight_loss':     0.78,
        'fever':           0.62,
        'rectal_bleeding': 0.52,
        'nausea':          0.52,
        'vomiting':        0.38,
        'bloating':        0.58,
        'fatigue':         0.72,
        'constipation':    0.12,
        'heartburn':       0.16,
        'chest_pain':      0.10,
        'regurgitation':   0.10,
        'dysphagia':       0.16,
        'hoarseness':      0.05,
        'chronic_cough':   0.10,
    },
    'Ulcerative Colitis': {
        'rectal_bleeding': 0.92,
        'diarrhea':        0.90,
        'abdominal_pain':  0.82,
        'fever':           0.58,
        'weight_loss':     0.62,
        'nausea':          0.48,
        'vomiting':        0.32,
        'bloating':        0.52,
        'fatigue':         0.68,
        'constipation':    0.16,
        'heartburn':       0.10,
        'chest_pain':      0.08,
        'regurgitation':   0.08,
        'dysphagia':       0.10,
        'hoarseness':      0.04,
        'chronic_cough':   0.05,
    },
    'Celiac Disease': {
        'diarrhea':        0.86,
        'bloating':        0.82,
        'abdominal_pain':  0.76,
        'weight_loss':     0.72,
        'nausea':          0.58,
        'fatigue':         0.82,
        'constipation':    0.32,
        'vomiting':        0.38,
        'heartburn':       0.16,
        'rectal_bleeding': 0.06,
        'fever':           0.12,
        'regurgitation':   0.10,
        'chest_pain':      0.06,
        'dysphagia':       0.16,
        'hoarseness':      0.05,
        'chronic_cough':   0.06,
    },
    'Gallstones / Cholecystitis': {
        'abdominal_pain':  0.94,
        'nausea':          0.78,
        'vomiting':        0.68,
        'fever':           0.48,
        'bloating':        0.52,
        'heartburn':       0.32,
        'weight_loss':     0.28,
        'diarrhea':        0.28,
        'chest_pain':      0.32,
        'fatigue':         0.48,
        'rectal_bleeding': 0.04,
        'constipation':    0.16,
        'regurgitation':   0.22,
        'dysphagia':       0.10,
        'hoarseness':      0.05,
        'chronic_cough':   0.05,
    },
    'Colorectal Cancer': {
        'rectal_bleeding': 0.82,
        'weight_loss':     0.80,
        'diarrhea':        0.66,
        'constipation':    0.62,
        'abdominal_pain':  0.72,
        'nausea':          0.48,
        'vomiting':        0.38,
        'bloating':        0.58,
        'fever':           0.32,
        'fatigue':         0.78,
        'heartburn':       0.16,
        'chest_pain':      0.10,
        'regurgitation':   0.08,
        'dysphagia':       0.22,
        'hoarseness':      0.06,
        'chronic_cough':   0.08,
    },
}

SYMPTOMS = [
    'heartburn', 'regurgitation', 'chest_pain', 'hoarseness',
    'chronic_cough', 'dysphagia', 'bloating', 'nausea',
    'vomiting', 'abdominal_pain', 'diarrhea', 'constipation',
    'rectal_bleeding', 'weight_loss', 'fever', 'fatigue',
]

NOISE_P = 0.05   # baseline false-positive rate for unlisted symptoms

# ─────────────────────────────────────────────────────────────
# PATIENT PROFILE GENERATORS
# Age and gender distributions are clinically motivated.
# ─────────────────────────────────────────────────────────────
PROFILE = {
    'GERD':                       dict(age_mu=48, age_sd=14, female_p=0.45, bmi_mu=28, bmi_sd=5),
    'Irritable Bowel Syndrome':   dict(age_mu=34, age_sd=12, female_p=0.72, bmi_mu=24, bmi_sd=4),
    'Peptic Ulcer Disease':       dict(age_mu=52, age_sd=15, female_p=0.38, bmi_mu=25, bmi_sd=5),
    "Crohn's Disease":            dict(age_mu=30, age_sd=12, female_p=0.50, bmi_mu=22, bmi_sd=4),
    'Ulcerative Colitis':         dict(age_mu=38, age_sd=14, female_p=0.48, bmi_mu=23, bmi_sd=4),
    'Celiac Disease':             dict(age_mu=36, age_sd=14, female_p=0.62, bmi_mu=22, bmi_sd=4),
    'Gallstones / Cholecystitis': dict(age_mu=50, age_sd=14, female_p=0.68, bmi_mu=29, bmi_sd=6),
    'Colorectal Cancer':          dict(age_mu=64, age_sd=11, female_p=0.44, bmi_mu=27, bmi_sd=5),
}

SMOKING = {
    'GERD':                       0.35,
    'Irritable Bowel Syndrome':   0.22,
    'Peptic Ulcer Disease':       0.48,
    "Crohn's Disease":            0.40,
    'Ulcerative Colitis':         0.14,   # protective in UC
    'Celiac Disease':             0.18,
    'Gallstones / Cholecystitis': 0.22,
    'Colorectal Cancer':          0.32,
}

ALCOHOL = {
    'GERD':                       0.42,
    'Irritable Bowel Syndrome':   0.28,
    'Peptic Ulcer Disease':       0.45,
    "Crohn's Disease":            0.25,
    'Ulcerative Colitis':         0.22,
    'Celiac Disease':             0.20,
    'Gallstones / Cholecystitis': 0.35,
    'Colorectal Cancer':          0.40,
}

FAMILY_HX = {
    'GERD':                       0.30,
    'Irritable Bowel Syndrome':   0.28,
    'Peptic Ulcer Disease':       0.25,
    "Crohn's Disease":            0.20,
    'Ulcerative Colitis':         0.18,
    'Celiac Disease':             0.35,
    'Gallstones / Cholecystitis': 0.30,
    'Colorectal Cancer':          0.30,
}

DURATION_DAYS = {   # typical symptom duration range
    'GERD':                       (30,  365*5),
    'Irritable Bowel Syndrome':   (90,  365*8),
    'Peptic Ulcer Disease':       (7,   180),
    "Crohn's Disease":            (30,  365*10),
    'Ulcerative Colitis':         (14,  365*8),
    'Celiac Disease':             (90,  365*10),
    'Gallstones / Cholecystitis': (1,   90),
    'Colorectal Cancer':          (30,  365*3),
}


def generate(n_per_disease: int = 3_750) -> pd.DataFrame:
    rows, targets = [], []

    for disease, sym_probs in DISEASES.items():
        prof = PROFILE[disease]

        for _ in range(n_per_disease):
            # ── Symptom columns ───────────────────────────────
            row = {}
            for sym in SYMPTOMS:
                p = sym_probs.get(sym, NOISE_P)
                row[sym] = int(np.random.rand() < p)

            # ── Patient profile ───────────────────────────────
            age = int(np.clip(np.random.normal(prof['age_mu'], prof['age_sd']), 18, 90))
            row['age']            = age
            row['gender']         = int(np.random.rand() > prof['female_p'])  # 1=Male, 0=Female
            row['bmi']            = round(np.clip(np.random.normal(prof['bmi_mu'], prof['bmi_sd']), 15, 50), 1)
            row['smoking']        = int(np.random.rand() < SMOKING[disease])
            row['alcohol_use']    = int(np.random.rand() < ALCOHOL[disease])
            row['family_history'] = int(np.random.rand() < FAMILY_HX[disease])

            lo, hi = DURATION_DAYS[disease]
            row['symptom_duration_days'] = int(np.random.randint(lo, hi + 1))

            rows.append(row)
            targets.append(disease)

    df = pd.DataFrame(rows)
    df['disease'] = targets

    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df


if __name__ == '__main__':
    os.makedirs('datasets', exist_ok=True)
    df = generate(n_per_disease=3_750)
    out = 'datasets/gastroenterology_symptoms.csv'
    df.to_csv(out, index=False)
    print(f'Saved {len(df):,} rows to {out}')
    print(f'Columns: {list(df.columns)}')
    print(f'Class distribution:\n{df["disease"].value_counts()}')