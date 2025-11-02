"""
Advanced Model Improvements - HiLabs Hackathon 2025
Techniques to maximize accuracy and real-world relevance

This script implements:
1. Hyperparameter optimization (Optuna)
2. Target transformation for skewed distributions
3. Advanced feature engineering
4. Model stacking
5. Outlier handling
6. Feature importance-based selection
7. Post-processing techniques
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# ML libraries
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import RobustScaler, PowerTransformer
import xgboost as xgb
import lightgbm as lgb

# Statistical analysis
from scipy import stats
try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

# Hyperparameter optimization
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Optuna not available - install with: pip install optuna")

import pickle
import time

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

# Visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Define paths
BASE_DIR = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
TRAIN_DIR = BASE_DIR / 'pcms_hackathon_data' / 'train'
TEST_DIR = BASE_DIR / 'pcms_hackathon_data' / 'test'
OUTPUT_DIR = BASE_DIR / 'analysis_outputs'
OUTPUT_DIR.mkdir(exist_ok=True)
MODELS_DIR = BASE_DIR / 'models'
MODELS_DIR.mkdir(exist_ok=True)

print("="*80)
print("ADVANCED MODEL IMPROVEMENTS")
print("="*80)

# =============================================================================
# 1. LOAD DATA (Use existing preprocessed data or reload)
# =============================================================================

print("\n1. Loading and preparing data...")

# Load data
train_patient = pd.read_csv(TRAIN_DIR / 'patient.csv')
train_risk = pd.read_csv(TRAIN_DIR / 'risk.csv')
train_visit = pd.read_csv(TRAIN_DIR / 'visit.csv')
train_diagnosis = pd.read_csv(TRAIN_DIR / 'diagnosis.csv')
train_care = pd.read_csv(TRAIN_DIR / 'care.csv')

test_patient = pd.read_csv(TEST_DIR / 'patient.csv')
test_visit = pd.read_csv(TEST_DIR / 'visit.csv')
test_diagnosis = pd.read_csv(TEST_DIR / 'diagnosis.csv')
test_care = pd.read_csv(TEST_DIR / 'care.csv')

# Use the improved feature engineering from 03_improved_model_training.py
# (Import or copy the engineer_focused_features function)
# For now, let's create a simplified version here

print("✓ Data loaded")

# Visualization: Data Overview
print("\n  Creating data overview visualizations...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Step 1: Data Loading - Overview', fontsize=16, fontweight='bold')

# Data shapes
data_shapes = {
    'patient': len(train_patient),
    'visit': len(train_visit),
    'diagnosis': len(train_diagnosis),
    'care': len(train_care),
    'risk': len(train_risk)
}
axes[0, 0].bar(data_shapes.keys(), data_shapes.values(), color='steelblue')
axes[0, 0].set_title('Training Data Record Counts', fontweight='bold')
axes[0, 0].set_ylabel('Number of Records')
axes[0, 0].tick_params(axis='x', rotation=45)
for i, v in enumerate(data_shapes.values()):
    axes[0, 0].text(i, v, str(v), ha='center', va='bottom')

# Missing values
train_data_list = [train_patient, train_visit, train_diagnosis, train_care, train_risk]
train_data_names = ['patient', 'visit', 'diagnosis', 'care', 'risk']
missing_counts = [df.isnull().sum().sum() for df in train_data_list]
axes[0, 1].bar(train_data_names, missing_counts, color='coral')
axes[0, 1].set_title('Missing Values Count', fontweight='bold')
axes[0, 1].set_ylabel('Missing Values')
axes[0, 1].tick_params(axis='x', rotation=45)

# Age distribution
axes[1, 0].hist(train_patient['age'].dropna(), bins=30, color='skyblue', edgecolor='black')
axes[1, 0].set_title('Age Distribution', fontweight='bold')
axes[1, 0].set_xlabel('Age')
axes[1, 0].set_ylabel('Frequency')

# Risk score distribution
axes[1, 1].hist(train_risk['risk_score'], bins=50, color='lightgreen', edgecolor='black')
axes[1, 1].set_title('Risk Score Distribution (Target)', fontweight='bold')
axes[1, 1].set_xlabel('Risk Score')
axes[1, 1].set_ylabel('Frequency')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '01_data_overview.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: analysis_outputs/01_data_overview.png")

# =============================================================================
# 2. TARGET TRANSFORMATION (For skewed distributions)
# =============================================================================

print("\n2. Analyzing target distribution for transformation...")

# Check target skewness
target_skew = train_risk['risk_score'].skew()
target_kurt = train_risk['risk_score'].kurtosis()

print(f"  Target Skewness: {target_skew:.2f}")
print(f"  Target Kurtosis: {target_kurt:.2f}")

# Try different transformations
transformations = {
    'original': lambda x: x,
    'log1p': lambda x: np.log1p(x),
    'sqrt': lambda x: np.sqrt(x),
    'box_cox': None  # Will use PowerTransformer
}

best_transformation = 'original'
best_score = float('inf')

# Test transformations with simple model
X_sample = train_patient[['age']].fillna(train_patient['age'].median())
y_sample = train_risk['risk_score']

for trans_name, trans_func in transformations.items():
    if trans_func is None:
        continue
    try:
        y_transformed = trans_func(y_sample)
        # Simple correlation test
        corr = abs(X_sample['age'].corr(y_transformed))
        if corr > 0.1:  # Some correlation exists
            print(f"  {trans_name}: Correlation = {corr:.3f}")
    except:
        pass

print("✓ Target analysis complete")

# Visualization: Target Distribution Analysis
print("\n  Creating target transformation visualizations...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Step 2: Target Transformation Analysis', fontsize=16, fontweight='bold')

# Original distribution
axes[0, 0].hist(y_sample, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
axes[0, 0].axvline(y_sample.mean(), color='red', linestyle='--', label=f'Mean: {y_sample.mean():.2f}')
axes[0, 0].axvline(y_sample.median(), color='green', linestyle='--', label=f'Median: {y_sample.median():.2f}')
axes[0, 0].set_title(f'Original Distribution\n(Skewness: {target_skew:.2f}, Kurtosis: {target_kurt:.2f})', fontweight='bold')
axes[0, 0].set_xlabel('Risk Score')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Log-transformed distribution
y_log = np.log1p(y_sample)
log_skew = y_log.skew()
log_kurt = y_log.kurtosis()
axes[0, 1].hist(y_log, bins=50, color='coral', edgecolor='black', alpha=0.7)
axes[0, 1].axvline(y_log.mean(), color='red', linestyle='--', label=f'Mean: {y_log.mean():.2f}')
axes[0, 1].axvline(y_log.median(), color='green', linestyle='--', label=f'Median: {y_log.median():.2f}')
axes[0, 1].set_title(f'Log1p Transformed Distribution\n(Skewness: {log_skew:.2f}, Kurtosis: {log_kurt:.2f})', fontweight='bold')
axes[0, 1].set_xlabel('log1p(Risk Score)')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Q-Q plot original
stats.probplot(y_sample, dist="norm", plot=axes[1, 0])
axes[1, 0].set_title('Q-Q Plot: Original Distribution', fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

# Q-Q plot transformed
stats.probplot(y_log, dist="norm", plot=axes[1, 1])
axes[1, 1].set_title('Q-Q Plot: Log-Transformed Distribution', fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '02_target_transformation.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: analysis_outputs/02_target_transformation.png")

# =============================================================================
# 3. OUTLIER HANDLING
# =============================================================================

print("\n3. Outlier detection and handling...")

def detect_outliers_iqr(data, columns):
    """Detect outliers using IQR method"""
    outliers = {}
    for col in columns:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outlier_count = ((data[col] < lower_bound) | (data[col] > upper_bound)).sum()
        outliers[col] = {
            'count': outlier_count,
            'percentage': (outlier_count / len(data)) * 100,
            'bounds': (lower_bound, upper_bound)
        }
    return outliers

# Detect outliers in target
target_outliers = detect_outliers_iqr(train_risk, ['risk_score'])
print(f"  Risk score outliers: {target_outliers['risk_score']['count']} ({target_outliers['risk_score']['percentage']:.2f}%)")

# Strategy: Cap extreme outliers in features, but keep in target (they're real high-risk patients)
print("✓ Outlier analysis complete")

# Visualization: Outlier Analysis
print("\n  Creating outlier analysis visualizations...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Step 3: Outlier Detection and Handling', fontsize=16, fontweight='bold')

# Box plot for risk score
risk_data = train_risk['risk_score']
Q1 = risk_data.quantile(0.25)
Q3 = risk_data.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

axes[0].boxplot(risk_data, vert=True, patch_artist=True,
                boxprops=dict(facecolor='lightblue', alpha=0.7))
axes[0].axhline(lower_bound, color='red', linestyle='--', label=f'Lower bound: {lower_bound:.2f}')
axes[0].axhline(upper_bound, color='red', linestyle='--', label=f'Upper bound: {upper_bound:.2f}')
axes[0].set_title(f'Risk Score Outliers\n(Outliers: {target_outliers["risk_score"]["count"]}, {target_outliers["risk_score"]["percentage"]:.2f}%)', 
                  fontweight='bold')
axes[0].set_ylabel('Risk Score')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Distribution with outlier region highlighted
axes[1].hist(risk_data, bins=50, color='steelblue', edgecolor='black', alpha=0.7, label='All Data')
outliers = risk_data[(risk_data < lower_bound) | (risk_data > upper_bound)]
if len(outliers) > 0:
    axes[1].hist(outliers, bins=20, color='red', edgecolor='black', alpha=0.7, label='Outliers')
axes[1].axvline(lower_bound, color='red', linestyle='--', linewidth=2)
axes[1].axvline(upper_bound, color='red', linestyle='--', linewidth=2)
axes[1].fill_betweenx([0, axes[1].get_ylim()[1]], lower_bound, upper_bound, 
                       alpha=0.2, color='green', label='Normal Range')
axes[1].set_title('Risk Score Distribution with Outlier Bounds', fontweight='bold')
axes[1].set_xlabel('Risk Score')
axes[1].set_ylabel('Frequency')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '03_outlier_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: analysis_outputs/03_outlier_analysis.png")

# =============================================================================
# 4. ADVANCED FEATURE ENGINEERING
# =============================================================================

print("\n4. Advanced feature engineering...")

def engineer_advanced_features(patient, visit, diagnosis, care):
    """Advanced feature engineering with domain insights"""
    
    features = patient[['patient_id', 'age']].copy()
    
    # Age bins (more granular)
    features['age_bin_18_35'] = ((features['age'] >= 18) & (features['age'] <= 35)).astype(int)
    features['age_bin_36_50'] = ((features['age'] >= 36) & (features['age'] <= 50)).astype(int)
    features['age_bin_51_65'] = ((features['age'] >= 51) & (features['age'] <= 65)).astype(int)
    features['age_bin_65_plus'] = (features['age'] > 65).astype(int)
    
    # Hot spotter
    features['is_hot_spotter'] = (patient['hot_spotter_identified_at'] != '0001-01-01').astype(int)
    features['hot_spotter_readmission_flag'] = (patient['hot_spotter_readmission_flag'] == 't').astype(int)
    features['hot_spotter_chronic_flag'] = (patient['hot_spotter_chronic_flag'] == 't').astype(int)
    
    # ===== VISIT FEATURES (Enhanced) =====
    visit_agg = visit.groupby('patient_id').agg({
        'visit_id': 'count',
        'readmsn_ind': lambda x: (x == 't').sum(),
    }).reset_index()
    visit_agg.columns = ['patient_id', 'total_visits', 'readmission_count']
    
    # Visit type counts
    visit_type_counts = visit.groupby(['patient_id', 'visit_type']).size().unstack(fill_value=0)
    visit_type_cols = []
    for col in visit_type_counts.columns:
        col_name = f'visits_{col.lower().replace(" ", "_")}'
        visit_type_counts[col_name] = visit_type_counts[col]
        visit_type_cols.append(col_name)
    visit_type_counts = visit_type_counts[visit_type_cols]
    
    # Visit duration features
    visit['visit_start_dt'] = pd.to_datetime(visit['visit_start_dt'], errors='coerce')
    visit['visit_end_dt'] = pd.to_datetime(visit['visit_end_dt'], errors='coerce')
    visit['visit_duration'] = (visit['visit_end_dt'] - visit['visit_start_dt']).dt.days.fillna(0).clip(0, 365)
    
    visit_duration_agg = visit.groupby('patient_id')['visit_duration'].agg(['mean', 'max', 'min', 'std', 'sum']).reset_index()
    visit_duration_agg.columns = ['patient_id', 'avg_visit_duration', 'max_visit_duration', 
                                  'min_visit_duration', 'std_visit_duration', 'total_visit_days']
    
    # Visit frequency patterns
    visit['visit_start_dt'] = pd.to_datetime(visit['visit_start_dt'], errors='coerce')
    REFERENCE_DATE = pd.to_datetime('2025-03-01')
    visit['days_since_visit'] = (REFERENCE_DATE - visit['visit_start_dt']).dt.days.fillna(365).clip(0, 365*5)
    
    # Recent visits
    for days in [30, 60, 90]:
        recent = visit[visit['days_since_visit'] <= days].groupby('patient_id').size()
        visit_agg[f'visits_last_{days}_days'] = visit_agg['patient_id'].map(recent).fillna(0)
    
    # Emergency visits
    visit['is_emergency'] = ((visit['visit_type'] == 'ER') | (visit['visit_type'] == 'INPATIENT')).astype(int)
    emergency_agg = visit.groupby('patient_id')['is_emergency'].agg(['sum', 'mean']).reset_index()
    emergency_agg.columns = ['patient_id', 'emergency_visit_count', 'emergency_visit_rate']
    
    features = features.merge(visit_agg, on='patient_id', how='left')
    features = features.merge(visit_type_counts, on='patient_id', how='left')
    features = features.merge(visit_duration_agg, on='patient_id', how='left')
    features = features.merge(emergency_agg, on='patient_id', how='left')
    
    # Visit ratios
    if 'total_visits' in features.columns:
        for col in visit_type_cols:
            if col in features.columns:
                features[f'{col}_ratio'] = features[col] / (features['total_visits'] + 1)
    
    # ===== DIAGNOSIS FEATURES =====
    diagnosis['is_chronic'] = (diagnosis['is_chronic'] == 't').astype(int) if diagnosis['is_chronic'].dtype == 'object' else diagnosis['is_chronic'].astype(int)
    chronic_conditions = diagnosis[diagnosis['is_chronic'] == 1]
    
    diagnosis_agg = diagnosis.groupby('patient_id').agg({
        'diagnosis_id': 'count',
        'is_chronic': 'sum',
    }).reset_index()
    diagnosis_agg.columns = ['patient_id', 'total_diagnoses', 'chronic_count']
    
    # Specific conditions
    for condition in ['CANCER', 'DIABETES', 'HYPERTENSION']:
        condition_patients = chronic_conditions[chronic_conditions['condition_name'] == condition]['patient_id'].unique()
        diagnosis_agg[f'has_{condition.lower()}'] = diagnosis_agg['patient_id'].isin(condition_patients).astype(int)
    
    # Severity score
    diagnosis_agg['chronic_severity_score'] = (
        diagnosis_agg['has_cancer'] * 5 +
        diagnosis_agg['has_diabetes'] * 2 +
        diagnosis_agg['has_hypertension'] * 1
    )
    
    features = features.merge(diagnosis_agg, on='patient_id', how='left')
    
    # ===== CARE FEATURES =====
    care['care_gap_ind'] = (care['care_gap_ind'] == 't').astype(int) if care['care_gap_ind'].dtype == 'object' else care['care_gap_ind'].astype(int)
    
    care_agg = care.groupby('patient_id').agg({
        'care_id': 'count',
        'care_gap_ind': 'sum',
    }).reset_index()
    care_agg.columns = ['patient_id', 'total_care_records', 'care_gaps']
    care_agg['care_gap_rate'] = care_agg['care_gaps'] / (care_agg['total_care_records'] + 1)
    
    features = features.merge(care_agg, on='patient_id', how='left')
    
    # Fill missing values
    numeric_cols = features.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col != 'patient_id':
            features[col] = features[col].fillna(0)
    
    # ===== ADVANCED INTERACTIONS =====
    
    # High-risk combinations
    if 'has_cancer' in features.columns and 'visits_inpatient' in features.columns:
        features['cancer_x_inpatient'] = features['has_cancer'] * features['visits_inpatient']
    
    if 'has_cancer' in features.columns and 'readmission_count' in features.columns:
        features['cancer_x_readmission'] = features['has_cancer'] * features['readmission_count']
    
    if 'chronic_severity_score' in features.columns and 'emergency_visit_count' in features.columns:
        features['chronic_x_emergency'] = features['chronic_severity_score'] * features['emergency_visit_count']
    
    if 'care_gap_rate' in features.columns and 'has_cancer' in features.columns:
        features['gaps_x_cancer'] = features['care_gap_rate'] * features['has_cancer'] * 10
    
    # Age risk interactions
    if 'age' in features.columns:
        if 'has_cancer' in features.columns:
            features['elderly_cancer'] = ((features['age'] > 65) & (features['has_cancer'] == 1)).astype(int)
        if 'emergency_visit_count' in features.columns:
            features['elderly_emergency'] = ((features['age'] > 65) & (features['emergency_visit_count'] > 0)).astype(int)
    
    # Composite high-risk flags
    features['high_risk_flag'] = (
        (features.get('has_cancer', 0) == 1) |
        (features.get('visits_inpatient', 0) > 2) |
        (features.get('readmission_count', 0) > 1) |
        (features.get('is_hot_spotter', 0) == 1)
    ).astype(int)
    
    # Risk score components
    features['visit_risk_component'] = (
        features.get('emergency_visit_count', 0) * 2 +
        features.get('readmission_count', 0) * 3
    )
    
    features['chronic_risk_component'] = features.get('chronic_severity_score', 0) * 2
    
    features['care_risk_component'] = features.get('care_gap_rate', 0) * 10
    
    return features

# Engineer features
print("  Engineering features...")
train_features = engineer_advanced_features(train_patient, train_visit, train_diagnosis, train_care)
test_features = engineer_advanced_features(test_patient, test_visit, test_diagnosis, test_care)

print(f"✓ Features engineered: {train_features.shape[1]} features")

# Visualization: Feature Engineering Summary
print("\n  Creating feature engineering visualizations...")
feature_categories = {
    'Demographics': ['age', 'age_bin_18_35', 'age_bin_36_50', 'age_bin_51_65', 'age_bin_65_plus'],
    'Hot Spotter': ['is_hot_spotter', 'hot_spotter_readmission_flag', 'hot_spotter_chronic_flag'],
    'Visits': [col for col in train_features.columns if 'visit' in col.lower() and col not in ['patient_id']],
    'Diagnosis': [col for col in train_features.columns if 'diagnosis' in col.lower() or 'chronic' in col.lower() or 'cancer' in col.lower() or 'diabetes' in col.lower() or 'hypertension' in col.lower()],
    'Care': [col for col in train_features.columns if 'care' in col.lower() or 'gap' in col.lower()],
    'Interactions': [col for col in train_features.columns if 'x_' in col or 'elderly' in col.lower() or 'risk_component' in col.lower() or 'high_risk' in col.lower()]
}
category_counts = {cat: len([f for f in feats if f in train_features.columns]) for cat, feats in feature_categories.items()}

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Step 4: Feature Engineering Summary', fontsize=16, fontweight='bold')

# Feature counts by category
axes[0].bar(category_counts.keys(), category_counts.values(), color='steelblue')
axes[0].set_title('Features by Category', fontweight='bold')
axes[0].set_ylabel('Number of Features')
axes[0].tick_params(axis='x', rotation=45)
for i, (cat, count) in enumerate(category_counts.items()):
    axes[0].text(i, count, str(count), ha='center', va='bottom', fontweight='bold')

# Top 15 feature correlation with target (if available)
train_data_viz = train_features.merge(train_risk[['patient_id', 'risk_score']], on='patient_id', how='inner')
if 'risk_score' in train_data_viz.columns:
    numeric_features = train_data_viz.select_dtypes(include=[np.number]).columns.tolist()
    if 'patient_id' in numeric_features:
        numeric_features.remove('patient_id')
    if 'risk_score' in numeric_features:
        numeric_features.remove('risk_score')
    
    if len(numeric_features) > 0:
        correlations = train_data_viz[numeric_features + ['risk_score']].corr()['risk_score'].abs().sort_values(ascending=False)
        top_features = correlations.head(15).index.tolist()
        if 'risk_score' in top_features:
            top_features.remove('risk_score')
        top_corrs = correlations[top_features].sort_values(ascending=True)
        
        axes[1].barh(range(len(top_corrs)), top_corrs.values, color='coral')
        axes[1].set_yticks(range(len(top_corrs)))
        axes[1].set_yticklabels([f[:30] for f in top_corrs.index])  # Truncate long names
        axes[1].set_title('Top 15 Feature Correlations with Risk Score', fontweight='bold')
        axes[1].set_xlabel('Absolute Correlation')
        axes[1].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '04_feature_engineering.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: analysis_outputs/04_feature_engineering.png")

# =============================================================================
# 5. PREPARE DATA WITH TRANSFORMATIONS
# =============================================================================

print("\n5. Preparing data with transformations...")

train_data = train_features.merge(train_risk[['patient_id', 'risk_score']], on='patient_id', how='inner')

feature_cols = [col for col in train_data.columns if col not in ['patient_id', 'risk_score']]

X = train_data[feature_cols].fillna(0)
y = train_data['risk_score']

# Handle outliers in features (cap extreme values)
for col in X.columns:
    if X[col].dtype in [np.int64, np.float64]:
        Q1, Q3 = X[col].quantile([0.01, 0.99])
        X[col] = X[col].clip(lower=Q1, upper=Q3)

# Replace inf and large values
X = X.replace([np.inf, -np.inf], 0)
X = X.clip(lower=-1000, upper=1000)

# Split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"✓ Training set: {X_train.shape}")
print(f"✓ Validation set: {X_val.shape}")

# Try log transformation for target
print("\n  Testing target transformations...")
y_train_log = np.log1p(y_train)
y_val_log = np.log1p(y_val)

# Compare original vs log-transformed
orig_skew = y_train.skew()
log_skew = y_train_log.skew()

print(f"  Original skewness: {orig_skew:.2f}")
print(f"  Log-transformed skewness: {log_skew:.2f}")

use_log_transform = abs(log_skew) < abs(orig_skew)
print(f"  Will use log transformation: {use_log_transform}")

if use_log_transform:
    y_train_model = y_train_log
    y_val_model = y_val_log
else:
    y_train_model = y_train
    y_val_model = y_val

# Visualization: Data Preparation
print("\n  Creating data preparation visualizations...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Step 5: Data Preparation and Transformation', fontsize=16, fontweight='bold')

# Train/Val split visualization
split_data = {'Train': len(y_train), 'Validation': len(y_val)}
axes[0, 0].pie(split_data.values(), labels=split_data.keys(), autopct='%1.1f%%', 
               colors=['steelblue', 'coral'], startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
axes[0, 0].set_title(f'Train/Validation Split\n(Total: {len(y_train) + len(y_val)} samples)', fontweight='bold')

# Target distribution comparison
axes[0, 1].hist(y_train, bins=30, alpha=0.7, label='Train', color='steelblue', edgecolor='black')
axes[0, 1].hist(y_val, bins=30, alpha=0.7, label='Validation', color='coral', edgecolor='black')
axes[0, 1].set_title('Target Distribution: Train vs Validation', fontweight='bold')
axes[0, 1].set_xlabel('Risk Score')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Transformation comparison
axes[1, 0].hist(y_train_model if use_log_transform else y_train, bins=30, 
                color='lightgreen', edgecolor='black', alpha=0.7)
transform_type = 'log1p' if use_log_transform else 'original'
axes[1, 0].set_title(f'Transformed Target Distribution\n(Using: {transform_type})', fontweight='bold')
axes[1, 0].set_xlabel(f'Risk Score ({transform_type})')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].grid(True, alpha=0.3)

# Feature count by type
numeric_features_count = X.select_dtypes(include=[np.number]).shape[1]
axes[1, 1].bar(['Numeric Features'], [numeric_features_count], color='skyblue', width=0.5)
axes[1, 1].text(0, numeric_features_count, str(numeric_features_count), 
               ha='center', va='bottom', fontsize=14, fontweight='bold')
axes[1, 1].set_title(f'Final Feature Set\n({numeric_features_count} features)', fontweight='bold')
axes[1, 1].set_ylabel('Count')
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '05_data_preparation.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: analysis_outputs/05_data_preparation.png")

# =============================================================================
# 6. HYPERPARAMETER OPTIMIZATION (Optuna)
# =============================================================================

print("\n6. Hyperparameter optimization...")

if OPTUNA_AVAILABLE:
    print("  Using Optuna for hyperparameter tuning...")
    
    def objective_xgb(trial):
        """XGBoost objective function"""
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'max_depth': trial.suggest_int('max_depth', 4, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 200, 1500),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 2.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 2.0, log=True),
            'random_state': 42,
            'verbosity': 0
        }
        
        model = xgb.XGBRegressor(**params)
        scores = cross_val_score(model, X_train, y_train_model, 
                                cv=KFold(n_splits=3, shuffle=True, random_state=42),
                                scoring='neg_root_mean_squared_error',
                                n_jobs=-1)
        return -scores.mean()
    
    # Run optimization
    print("  Optimizing XGBoost...")
    study_xgb = optuna.create_study(direction='minimize', study_name='xgb_optimization')
    study_xgb.optimize(objective_xgb, n_trials=20, show_progress_bar=True)
    
    best_xgb_params = study_xgb.best_params
    best_xgb_params['objective'] = 'reg:squarederror'
    best_xgb_params['eval_metric'] = 'rmse'
    best_xgb_params['random_state'] = 42
    best_xgb_params['verbosity'] = 0
    
    print(f"  ✓ Best XGBoost RMSE: {study_xgb.best_value:.4f}")
    print(f"  ✓ Best params: {best_xgb_params}")
    
    # Visualization: Optuna Optimization History
    print("\n  Creating hyperparameter optimization visualizations...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Step 6: Hyperparameter Optimization (Optuna)', fontsize=16, fontweight='bold')
    
    # Optimization history
    opt_history = study_xgb.trials_dataframe()
    axes[0].plot(range(len(opt_history)), opt_history['value'], marker='o', 
                 color='steelblue', linewidth=2, markersize=4)
    axes[0].axhline(study_xgb.best_value, color='red', linestyle='--', 
                   label=f'Best RMSE: {study_xgb.best_value:.4f}', linewidth=2)
    axes[0].set_title('Optimization History', fontweight='bold')
    axes[0].set_xlabel('Trial Number')
    axes[0].set_ylabel('RMSE')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Parameter importance (if available)
    try:
        param_importance = optuna.importance.get_param_importances(study_xgb)
        if param_importance:
            params = list(param_importance.keys())
            importances = list(param_importance.values())
            axes[1].barh(range(len(params)), importances, color='coral')
            axes[1].set_yticks(range(len(params)))
            axes[1].set_yticklabels(params)
            axes[1].set_title('Parameter Importance', fontweight='bold')
            axes[1].set_xlabel('Importance')
            axes[1].grid(True, alpha=0.3, axis='x')
    except:
        axes[1].text(0.5, 0.5, 'Parameter importance\nnot available', 
                    ha='center', va='center', transform=axes[1].transAxes,
                    fontsize=12, style='italic')
        axes[1].set_title('Parameter Importance', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '06_hyperparameter_optimization.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: analysis_outputs/06_hyperparameter_optimization.png")
else:
    print("  Optuna not available, using optimized defaults...")
    best_xgb_params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'max_depth': 7,
        'learning_rate': 0.03,
        'n_estimators': 1000,
        'subsample': 0.85,
        'colsample_bytree': 0.85,
        'min_child_weight': 2,
        'reg_alpha': 0.2,
        'reg_lambda': 2,
        'random_state': 42,
        'verbosity': 0
    }

# =============================================================================
# 7. TRAIN OPTIMIZED MODELS
# =============================================================================

print("\n7. Training optimized models...")

models_optimized = {}
predictions_val_opt = {}

# XGBoost (optimized)
print("  Training XGBoost...")
start = time.time()
xgb_opt = xgb.XGBRegressor(**best_xgb_params)
xgb_opt.fit(X_train, y_train_model,
            eval_set=[(X_train, y_train_model), (X_val, y_val_model)],
            verbose=False)
models_optimized['xgb'] = xgb_opt

pred_val_xgb = xgb_opt.predict(X_val)
if use_log_transform:
    pred_val_xgb = np.expm1(pred_val_xgb)
predictions_val_opt['xgb'] = pred_val_xgb

xgb_rmse = np.sqrt(mean_squared_error(y_val, pred_val_xgb))
print(f"    ✓ XGBoost RMSE: {xgb_rmse:.4f} (trained in {time.time()-start:.2f}s)")

# LightGBM (optimized)
print("  Training LightGBM...")
start = time.time()
lgb_opt = lgb.LGBMRegressor(
    objective='regression',
    metric='rmse',
    max_depth=7,
    learning_rate=0.03,
    n_estimators=1000,
    subsample=0.85,
    colsample_bytree=0.85,
    min_child_samples=2,
    reg_alpha=0.2,
    reg_lambda=2,
    random_state=42,
    verbosity=-1
)
lgb_opt.fit(X_train, y_train_model,
           eval_set=[(X_train, y_train_model), (X_val, y_val_model)],
           callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False),
                     lgb.log_evaluation(period=0)])
models_optimized['lgb'] = lgb_opt

pred_val_lgb = lgb_opt.predict(X_val)
if use_log_transform:
    pred_val_lgb = np.expm1(pred_val_lgb)
predictions_val_opt['lgb'] = pred_val_lgb

lgb_rmse = np.sqrt(mean_squared_error(y_val, pred_val_lgb))
print(f"    ✓ LightGBM RMSE: {lgb_rmse:.4f} (trained in {time.time()-start:.2f}s)")

# Visualization: Model Training Results
print("\n  Creating model training visualizations...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Step 7: Model Training Results', fontsize=16, fontweight='bold')

# Predictions vs Actual - XGBoost
axes[0, 0].scatter(y_val, pred_val_xgb, alpha=0.5, color='steelblue', s=20)
axes[0, 0].plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 
                'r--', linewidth=2, label='Perfect Prediction')
axes[0, 0].set_title(f'XGBoost: Predictions vs Actual\n(RMSE: {xgb_rmse:.4f})', fontweight='bold')
axes[0, 0].set_xlabel('Actual Risk Score')
axes[0, 0].set_ylabel('Predicted Risk Score')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Predictions vs Actual - LightGBM
axes[0, 1].scatter(y_val, pred_val_lgb, alpha=0.5, color='coral', s=20)
axes[0, 1].plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 
                'r--', linewidth=2, label='Perfect Prediction')
axes[0, 1].set_title(f'LightGBM: Predictions vs Actual\n(RMSE: {lgb_rmse:.4f})', fontweight='bold')
axes[0, 1].set_xlabel('Actual Risk Score')
axes[0, 1].set_ylabel('Predicted Risk Score')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Residuals - XGBoost
residuals_xgb = y_val - pred_val_xgb
axes[1, 0].scatter(pred_val_xgb, residuals_xgb, alpha=0.5, color='steelblue', s=20)
axes[1, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
axes[1, 0].set_title('XGBoost: Residuals Plot', fontweight='bold')
axes[1, 0].set_xlabel('Predicted Risk Score')
axes[1, 0].set_ylabel('Residuals (Actual - Predicted)')
axes[1, 0].grid(True, alpha=0.3)

# Residuals - LightGBM
residuals_lgb = y_val - pred_val_lgb
axes[1, 1].scatter(pred_val_lgb, residuals_lgb, alpha=0.5, color='coral', s=20)
axes[1, 1].axhline(y=0, color='r', linestyle='--', linewidth=2)
axes[1, 1].set_title('LightGBM: Residuals Plot', fontweight='bold')
axes[1, 1].set_xlabel('Predicted Risk Score')
axes[1, 1].set_ylabel('Residuals (Actual - Predicted)')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '07_model_training.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: analysis_outputs/07_model_training.png")

# =============================================================================
# 8. STACKING MODEL
# =============================================================================

print("\n8. Creating stacking model...")

# Use predictions from base models as features for meta-model
X_val_stacked = pd.DataFrame({
    'xgb_pred': predictions_val_opt['xgb'],
    'lgb_pred': predictions_val_opt['lgb'],
})

# Train meta-model (XGBoost)
meta_model = xgb.XGBRegressor(
    max_depth=3,
    learning_rate=0.1,
    n_estimators=100,
    random_state=42,
    verbosity=0
)

# Get train predictions for stacking
X_train_stacked = pd.DataFrame({
    'xgb_pred': xgb_opt.predict(X_train),
    'lgb_pred': lgb_opt.predict(X_train),
})

if use_log_transform:
    X_train_stacked['xgb_pred'] = np.expm1(X_train_stacked['xgb_pred'])
    X_train_stacked['lgb_pred'] = np.expm1(X_train_stacked['lgb_pred'])

meta_model.fit(X_train_stacked, y_train)

# Stacking predictions
y_val_stacked = meta_model.predict(X_val_stacked)
stacked_rmse = np.sqrt(mean_squared_error(y_val, y_val_stacked))

print(f"  ✓ Stacking RMSE: {stacked_rmse:.4f}")

models_optimized['stacked'] = {
    'base_models': models_optimized,
    'meta_model': meta_model,
    'use_log_transform': use_log_transform
}

# Visualization: Stacking Model
print("\n  Creating stacking model visualizations...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Step 8: Stacking Model', fontsize=16, fontweight='bold')

# Base model predictions correlation
axes[0].scatter(predictions_val_opt['xgb'], predictions_val_opt['lgb'], 
                alpha=0.5, color='steelblue', s=20)
corr_coef = np.corrcoef(predictions_val_opt['xgb'], predictions_val_opt['lgb'])[0, 1]
axes[0].set_title(f'Base Model Predictions Correlation\n(Correlation: {corr_coef:.3f})', fontweight='bold')
axes[0].set_xlabel('XGBoost Predictions')
axes[0].set_ylabel('LightGBM Predictions')
axes[0].grid(True, alpha=0.3)

# Stacking predictions vs Actual
axes[1].scatter(y_val, y_val_stacked, alpha=0.5, color='coral', s=20)
axes[1].plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 
             'r--', linewidth=2, label='Perfect Prediction')
axes[1].set_title(f'Stacked Model: Predictions vs Actual\n(RMSE: {stacked_rmse:.4f})', fontweight='bold')
axes[1].set_xlabel('Actual Risk Score')
axes[1].set_ylabel('Predicted Risk Score')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '08_stacking_model.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: analysis_outputs/08_stacking_model.png")

# =============================================================================
# 9. COMPARE ALL APPROACHES
# =============================================================================

print("\n9. Model Comparison:")
print("-" * 80)
print(f"{'Model':<20} {'RMSE':<15} {'MAE':<15} {'R²':<15}")
print("-" * 80)

for model_name, pred in predictions_val_opt.items():
    rmse = np.sqrt(mean_squared_error(y_val, pred))
    mae = mean_absolute_error(y_val, pred)
    r2 = r2_score(y_val, pred)
    print(f"{model_name.upper():<20} {rmse:<15.4f} {mae:<15.4f} {r2:<15.4f}")

print(f"{'STACKED':<20} {stacked_rmse:<15.4f} {mean_absolute_error(y_val, y_val_stacked):<15.4f} {r2_score(y_val, y_val_stacked):<15.4f}")
print("-" * 80)

# Select best model
all_results = {name: np.sqrt(mean_squared_error(y_val, pred)) for name, pred in predictions_val_opt.items()}
all_results['stacked'] = stacked_rmse
best_model_name = min(all_results, key=all_results.get)
best_rmse = all_results[best_model_name]

print(f"\n✓ Best model: {best_model_name.upper()} with RMSE: {best_rmse:.4f}")

# Visualization: Model Comparison
print("\n  Creating model comparison visualizations...")
model_results = {}
for model_name, pred in predictions_val_opt.items():
    rmse = np.sqrt(mean_squared_error(y_val, pred))
    mae = mean_absolute_error(y_val, pred)
    r2 = r2_score(y_val, pred)
    model_results[model_name.upper()] = {'RMSE': rmse, 'MAE': mae, 'R²': r2}

model_results['STACKED'] = {
    'RMSE': stacked_rmse,
    'MAE': mean_absolute_error(y_val, y_val_stacked),
    'R²': r2_score(y_val, y_val_stacked)
}

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Step 9: Model Comparison', fontsize=16, fontweight='bold')

models = list(model_results.keys())
rmse_values = [model_results[m]['RMSE'] for m in models]
mae_values = [model_results[m]['MAE'] for m in models]
r2_values = [model_results[m]['R²'] for m in models]

# RMSE comparison
colors = ['red' if m == best_model_name.upper() else 'steelblue' for m in models]
axes[0].bar(models, rmse_values, color=colors, alpha=0.7)
axes[0].set_title('RMSE Comparison', fontweight='bold')
axes[0].set_ylabel('RMSE')
axes[0].tick_params(axis='x', rotation=45)
axes[0].grid(True, alpha=0.3, axis='y')
for i, (model, rmse) in enumerate(zip(models, rmse_values)):
    axes[0].text(i, rmse, f'{rmse:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# MAE comparison
axes[1].bar(models, mae_values, color=colors, alpha=0.7)
axes[1].set_title('MAE Comparison', fontweight='bold')
axes[1].set_ylabel('MAE')
axes[1].tick_params(axis='x', rotation=45)
axes[1].grid(True, alpha=0.3, axis='y')
for i, (model, mae) in enumerate(zip(models, mae_values)):
    axes[1].text(i, mae, f'{mae:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# R² comparison
axes[2].bar(models, r2_values, color=colors, alpha=0.7)
axes[2].set_title('R² Score Comparison', fontweight='bold')
axes[2].set_ylabel('R² Score')
axes[2].tick_params(axis='x', rotation=45)
axes[2].grid(True, alpha=0.3, axis='y')
for i, (model, r2) in enumerate(zip(models, r2_values)):
    axes[2].text(i, r2, f'{r2:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '09_model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: analysis_outputs/09_model_comparison.png")

# =============================================================================
# 10. FEATURE IMPORTANCE ANALYSIS
# =============================================================================

print("\n10. Analyzing feature importance...")

# Get importance from XGBoost
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': xgb_opt.feature_importances_
}).sort_values('importance', ascending=False)

print("\n  Top 30 Features:")
print("  " + "-" * 60)
for i, row in feature_importance.head(30).iterrows():
    print(f"  {i+1:2d}. {row['feature']:40s} {row['importance']:>10.4f}")

# Save importance
feature_importance.to_csv(OUTPUT_DIR / 'feature_importance_advanced.csv', index=False)

# Visualization: Feature Importance
print("\n  Creating feature importance visualizations...")
fig, axes = plt.subplots(1, 2, figsize=(16, 8))
fig.suptitle('Step 10: Feature Importance Analysis', fontsize=16, fontweight='bold')

# Top 20 features
top_features = feature_importance.head(20)
axes[0].barh(range(len(top_features)), top_features['importance'].values, color='steelblue')
axes[0].set_yticks(range(len(top_features)))
axes[0].set_yticklabels([f[:40] for f in top_features['feature'].values], fontsize=9)
axes[0].set_title('Top 20 Most Important Features', fontweight='bold')
axes[0].set_xlabel('Importance')
axes[0].grid(True, alpha=0.3, axis='x')

# Importance distribution
axes[1].hist(feature_importance['importance'], bins=30, color='coral', edgecolor='black', alpha=0.7)
axes[1].axvline(feature_importance['importance'].mean(), color='red', linestyle='--', 
                label=f'Mean: {feature_importance["importance"].mean():.4f}', linewidth=2)
axes[1].axvline(feature_importance['importance'].median(), color='green', linestyle='--', 
                label=f'Median: {feature_importance["importance"].median():.4f}', linewidth=2)
axes[1].set_title('Feature Importance Distribution', fontweight='bold')
axes[1].set_xlabel('Importance')
axes[1].set_ylabel('Frequency')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '10_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: analysis_outputs/10_feature_importance.png")

# =============================================================================
# 11. GENERATE FINAL PREDICTIONS
# =============================================================================

print("\n11. Generating final predictions...")

# Prepare test features
X_test = test_features[feature_cols].fillna(0)

# Handle outliers and inf
for col in X_test.columns:
    if X_test[col].dtype in [np.int64, np.float64]:
        Q1, Q3 = X_test[col].quantile([0.01, 0.99])
        X_test[col] = X_test[col].clip(lower=Q1, upper=Q3)
X_test = X_test.replace([np.inf, -np.inf], 0).clip(-1000, 1000)

# Generate predictions with best model
if best_model_name == 'stacked':
    # Stacking predictions
    pred_xgb_test = xgb_opt.predict(X_test)
    pred_lgb_test = lgb_opt.predict(X_test)
    
    if use_log_transform:
        pred_xgb_test = np.expm1(pred_xgb_test)
        pred_lgb_test = np.expm1(pred_lgb_test)
    
    X_test_stacked = pd.DataFrame({
        'xgb_pred': pred_xgb_test,
        'lgb_pred': pred_lgb_test,
    })
    test_predictions = meta_model.predict(X_test_stacked)
else:
    pred = models_optimized[best_model_name].predict(X_test)
    if use_log_transform:
        test_predictions = np.expm1(pred)
    else:
        test_predictions = pred

# Create submission
submission = pd.DataFrame({
    'patient_id': test_features['patient_id'],
    'predicted_risk_score': test_predictions
})

# Post-processing: Ensure reasonable range
submission['predicted_risk_score'] = submission['predicted_risk_score'].clip(lower=0.1, upper=100)

# Save
submission_file = BASE_DIR / 'predictions.csv'
submission.to_csv(submission_file, index=False)

print(f"\n✓ Predictions saved to {submission_file}")
print(f"\n  Prediction statistics:")
print(f"    Mean: {submission['predicted_risk_score'].mean():.2f}")
print(f"    Median: {submission['predicted_risk_score'].median():.2f}")
print(f"    Std: {submission['predicted_risk_score'].std():.2f}")
print(f"    Min: {submission['predicted_risk_score'].min():.2f}")
print(f"    Max: {submission['predicted_risk_score'].max():.2f}")

# Visualization: Final Predictions
print("\n  Creating final predictions visualizations...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Step 11: Final Predictions Analysis', fontsize=16, fontweight='bold')

# Prediction distribution
axes[0, 0].hist(submission['predicted_risk_score'], bins=50, color='lightgreen', edgecolor='black', alpha=0.7)
axes[0, 0].axvline(submission['predicted_risk_score'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {submission["predicted_risk_score"].mean():.2f}', linewidth=2)
axes[0, 0].axvline(submission['predicted_risk_score'].median(), color='blue', linestyle='--', 
                   label=f'Median: {submission["predicted_risk_score"].median():.2f}', linewidth=2)
axes[0, 0].set_title('Predicted Risk Score Distribution', fontweight='bold')
axes[0, 0].set_xlabel('Predicted Risk Score')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Comparison with training distribution
axes[0, 1].hist(train_risk['risk_score'], bins=50, alpha=0.5, label='Training Data', 
                color='steelblue', edgecolor='black')
axes[0, 1].hist(submission['predicted_risk_score'], bins=50, alpha=0.5, label='Predictions', 
                color='coral', edgecolor='black')
axes[0, 1].set_title('Training vs Prediction Distribution', fontweight='bold')
axes[0, 1].set_xlabel('Risk Score')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Box plot comparison
box_data = [train_risk['risk_score'], submission['predicted_risk_score']]
bp = axes[1, 0].boxplot(box_data, labels=['Training', 'Predictions'], patch_artist=True)
bp['boxes'][0].set_facecolor('steelblue')
bp['boxes'][1].set_facecolor('coral')
axes[1, 0].set_title('Risk Score Distribution: Training vs Predictions', fontweight='bold')
axes[1, 0].set_ylabel('Risk Score')
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Statistics comparison
stats_comparison = pd.DataFrame({
    'Training': [
        train_risk['risk_score'].mean(),
        train_risk['risk_score'].median(),
        train_risk['risk_score'].std(),
        train_risk['risk_score'].min(),
        train_risk['risk_score'].max()
    ],
    'Predictions': [
        submission['predicted_risk_score'].mean(),
        submission['predicted_risk_score'].median(),
        submission['predicted_risk_score'].std(),
        submission['predicted_risk_score'].min(),
        submission['predicted_risk_score'].max()
    ]
}, index=['Mean', 'Median', 'Std', 'Min', 'Max'])

x = np.arange(len(stats_comparison.index))
width = 0.35
axes[1, 1].bar(x - width/2, stats_comparison['Training'], width, label='Training', color='steelblue', alpha=0.7)
axes[1, 1].bar(x + width/2, stats_comparison['Predictions'], width, label='Predictions', color='coral', alpha=0.7)
axes[1, 1].set_xlabel('Statistic')
axes[1, 1].set_ylabel('Value')
axes[1, 1].set_title('Statistical Comparison', fontweight='bold')
axes[1, 1].set_xticks(x)
axes[1, 1].set_xticklabels(stats_comparison.index)
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '11_final_predictions.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: analysis_outputs/11_final_predictions.png")

# Save models
for name, model in models_optimized.items():
    if name != 'stacked':
        with open(MODELS_DIR / f'{name}_advanced.pkl', 'wb') as f:
            pickle.dump(model, f)
    else:
        with open(MODELS_DIR / 'stacked_advanced.pkl', 'wb') as f:
            pickle.dump(model, f)

print("\n✓ Models saved!")

print("\n" + "="*80)
print("ADVANCED IMPROVEMENTS COMPLETE!")
print("="*80)
print(f"\nBest Model: {best_model_name.upper()}")
print(f"Best RMSE: {best_rmse:.4f}")
print(f"Predictions: {submission_file}")

