## Overview

This project implements an advanced machine learning pipeline for predicting patient risk scores in a healthcare setting. The solution employs ensemble methods, feature engineering, and hyperparameter optimization to maximize prediction accuracy.

---

## 1. Overall Approach and Data Architecture

### Problem Statement
Predict patient risk scores based on historical healthcare data including patient demographics, visit patterns, diagnoses, and care records.

### Data Architecture

The dataset consists of five interconnected tables:

#### Training Data (`pcms_hackathon_data/train/`)
- **`patient.csv`**: Core patient demographics and identifiers
  - `patient_id`: Unique patient identifier
  - `age`: Patient age
  - `hot_spotter_identified_at`: Date when patient was identified as a high-utilizer
  - `hot_spotter_readmission_flag`: Boolean flag for readmission patterns
  - `hot_spotter_chronic_flag`: Boolean flag for chronic conditions

- **`visit.csv`**: Patient visit records
  - `visit_id`: Unique visit identifier
  - `patient_id`: Foreign key to patient table
  - `visit_type`: Type of visit (ER, INPATIENT, OUTPATIENT, etc.)
  - `visit_start_dt`: Visit start date
  - `visit_end_dt`: Visit end date
  - `readmsn_ind`: Readmission indicator

- **`diagnosis.csv`**: Patient diagnoses
  - `diagnosis_id`: Unique diagnosis identifier
  - `patient_id`: Foreign key to patient table
  - `condition_name`: Name of the medical condition
  - `is_chronic`: Boolean indicating chronic condition

- **`care.csv`**: Care management records
  - `care_id`: Unique care record identifier
  - `patient_id`: Foreign key to patient table
  - `care_gap_ind`: Care gap indicator

- **`risk.csv`**: Target variable
  - `patient_id`: Foreign key to patient table
  - `risk_score`: Target variable (continuous)

#### Test Data (`pcms_hackathon_data/test/`)
- Same structure as training data, excluding `risk.csv` (to be predicted)

### Data Flow Pipeline

```
Raw CSV Files
    ↓
Data Loading & Validation
    ↓
Feature Engineering (Patient, Visit, Diagnosis, Care)
    ↓
Data Preprocessing (Outlier Handling, Transformation)
    ↓
Train/Validation Split (80/20)
    ↓
Model Training (XGBoost, LightGBM)
    ↓
Ensemble Stacking
    ↓
Prediction & Post-processing
    ↓
Output: predictions.csv
```

### Output Structure

- **`predictions.csv`**: Final predictions for test patients
  - `patient_id`: Patient identifier
  - `predicted_risk_score`: Predicted risk score
  
- **`models/`**: Trained model artifacts (`.pkl` files)
  - `xgb_advanced.pkl`: XGBoost model
  - `lgb_advanced.pkl`: LightGBM model
  - `stacked_advanced.pkl`: Stacked ensemble model

- **`analysis_outputs/`**: Analysis artifacts
  - `feature_importance_advanced.csv`: Feature importance rankings

---

## 2. Feature Selection Logic and Assumptions

### Feature Engineering Strategy

Features are engineered from four main data sources:

#### A. Patient-Level Features
- **Demographics**: Age (continuous and binned)
- **Hot Spotter Indicators**: Boolean flags for high-utilization patients
  - Assumption: Patients identified as "hot spotters" have elevated risk
- **Age Bins**: Categorical age groups (18-35, 36-50, 51-65, 65+)
  - Assumption: Risk patterns vary by age cohort

#### B. Visit Aggregation Features
- **Volume Metrics**:
  - `total_visits`: Total number of visits per patient
  - `readmission_count`: Total readmissions
  - `visits_last_{30,60,90}_days`: Recent visit frequency
  - Assumption: Higher visit frequency correlates with higher risk

- **Visit Type Breakdown**:
  - Counts by visit type (ER, INPATIENT, OUTPATIENT, etc.)
  - Visit type ratios (proportion of each type)
  - Assumption: Emergency and inpatient visits indicate higher acuity

- **Temporal Features**:
  - `avg_visit_duration`: Average length of stay
  - `max_visit_duration`: Longest single visit
  - `days_since_visit`: Recency of last visit
  - Assumption: Recent and longer visits suggest ongoing health issues

- **Emergency Visit Metrics**:
  - `emergency_visit_count`: Total emergency visits
  - `emergency_visit_rate`: Proportion of emergency visits
  - Assumption: Emergency utilization is a strong risk indicator

#### C. Diagnosis Features
- **Condition Counts**:
  - `total_diagnoses`: Total number of diagnoses
  - `chronic_count`: Number of chronic conditions
  - Assumption: More conditions = higher complexity = higher risk

- **Specific Conditions**:
  - `has_cancer`: Binary indicator
  - `has_diabetes`: Binary indicator
  - `has_hypertension`: Binary indicator
  - Assumption: Specific conditions have different risk weights

- **Severity Score**:
  - `chronic_severity_score`: Weighted combination
    - Cancer: 5 points
    - Diabetes: 2 points
    - Hypertension: 1 point
  - Assumption: Conditions can be weighted by clinical severity

#### D. Care Management Features
- **Care Gap Metrics**:
  - `total_care_records`: Total care management records
  - `care_gaps`: Number of care gaps identified
  - `care_gap_rate`: Proportion of care gaps
  - Assumption: Care gaps indicate unmet needs = higher risk

#### E. Interaction Features (Domain-Driven)
- **High-Risk Combinations**:
  - `cancer_x_inpatient`: Cancer × Inpatient visits
  - `cancer_x_readmission`: Cancer × Readmissions
  - `chronic_x_emergency`: Chronic severity × Emergency visits
  - `gaps_x_cancer`: Care gaps × Cancer (weighted by 10)
  - Assumption: Interactions capture synergistic risk effects

- **Age-Condition Interactions**:
  - `elderly_cancer`: Age > 65 AND Cancer
  - `elderly_emergency`: Age > 65 AND Emergency visits
  - Assumption: Elderly patients with specific conditions have amplified risk

- **Composite Flags**:
  - `high_risk_flag`: Binary flag combining multiple high-risk indicators
  - Assumption: Multiple risk factors compound

- **Risk Components**:
  - `visit_risk_component`: Weighted emergency visits and readmissions
  - `chronic_risk_component`: Weighted chronic severity
  - `care_risk_component`: Weighted care gap rate
  - Assumption: Different risk domains contribute additively

### Feature Selection Assumptions

1. **Missing Value Handling**: All missing numeric features are imputed to 0
   - Rationale: Absence of records (e.g., no visits) is meaningful and indicates lower risk

2. **Outlier Treatment**: Features are clipped at 1st and 99th percentiles
   - Rationale: Extreme outliers likely represent data errors rather than true risk signals
   - Exception: Target variable outliers are preserved (real high-risk patients)

3. **Temporal Assumptions**:
   - Reference date: 2025-03-01 (used for recency calculations)
   - Visit durations are capped at 365 days (to handle data errors)
   - Days since visit capped at 5 years

4. **Clinical Assumptions**:
   - Cancer is weighted higher than diabetes/hypertension
   - Emergency and inpatient visits are stronger risk signals than outpatient
   - Care gaps in the presence of cancer amplify risk significantly

---

## 3. Model Architecture and Parameter Tuning

### Model Stack

#### Base Models

1. **XGBoost (Gradient Boosting)**
   - Algorithm: Gradient boosted decision trees
   - Objective: `reg:squarederror` (mean squared error)
   - Evaluation Metric: RMSE

2. **LightGBM (Gradient Boosting)**
   - Algorithm: Gradient boosted decision trees with leaf-wise growth
   - Objective: `regression`
   - Evaluation Metric: RMSE
   - Early stopping: 100 rounds without improvement

#### Ensemble Method: Stacking

- **Meta-Model**: XGBoost (shallow tree)
- **Base Model Predictions**: Used as features for meta-model
- **Strategy**: Two-level ensemble
  1. Level 1: XGBoost and LightGBM predictions
  2. Level 2: Meta-model combines Level 1 predictions

### Hyperparameter Optimization

#### Automated Optimization (Optuna)

When Optuna is available, XGBoost hyperparameters are optimized using Bayesian optimization:

**Search Space**:
- `max_depth`: [4, 10] (tree depth)
- `learning_rate`: [0.01, 0.2] (log scale)
- `n_estimators`: [200, 1500] (number of trees)
- `subsample`: [0.6, 1.0] (row sampling)
- `colsample_bytree`: [0.6, 1.0] (column sampling)
- `min_child_weight`: [1, 10] (minimum samples in leaf)
- `reg_alpha`: [0.01, 2.0] (L1 regularization, log scale)
- `reg_lambda`: [0.01, 2.0] (L2 regularization, log scale)

**Optimization Process**:
- Method: Tree-structured Parzen Estimator (TPE)
- Cross-Validation: 3-fold K-Fold
- Trials: 20 iterations
- Metric: Negative RMSE (minimized)

#### Default Parameters (Fallback)

If Optuna is unavailable, optimized defaults are used:

**XGBoost Defaults**:
```python
{
    'max_depth': 7,
    'learning_rate': 0.03,
    'n_estimators': 1000,
    'subsample': 0.85,
    'colsample_bytree': 0.85,
    'min_child_weight': 2,
    'reg_alpha': 0.2,
    'reg_lambda': 2,
    'random_state': 42
}
```

**LightGBM Defaults**:
```python
{
    'max_depth': 7,
    'learning_rate': 0.03,
    'n_estimators': 1000,
    'subsample': 0.85,
    'colsample_bytree': 0.85,
    'min_child_samples': 2,
    'reg_alpha': 0.2,
    'reg_lambda': 2,
    'random_state': 42
}
```

**Meta-Model (Stacking)**:
```python
{
    'max_depth': 3,
    'learning_rate': 0.1,
    'n_estimators': 100,
    'random_state': 42
}
```

### Target Transformation

**Adaptive Log Transformation**:
- Algorithm compares skewness of original vs. log1p-transformed target
- If log transformation reduces skewness, it is applied
- Transform is reversed after prediction (using `expm1`)

**Rationale**: Risk scores often follow a skewed distribution; log transformation helps normalize the target for regression models.

### Model Selection

The best model is selected based on **validation RMSE**:
- All base models are evaluated
- Stacked ensemble is evaluated
- Model with lowest RMSE is selected for final predictions

---

## 4. Setup and Execution Steps

### Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

### Installation

1. **Clone or navigate to the project directory**:
   ```bash
   cd /path/to/hilabs_hack
   ```

2. **Create a virtual environment (recommended)**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   This will install:
   - pandas (>=1.3.0)
   - numpy (>=1.21.0)
   - scikit-learn (>=1.0.0)
   - xgboost (>=1.5.0)
   - lightgbm (>=3.3.0)
   - optuna (>=3.0.0) - for hyperparameter optimization
   - matplotlib, seaborn (for visualization)
   - jupyter, ipykernel (for notebook support)

### Data Preparation

Ensure the data directory structure exists:
```
hilabs_hack/
├── pcms_hackathon_data/
│   ├── train/
│   │   ├── patient.csv
│   │   ├── visit.csv
│   │   ├── diagnosis.csv
│   │   ├── care.csv
│   │   └── risk.csv
│   └── test/
│       ├── patient.csv
│       ├── visit.csv
│       ├── diagnosis.csv
│       └── care.csv
```

### Execution

#### Option 1: Python Script
```bash
python submission.py
```

#### Option 2: Jupyter Notebook
1. Start Jupyter:
   ```bash
   jupyter notebook
   ```
2. Open `submission.ipynb`
3. Run all cells (Cell → Run All)

### Expected Output

The script will:

1. **Load and validate data** from CSV files
2. **Engineer features** from all data sources (~50+ features)
3. **Preprocess data** (outlier handling, transformation)
4. **Optimize hyperparameters** (if Optuna available, ~2-5 minutes)
5. **Train models**:
   - XGBoost (~30-60 seconds)
   - LightGBM (~30-60 seconds)
   - Stacked ensemble (~10 seconds)
6. **Evaluate models** and select best performer
7. **Generate predictions** for test set
8. **Save outputs**:
   - `predictions.csv` - Final predictions
   - `models/*.pkl` - Trained models
   - `analysis_outputs/feature_importance_advanced.csv` - Feature rankings

### Output Files

- **`predictions.csv`**: Main submission file
  - Format: `patient_id`, `predicted_risk_score`
  
- **`models/`**: Saved model artifacts
  - Can be loaded later for inference: `pickle.load(open('models/xgb_advanced.pkl', 'rb'))`
  
- **`analysis_outputs/`**: Analysis artifacts
  - Feature importance rankings for model interpretability

### Performance Notes

- **Total Runtime**: ~5-10 minutes (depending on hardware and Optuna optimization)
- **Memory Requirements**: ~2-4 GB RAM
- **CPU Usage**: Multi-threaded (utilizes all available cores)

### Troubleshooting

**Issue**: `FileNotFoundError` for data files
- **Solution**: Ensure `pcms_hackathon_data/` directory exists with train/test subdirectories

**Issue**: `Optuna not available`
- **Solution**: Install with `pip install optuna`. The script will fall back to default parameters if unavailable.

**Issue**: Memory errors
- **Solution**: Reduce `n_estimators` in model parameters or process data in chunks

**Issue**: Import errors
- **Solution**: Verify all packages are installed: `pip install -r requirements.txt`

---

## Model Performance

The pipeline evaluates multiple metrics:
- **RMSE** (Root Mean Squared Error): Primary metric for model selection
- **MAE** (Mean Absolute Error): Average prediction error
- **R²** (Coefficient of Determination): Proportion of variance explained

The best model (typically the stacked ensemble) is selected based on validation RMSE.

---

## File Structure

```
hilabs_hack/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── submission.ipynb
├── submission.py         # Main Python script
├── pcms_hackathon_data/                # Input data
│   ├── train/
│   └── test/
├── predictions.csv                     # Final predictions (generated)
├── models/                             # Saved models (generated)
│   ├── xgb_advanced.pkl
│   ├── lgb_advanced.pkl
│   └── stacked_advanced.pkl
└── analysis_outputs/                   # Analysis artifacts (generated)
    └── feature_importance_advanced.csv
```

---

## License and Acknowledgments

This project was developed for the HiLabs Hackathon 2025.

For questions or issues, please refer to the code comments or documentation.

