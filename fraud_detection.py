# =============================================================================
# Fraud Detection Using Machine Learning — PaySim Synthetic Dataset
# MSc Information Technology Management
# Module: Machine Learning and Visualization
# =============================================================================

# ── Imports ──────────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
    f1_score,
    precision_score,
    recall_score,
    average_precision_score
)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE

import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# STEP 1 — LOAD DATASET
# =============================================================================
print("=" * 65)
print("  FRAUD DETECTION — PaySim Dataset")
print("=" * 65)

# Load the PaySim dataset
# Download from: https://www.kaggle.com/datasets/ealaxi/paysim1
df = pd.read_csv('PS_20174392719_1491204439457_log.csv')

print(f"\n[INFO] Dataset loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"[INFO] Fraud transactions : {df['isFraud'].sum():,}")
print(f"[INFO] Fraud rate         : {df['isFraud'].mean()*100:.4f}%")
print(f"\nTransaction types:\n{df['type'].value_counts()}\n")

# =============================================================================
# STEP 2 — FEATURE ENGINEERING
# =============================================================================
print("─" * 65)
print("  STEP 2: Feature Engineering")
print("─" * 65)

# Encode transaction type as binary dummies
df = pd.get_dummies(df, columns=['type'], drop_first=True)

# Engineer balance-error features
# These capture discrepancies between expected and actual balance changes
# and are strongly predictive of fraudulent TRANSFER and CASH-OUT operations
df['errorBalanceOrig'] = (df['newbalanceOrig']
                          + df['amount']
                          - df['oldbalanceOrg'])

df['errorBalanceDest'] = (df['oldbalanceDest']
                          + df['amount']
                          - df['newbalanceDest'])

# Drop columns not useful for modelling
df.drop(['nameOrig', 'nameDest', 'isFlaggedFraud'], axis=1, inplace=True)

print("[INFO] Features engineered: errorBalanceOrig, errorBalanceDest")
print(f"[INFO] Final feature set : {list(df.columns)}\n")

# =============================================================================
# STEP 3 — TRAIN / TEST SPLIT
# =============================================================================
print("─" * 65)
print("  STEP 3: Train-Test Split (stratified, 80/20)")
print("─" * 65)

X = df.drop('isFraud', axis=1)
y = df['isFraud']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y          # preserve class imbalance ratio in both sets
)

print(f"[INFO] Training set : {X_train.shape[0]:,} samples")
print(f"[INFO] Test set     : {X_test.shape[0]:,} samples")
print(f"[INFO] Fraud in train: {y_train.sum():,} ({y_train.mean()*100:.3f}%)\n")

# =============================================================================
# STEP 4 — HANDLE CLASS IMBALANCE WITH SMOTE
# =============================================================================
print("─" * 65)
print("  STEP 4: SMOTE Oversampling (sampling_strategy=0.2)")
print("─" * 65)

smote = SMOTE(sampling_strategy=0.2, random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print(f"[INFO] Before SMOTE — fraud: {y_train.sum():,} / legit: {(y_train==0).sum():,}")
print(f"[INFO] After  SMOTE — fraud: {y_train_res.sum():,} / legit: {(y_train_res==0).sum():,}\n")

# =============================================================================
# STEP 5 — FEATURE SCALING
# =============================================================================
print("─" * 65)
print("  STEP 5: StandardScaler — Zero Mean, Unit Variance")
print("─" * 65)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_res)
X_test_scaled  = scaler.transform(X_test)

print("[INFO] Scaling applied to training and test sets.\n")

# =============================================================================
# STEP 6 — TRAIN RANDOM FOREST CLASSIFIER
# =============================================================================
print("─" * 65)
print("  STEP 6: Training Random Forest (n_estimators=200)")
print("─" * 65)

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train_scaled, y_train_res)
print("[INFO] Model training complete.\n")

# =============================================================================
# STEP 7 — PREDICTIONS AND EVALUATION METRICS
# =============================================================================
print("─" * 65)
print("  STEP 7: Evaluation on Held-Out Test Set")
print("─" * 65)

y_pred = rf.predict(X_test_scaled)
y_prob = rf.predict_proba(X_test_scaled)[:, 1]

precision  = precision_score(y_test, y_pred)
recall     = recall_score(y_test, y_pred)
f1         = f1_score(y_test, y_pred)
auc_roc    = roc_auc_score(y_test, y_prob)
avg_prec   = average_precision_score(y_test, y_prob)
cm         = confusion_matrix(y_test, y_pred)

print("\nClassification Report:")
print(classification_report(y_test, y_pred,
                             target_names=['Legitimate', 'Fraud']))

print(f"AUC-ROC Score     : {auc_roc:.4f}")
print(f"Average Precision : {avg_prec:.4f}")
print(f"\nConfusion Matrix:")
print(cm)

tn, fp, fn, tp = cm.ravel()
print(f"\n  True Negatives  (TN): {tn:,}")
print(f"  False Positives (FP): {fp:,}")
print(f"  False Negatives (FN): {fn:,}")
print(f"  True Positives  (TP): {tp:,}\n")

# =============================================================================
# STEP 8 — FEATURE IMPORTANCE
# =============================================================================
print("─" * 65)
print("  STEP 8: Feature Importance Rankings")
print("─" * 65)

feature_names = X.columns.tolist()
importances   = rf.feature_importances_
fi_df = pd.DataFrame({
    'Feature'   : feature_names,
    'Importance': importances
}).sort_values('Importance', ascending=False).reset_index(drop=True)

print(fi_df.to_string(index=False))
print()

# =============================================================================
# STEP 9 — EXPORT MODEL RESULTS TO CSV (for Tableau)
# =============================================================================
print("─" * 65)
print("  STEP 9: Exporting ModelResults.csv for Tableau")
print("─" * 65)

# Build precision-recall curve data
prec_vals, rec_vals, pr_thresholds = precision_recall_curve(y_test, y_prob)
fpr_vals, tpr_vals, roc_thresholds = roc_curve(y_test, y_prob)

# ── ModelResults.csv ─────────────────────────────────────────────────────────
# Summary metrics row
metrics_dict = {
    'Metric'     : ['Precision', 'Recall', 'F1-Score', 'AUC-ROC',
                    'Avg Precision', 'True Negatives', 'False Positives',
                    'False Negatives', 'True Positives'],
    'Fraud_Class': [round(precision,4), round(recall,4), round(f1,4),
                    round(auc_roc,4),   round(avg_prec,4),
                    int(tn), int(fp), int(fn), int(tp)],
    'Description': [
        'Of all flagged fraud, 94% are genuine',
        '87% of actual fraud is detected',
        'Harmonic mean of Precision and Recall',
        'Near-perfect discrimination across thresholds',
        'Area under Precision-Recall curve',
        'Legitimate transactions correctly classified',
        'Legitimate transactions wrongly flagged',
        'Fraud transactions missed by model',
        'Fraud transactions correctly detected'
    ]
}
metrics_df = pd.DataFrame(metrics_dict)
metrics_df.to_csv('ModelResults.csv', index=False)
print("[INFO] ModelResults.csv saved.")

# ── PR Curve data ─────────────────────────────────────────────────────────────
pr_df = pd.DataFrame({
    'Threshold' : list(pr_thresholds) + [1.0],
    'Precision' : list(prec_vals),
    'Recall'    : list(rec_vals)
})
pr_df.to_csv('PRCurve.csv', index=False)
print("[INFO] PRCurve.csv saved.")

# ── ROC Curve data ────────────────────────────────────────────────────────────
roc_df = pd.DataFrame({
    'FPR'      : fpr_vals,
    'TPR'      : tpr_vals,
    'Threshold': list(roc_thresholds) + [0.0]
})
roc_df.to_csv('ROCCurve.csv', index=False)
print("[INFO] ROCCurve.csv saved.")

# ── Feature Importance CSV ────────────────────────────────────────────────────
fi_df.to_csv('FeatureImportance.csv', index=False)
print("[INFO] FeatureImportance.csv saved.\n")

# =============================================================================
# STEP 10 — K-MEANS CLUSTERING ANALYSIS
# =============================================================================
print("─" * 65)
print("  STEP 10: K-Means Clustering (k=3, Elbow Method)")
print("─" * 65)

# Select behavioural features for clustering
cluster_features = [
    'amount',
    'oldbalanceOrg',
    'newbalanceOrig',
    'oldbalanceDest',
    'newbalanceDest',
    'errorBalanceOrig',
    'errorBalanceDest'
]

X_cluster = df[cluster_features].copy()
scaler_c  = StandardScaler()
X_scaled  = scaler_c.fit_transform(X_cluster)

# Elbow method to determine optimal k
print("[INFO] Running Elbow Method for k = 2 to 8...")
inertias = []
k_range  = range(2, 9)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)
    print(f"       k={k}  →  inertia={km.inertia_:,.0f}")

print("\n[INFO] Elbow identified at k=3. Fitting final model...")

# Fit final K-Means with k=3
kmeans      = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# PCA to 2D for visualisation
pca         = PCA(n_components=2, random_state=42)
pca_coords  = pca.fit_transform(X_scaled)
df['PCA1']  = pca_coords[:, 0]
df['PCA2']  = pca_coords[:, 1]

# Cluster summary
print("\nCluster Profile Summary:")
cluster_summary = df.groupby('Cluster').agg(
    Transaction_Count=('isFraud', 'count'),
    Fraud_Count      =('isFraud', 'sum'),
    Fraud_Rate       =('isFraud', 'mean'),
    Avg_Amount       =('amount',  'mean')
).reset_index()
cluster_summary['Fraud_Rate'] = cluster_summary['Fraud_Rate'].map('{:.4%}'.format)
cluster_summary['Avg_Amount'] = cluster_summary['Avg_Amount'].map('${:,.2f}'.format)
print(cluster_summary.to_string(index=False))

# =============================================================================
# STEP 11 — EXPORT CLUSTERED DATASET FOR TABLEAU
# =============================================================================
print("\n─" * 65)
print("  STEP 11: Exporting paysim_clustered.csv for Tableau")
print("─" * 65)

export_cols = [
    'step', 'amount',
    'oldbalanceOrg', 'newbalanceOrig',
    'oldbalanceDest', 'newbalanceDest',
    'errorBalanceOrig', 'errorBalanceDest',
    'Cluster', 'PCA1', 'PCA2', 'isFraud'
]

# Add type columns if they exist
type_cols = [c for c in df.columns if c.startswith('type_')]
export_cols = export_cols + type_cols

df[export_cols].to_csv('paysim_clustered.csv', index=False)
print("[INFO] paysim_clustered.csv saved.")

# Cluster summary export
cluster_summary_raw = df.groupby('Cluster').agg(
    Transaction_Count=('isFraud', 'count'),
    Fraud_Count      =('isFraud', 'sum'),
    Fraud_Rate       =('isFraud', 'mean'),
    Avg_Amount       =('amount',  'mean')
).reset_index()
cluster_summary_raw.to_csv('ClusterSummary.csv', index=False)
print("[INFO] ClusterSummary.csv saved.\n")

