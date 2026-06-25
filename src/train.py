# src/train.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, roc_curve

from src import processing
from src import config

def plot_roc_curves(results, save_path="model_benchmark_roc_curves.png"):
    """Plots ROC curves for all benchmarked models on a single graph."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(10, 8))

    for res in results:
        plt.plot(res['fpr'], res['tpr'], label=f"{res['Model']} (AUC = {res['AUC']:.4f})")

    plt.plot([0, 1], [0, 1], 'r--', label='Random Chance')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison for PD Models', fontsize=16)
    plt.legend(loc='lower right')
    plt.grid(True)
    
    plt.savefig(save_path)
    print(f"\n✓ ROC curve comparison chart saved to '{save_path}'")

def run_training():
    """
    Loads and preprocesses data, trains and benchmarks models.
    NOW RETURNS the results dictionary for external use.
    """
    df = processing.load_data()
    X_train_scaled, X_test_scaled, y_train, y_test, features = processing.preprocess_data(df)
    
    config.FEATURES = features
    
    print("\n" + "="*50)
    print("          STARTING MODEL BENCHMARKING")
    print("="*50 + "\n")

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }

    results = []
    for name, model in models.items():
        print(f"--- Training {name} ---")
        model.fit(X_train_scaled, y_train)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        auc = roc_auc_score(y_test, y_pred_proba)
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        ks_stat = max(tpr - fpr)
        
        print(f"✓ {name} | AUC: {auc:.4f} | KS: {ks_stat:.4f}")
        
        results.append({
            "Model": name, "AUC": auc, "KS": ks_stat, "fpr": fpr, "tpr": tpr
        })
        print("-"*(20 + len(name)) + "\n")

    print("="*50)
    print("            BENCHMARKING SUMMARY")
    print("="*50)
    
    summary_df = pd.DataFrame(results)[['Model', 'AUC', 'KS']].sort_values(by='AUC', ascending=False).reset_index(drop=True)
    print(summary_df)
    
    print("\n✓ Benchmarking complete.")
    
    # *** KEY CHANGE: RETURN THE RESULTS ***
    return results

if __name__ == "__main__":
    # This block is for running the script directly from the command line
    training_results = run_training()
    plot_roc_curves(training_results)
