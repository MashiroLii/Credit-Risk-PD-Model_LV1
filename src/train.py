# src/train.py
import pandas as pd
import numpy as np

# Model Algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Model Evaluation
from sklearn.metrics import roc_auc_score, roc_curve

# --- Important Note on Imports ---
# The following imports assume that you are running this script from a local
# environment where 'src' is a recognized package. When running in a cloud
# notebook like Colab after cloning the repo, you might need to adjust the system path.
# For now, we write the code as it's meant to be structured.
from src import processing
from src import config

def run_training():
    """
    Loads and preprocesses data, then trains and benchmarks multiple models.
    """
    # 1. Load and preprocess data using our processing script
    # Note: The preprocess_data function was modified to return the feature list
    df = processing.load_data()
    X_train_scaled, X_test_scaled, y_train, y_test, features = processing.preprocess_data(df)
    
    # Update the config with the actual feature names
    config.FEATURES = features
    
    print("\n" + "="*50)
    print("          STARTING MODEL BENCHMARKING")
    print("="*50 + "\n")

    # 2. Define all models to be benchmarked
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }

    # 3. Train and evaluate each model
    results = []

    for name, model in models.items():
        print(f"--- Training {name} ---")
        
        # Train the model
        model.fit(X_train_scaled, y_train)
        
        # Predict probabilities on the test set
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        auc = roc_auc_score(y_test, y_pred_proba)
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        ks_stat = max(tpr - fpr)
        
        print(f"✓ {name} | AUC: {auc:.4f} | KS: {ks_stat:.4f}")
        
        results.append({
            "Model": name,
            "AUC": auc,
            "KS": ks_stat
        })
        print("-"*(20 + len(name)) + "\n")

    # 4. Summarize and display results
    print("="*50)
    print("            BENCHMARKING SUMMARY")
    print("="*50)
    
    summary_df = pd.DataFrame(results).sort_values(by='AUC', ascending=False).reset_index(drop=True)
    print(summary_df)
    print("\n✓ Benchmarking complete.")


# This block allows the script to be run directly.
if __name__ == "__main__":
    run_training()
