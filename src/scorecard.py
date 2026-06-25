# src/scorecard.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Assuming this script is run within the project structure
from src import processing
from src import config

def probability_to_score(proba, min_score=300, max_score=850):
    """
    Converts a probability of default into a credit score.
    A higher probability results in a lower score.
    """
    # Using a linear transformation.
    # We map proba=0 to max_score and proba=1 to min_score.
    score = max_score - (proba * (max_score - min_score))
    return score.astype(int)

def assign_grades(score):
    """Assigns a rating grade based on the score."""
    if score >= 800:
        return 'A'
    elif score >= 700:
        return 'B'
    elif score >= 600:
        return 'C'
    elif score >= 500:
        return 'D'
    else:
        return 'E'

def run_scorecard_generation():
    """
    Trains the best model, generates scores, assigns grades,
    and produces a final analysis report.
    """
    print("\n" + "="*50)
    print("        GENERATING CREDIT SCORECARD")
    print("="*50 + "\n")

    # 1. Load and preprocess data
    df = processing.load_data()
    X_train_scaled, X_test_scaled, y_train, y_test, features = processing.preprocess_data(df)
    
    # 2. Train the chosen best model (Random Forest)
    print("--- Training the best performing model: Random Forest ---")
    best_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    best_model.fit(X_train_scaled, y_train)
    print("✓ Best model trained successfully.")

    # 3. Predict default probabilities on the test set
    print("--- Generating predictions and scores ---")
    y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]
    
    # 4. Convert probabilities to scores
    scores = probability_to_score(y_pred_proba)
    print("✓ Scores generated successfully.")
    
    # 5. Assign grades based on scores
    grades = [assign_grades(s) for s in scores]
    print("✓ Grades assigned successfully.")
    
    # 6. Create and display the scorecard analysis
    print("\n" + "="*70)
    print("                     CREDIT SCORECARD ANALYSIS")
    print("="*70)

    analysis_df = pd.DataFrame({
        'Score': scores,
        'Grade': grades,
        'Actual_Default': y_test.values  # Use .values to avoid index mismatch
    })

    # Group by grade and calculate statistics
    grade_stats = analysis_df.groupby('Grade').agg(
        Min_Score=('Score', 'min'),
        Max_Score=('Score', 'max'),
        Count=('Score', 'count'),
        Default_Count=('Actual_Default', 'sum'),
        Default_Rate=('Actual_Default', 'mean')
    ).reindex(['A', 'B', 'C', 'D', 'E']) # Ensure grades are in order

    # Format the default rate as a percentage
    grade_stats['Default_Rate'] = (grade_stats['Default_Rate'] * 100).map('{:.2f}%'.format)

    print(grade_stats)
    print("\n✓ Analysis complete. The table above shows the risk distribution across grades.")
    

if __name__ == "__main__":
    run_scorecard_generation()
