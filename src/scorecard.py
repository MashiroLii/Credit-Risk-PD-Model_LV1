# src/scorecard.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src import processing
from src import config

def probability_to_score(proba, min_score=300, max_score=850):
    """Converts a probability of default into a credit score."""
    score = max_score - (proba * (max_score - min_score))
    return score.astype(int)

# --- NEW: Quantile-based Grade Assignment ---
def assign_grades_by_quantile(scores):
    """
    Assigns rating grades based on score quantiles.
    This ensures a more stable distribution of samples across grades.
    """
    # Calculate the quantile boundaries from the scores
    # q_80 is the 80th percentile, q_60 is the 60th, and so on.
    q_80 = np.quantile(scores, 0.8) # Top 20% boundary
    q_60 = np.quantile(scores, 0.6)
    q_40 = np.quantile(scores, 0.4)
    q_20 = np.quantile(scores, 0.2)
    
    print("\n--- Grade Cutoffs based on Score Quantiles ---")
    print(f"Grade A (Top 20%): Score >= {q_80:.0f}")
    print(f"Grade B (20%-40%): Score >= {q_60:.0f}")
    print(f"Grade C (40%-60%): Score >= {q_40:.0f}")
    print(f"Grade D (60%-80%): Score >= {q_20:.0f}")
    print(f"Grade E (Bottom 20%): Score < {q_20:.0f}")

    grades = []
    for s in scores:
        if s >= q_80:
            grades.append('A')
        elif s >= q_60:
            grades.append('B')
        elif s >= q_40:
            grades.append('C')
        elif s >= q_20:
            grades.append('D')
        else:
            grades.append('E')
    return grades

def run_scorecard_generation(use_quantile_grading=True):
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
    print("\n--- Generating predictions and scores ---")
    y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]
    
    # 4. Convert probabilities to scores
    scores = probability_to_score(y_pred_proba)
    print("✓ Scores generated successfully.")
    
    # 5. Assign grades based on scores (using the new method)
    if use_quantile_grading:
        grades = assign_grades_by_quantile(scores)
    else: # Keep the old method for comparison if needed
        grades = [assign_grades_by_score(s) for s in scores] # Assuming old function is named assign_grades_by_score
    print("\n✓ Grades assigned successfully.")
    
    # 6. Create and display the scorecard analysis
    print("\n" + "="*70)
    print("                     CREDIT SCORECARD ANALYSIS (Quantile-based)")
    print("="*70)

    analysis_df = pd.DataFrame({
        'Score': scores,
        'Grade': grades,
        'Actual_Default': y_test.values
    })

    grade_stats = analysis_df.groupby('Grade').agg(
        Min_Score=('Score', 'min'),
        Max_Score=('Score', 'max'),
        Count=('Score', 'count'),
        Default_Count=('Actual_Default', 'sum'),
        Default_Rate=('Actual_Default', 'mean')
    ).reindex(['A', 'B', 'C', 'D', 'E'])

    grade_stats['Default_Rate'] = (grade_stats['Default_Rate'] * 100).map('{:.2f}%'.format)

    print(grade_stats)
    print("\n✓ Analysis complete. Check if the default rate is now monotonic.")

if __name__ == "__main__":
    run_scorecard_generation()
