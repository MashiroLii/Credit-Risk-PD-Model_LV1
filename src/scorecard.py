# src/scorecard.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from sklearn.ensemble import RandomForestClassifier
from src import processing

def plot_grade_analysis(grade_stats, save_path="scorecard_grade_analysis.png"):
    """
    Plots a bar chart showing the default rate for each credit grade.
    NOW EXPECTS 'Default_Rate' to be a float.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 7))

    # --- THIS IS THE FIX ---
    # The 'Default_Rate' column is now a direct float, no string conversion needed.
    default_rates = grade_stats['Default_Rate']

    bars = ax.bar(grade_stats.index, default_rates, color='skyblue')
    
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_title('Default Rate by Credit Grade', fontsize=16)
    ax.set_ylabel('Default Rate')
    ax.set_xlabel('Credit Grade')
    ax.set_ylim(0, max(default_rates) * 1.2) # Set y-limit slightly above the max rate

    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f'{yval:.1%}', ha='center', va='bottom')

    plt.savefig(save_path)
    plt.close()
    
    print(f"\n✓ Grade analysis chart saved to '{save_path}'")

def probability_to_score(proba, min_score=300, max_score=850):
    """Converts a probability of default into a credit score."""
    score = max_score - (proba * (max_score - min_score))
    return score.astype(int)

def assign_grades_by_quantile(scores):
    """Assigns rating grades based on score quantiles."""
    q_80 = np.quantile(scores, 0.8)
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
        if s >= q_80: grades.append('A')
        elif s >= q_60: grades.append('B')
        elif s >= q_40: grades.append('C')
        elif s >= q_20: grades.append('D')
        else: grades.append('E')
    return grades

def run_scorecard_generation():
    """
    Trains the best model, generates scores, assigns grades,
    produces a final analysis report, and returns the report.
    """
    print("\n" + "="*50)
    print("        GENERATING CREDIT SCORECARD")
    print("="*50 + "\n")

    df = processing.load_data()
    X_train_scaled, X_test_scaled, y_train, y_test, features = processing.preprocess_data(df)
    
    print("--- Training the best performing model: Random Forest ---")
    best_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    best_model.fit(X_train_scaled, y_train)
    print("✓ Best model trained successfully.")

    print("\n--- Generating predictions and scores ---")
    y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]
    
    scores = probability_to_score(y_pred_proba)
    print("✓ Scores generated successfully.")
    
    grades = assign_grades_by_quantile(scores)
    print("\n✓ Grades assigned successfully.")
    
    print("\n" + "="*70)
    print("                     CREDIT SCORECARD ANALYSIS (Quantile-based)")
    print("="*70)

    analysis_df = pd.DataFrame({'Score': scores, 'Grade': grades, 'Actual_Default': y_test.values})
    grade_stats = analysis_df.groupby('Grade').agg(
        Min_Score=('Score', 'min'), Max_Score=('Score', 'max'),
        Count=('Score', 'count'), Default_Count=('Actual_Default', 'sum'),
        Default_Rate=('Actual_Default', 'mean')
    ).reindex(['A', 'B', 'C', 'D', 'E'])

    # Keep the original float rate for plotting, then format for display
    grade_stats_for_plot = grade_stats.copy()
    grade_stats['Default_Rate'] = (grade_stats['Default_Rate'] * 100).map('{:.2f}%'.format)
    
    print(grade_stats)
    print("\n✓ Analysis complete.")
    
    # Return the stats dataframe for plotting
    return grade_stats_for_plot


if __name__ == "__main__":
    stats = run_scorecard_generation()
    plot_grade_analysis(stats)
