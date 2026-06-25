# src/processing.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Because we are creating files on GitHub, we can't directly import from src.config.
# We will pass config as an argument or define it inside the functions.
# For simplicity now, let's redefine necessary configs here.
# We will resolve this when running the code in a proper environment.

DATA_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data'
COLUMN_NAMES = [
    'Status_of_existing_checking_account', 'Duration_in_month',
    'Credit_history', 'Purpose', 'Credit_amount', 'Savings_account',
    'Employment_since', 'Installment_rate', 'Personal_status',
    'Other_debtors', 'Residence_since', 'Property', 'Age',
    'Other_installment_plans', 'Housing', 'Existing_credits',
    'Job', 'Liable_people', 'Telephone', 'Foreign_worker', 'Credit_risk'
]
TARGET = 'Default'


def load_data():
    """Loads data from the URL."""
    df = pd.read_csv(DATA_URL, sep=' ', header=None, names=COLUMN_NAMES)
    print("✓ Data loaded successfully.")
    return df

def preprocess_data(df):
    """Performs all data preprocessing steps."""
    df[TARGET] = (df['Credit_risk'] == 2).astype(int)
    df = df.drop('Credit_risk', axis=1)
    
    categorical_cols = df.select_dtypes(include='object').columns.tolist()
    for col in categorical_cols:
        df[col], _ = pd.factorize(df[col])
    
    print("✓ Data preprocessing complete.")
    
    X = df.drop(TARGET, axis=1)
    y = df[TARGET]
    
    features = X.columns.tolist() # We'll need to pass this to the train script later

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"✓ Data split complete. Train set: {len(X_train)} rows | Test set: {len(X_test)} rows")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("✓ Feature scaling complete.")
    
    # We need to return the feature list to be used by the training script
    return X_train_scaled, X_test_scaled, y_train, y_test, features
