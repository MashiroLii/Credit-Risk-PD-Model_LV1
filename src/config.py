# src/config.py

# Data source URL
DATA_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data'

# Original column names
COLUMN_NAMES = [
    'Status_of_existing_checking_account', 'Duration_in_month',
    'Credit_history', 'Purpose', 'Credit_amount', 'Savings_account',
    'Employment_since', 'Installment_rate', 'Personal_status',
    'Other_debtors', 'Residence_since', 'Property', 'Age',
    'Other_installment_plans', 'Housing', 'Existing_credits',
    'Job', 'Liable_people', 'Telephone', 'Foreign_worker', 'Credit_risk'
]

# Target variable
TARGET = 'Default'

# Feature list will be populated during preprocessing
FEATURES = []
