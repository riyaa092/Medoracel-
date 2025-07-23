import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

# Load the dataset
df = pd.read_csv('disease.csv')

# Replace NaN with None
df.fillna('None', inplace=True)

# Get unique symptoms from all columns
symptom_columns = df.columns[:-1]  # All columns except 'Disease'
all_symptoms = set()
for col in symptom_columns:
    all_symptoms.update(df[col].unique())
all_symptoms.discard('None')
all_symptoms = sorted(list(all_symptoms))  # Consistent order

# ✅ Print the total number of unique symptoms
print(f"Total Symptoms Used in Training: {len(all_symptoms)}")

# Prepare training data as binary vectors
X = []
for _, row in df.iterrows():
    symptoms = set(row[symptom_columns])
    symptoms.discard('None')
    x_row = [1 if symptom in symptoms else 0 for symptom in all_symptoms]
    X.append(x_row)

# Encode diseases
le_disease = LabelEncoder()
y = le_disease.fit_transform(df['Disease'])

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Save everything
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('symptom_encoder.pkl', 'wb') as f:
    pickle.dump(all_symptoms, f)  # This is now just a list, not LabelEncoder

with open('disease_encoder.pkl', 'wb') as f:
    pickle.dump(le_disease, f)

print("✅ Model trained and saved successfully.")
