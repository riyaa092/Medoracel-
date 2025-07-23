import pickle
import pandas as pd
import csv

# Load and preview files
desc_df = pd.read_csv("symptom_Description.csv", quotechar='"')
print("Description file columns:", desc_df.columns.tolist())
severity_df = pd.read_csv("Symptom-severity.csv")
print("Severity file columns:", severity_df.columns.tolist())

# ✅ Clean column names
desc_df.columns = desc_df.columns.str.strip().str.lower()
severity_df.columns = severity_df.columns.str.strip().str.lower()

# Load the trained symptom list (used during model training)
with open("symptom_encoder.pkl", "rb") as f:
    symptom_list = pickle.load(f)

print("✅ Total symptoms used during training:", len(symptom_list))
print("\n🩺 Symptom list used during training:\n")
for i, s in enumerate(symptom_list, 1):
    print(f"{i}. {s}")


# ✅ New safe version for symptom descriptions
def load_symptom_descriptions(file_path='symptom_Description.csv'):
    description_map = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            disease = row['Disease'].strip().lower()
            description = row['Description'].strip()
            description_map[disease] = description
    return description_map

def get_disease_description(disease_name):
    descriptions = load_symptom_descriptions()
    return descriptions.get(disease_name.strip().lower(), "Description not available.")


# 💡 Your Original Function (untouched except cleaned column names above)
def get_symptom_details(symptom_input_list):
    """
    Returns a list of dictionaries containing description and severity of each symptom
    """
    details = []
    for symptom in symptom_input_list:
        # Matching symptom in CSV (case insensitive)
        desc_row = desc_df[desc_df["disease"].str.lower() == symptom.lower()]
        sev_row = severity_df[severity_df["symptom"].str.lower() == symptom.lower()]

        description = desc_row["description"].values[0] if not desc_row.empty else "No description available."
        severity = int(sev_row["severity"].values[0]) if not sev_row.empty else 0

        details.append({
            "symptom": symptom,
            "description": description,
            "severity": severity
        })

    return details
