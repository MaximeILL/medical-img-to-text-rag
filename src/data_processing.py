import pandas as pd
import random
from config import METADATA_CSV, SKIN_METADATA_CSV

def process_metadata():
    # load metadata file (pad-ufes-20)
    metadata = pd.read_csv(METADATA_CSV)

    age_templates = [
        "The patient is a {age}-year-old",
        "A {age}-year-old patient",
        "Patient, aged {age},"
    ]

    region_templates = [
        "with a lesion on the {region}",
        "having a lesion on the {region}",
        "with a lesion located on the {region}"
    ]

    symptoms_templates = [
        "The lesion {symptoms}.",
        "It {symptoms}.",
        "This lesion {symptoms}."
    ]

    def reformulate_question(row):
        age = random.choice(age_templates).format(age=row['age'])
        region = random.choice(region_templates).format(region=row['region']) if row['region'] != ' ' else "with a lesion"
        symptoms = []
        if row['itch'] != ' ':
            symptoms.append("is itchy" if row['itch'] == 'yes' else "is not itchy")
        if row['grew'] != ' ':
            symptoms.append("has grown" if row['grew'] == 'yes' else "has not grown")
        if row['hurt'] != ' ':
            symptoms.append("hurts" if row['hurt'] == 'yes' else "does not hurt")
        if row['changed'] != ' ':
            symptoms.append("has changed in appearance" if row['changed'] == 'yes' else "has not changed in appearance")
        if row['bleed'] != ' ':
            symptoms.append("bleeds" if row['bleed'] == 'yes' else "does not bleed")
        if row['elevation'] != ' ':
            symptoms.append("is elevated" if row['elevation'] == 'yes' else "is not elevated")
        if row['biopsed'] != ' ':
            symptoms.append("has been biopsied" if row['biopsed'] == 'yes' else "has not been biopsied")

        symptoms_str = ", ".join(symptoms)
        symptoms_sentence = random.choice(symptoms_templates).format(symptoms=symptoms_str)
        return f"{age} {region}. {symptoms_sentence}"

    # metadata preprocessing
    def preprocess_metadata(metadata):

        columns_to_drop = ["patient_id", "lesion_id"]
        columns_in_metadata = metadata.columns
        columns_to_drop = [col for col in columns_to_drop if col in columns_in_metadata]
        metadata = metadata.drop(columns=columns_to_drop)

        metadata = metadata.fillna(' ')
        metadata = metadata.applymap(lambda x: x.lower() if isinstance(x, str) else x)

        metadata = metadata.replace({'true': 'yes', 'false': 'no'})
        metadata = metadata.replace({True: 'yes', False: 'no'})

    
        diagnostic_replacements = {
            "bcc": "basal cell carcinoma",
            "scc": "squamous cell carcinoma",
            "ack": "actinic keratosis",
            "sek": "seborrheic keratosis",
            "bod": "bowen’s disease",
            "mel": "melanoma",
            "nev": "nevus"
        }
        if 'diagnostic' in metadata.columns:
            metadata['diagnostic'] = metadata['diagnostic'].replace(diagnostic_replacements)

        required_columns = {'age', 'region', 'itch', 'grew', 'hurt', 'changed', 'bleed', 'elevation', 'biopsed'}
        if required_columns.issubset(metadata.columns):
            metadata['question'] = metadata.apply(reformulate_question, axis=1)
        else:
            print("Des colonnes nécessaires pour créer 'question' sont manquantes.")

        metadata = metadata.rename(columns={'img_id': 'image'})

        diagnostic_descriptions = {
            "basal cell carcinoma": "a common skin cancer that is typically not aggressive",
            "squamous cell carcinoma": "a type of skin cancer that can become more aggressive if untreated",
            "actinic keratosis": "a rough, scaly patch on the skin, considered precancerous",
            "seborrheic keratosis": "a benign skin growth that appears as a brown, black or pale growth",
            "bowen’s disease": "a form of squamous cell carcinoma that is confined to the outer layer of the skin",
            "melanoma": "the most serious type of skin cancer, which can spread rapidly if not treated",
            "nevus": "a common benign skin lesion known as a mole"
        }
        metadata['answer'] = metadata['diagnostic'].apply(
            lambda diag: f"{diag}, {diagnostic_descriptions.get(diag, 'no description available')}"
        )

        metadata['description'] = metadata.apply(
            lambda row: f"{row['question']} He has {row['answer']}", axis=1
        )

        # keep 'image' and 'description' columns
        metadata = metadata[['image', 'description']]

        return metadata

    processed_metadata = preprocess_metadata(metadata)

    # save formatted metadata
    processed_metadata.to_csv(SKIN_METADATA_CSV, index=False)
    print(f"Processed metadata saved to '{SKIN_METADATA_CSV}'.")

if __name__ == "__main__":
    process_metadata()
