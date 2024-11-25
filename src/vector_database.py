# src/vector_database.py

import os
import pandas as pd
from PIL import Image
from datasets import load_from_disk
import torch
from transformers import CLIPProcessor, CLIPModel
import kdbai_client as kdbai
import time
from tqdm import tqdm
from config import COMBINED_DATASET_DIR, KDBAI_ENDPOINT, KDBAI_API_KEY

def data_to_embedding(data_in, data_type, clip_model, clip_processor, device):
    if data_type == 'image':
        inputs = clip_processor(images=data_in, return_tensors="pt").to(device)
        with torch.no_grad():
            embedding = clip_model.get_image_features(**inputs)
        vec = embedding.cpu().numpy().tolist()[0]
    elif data_type == 'text':
        inputs = clip_processor(text=data_in, return_tensors="pt", truncation=True, max_length=77).to(device)
        with torch.no_grad():
            embedding = clip_model.get_text_features(**inputs)
        vec = embedding.cpu().numpy().tolist()[0]
    return vec

def process_dataset_for_embeddings(dataset, df, clip_model, clip_processor, device):
    for idx, row in enumerate(dataset):
        image = row['image']
        description = row['description']

        # convert img
        if isinstance(image, dict) and 'path' in image:
            image = Image.open(image['path'])
        elif isinstance(image, bytes):
            image = Image.open(io.BytesIO(image))
        elif isinstance(image, str):
            image = Image.open(image)
        elif isinstance(image, Image.Image):
            pass
        else:
            print(f"Type d'image non supporté: {type(image)}")
            continue

        image_embedding = data_to_embedding(image, 'image', clip_model, clip_processor, device)
        text_embedding = data_to_embedding(description, 'text', clip_model, clip_processor, device)

        temp_image_path = f'data/temp_image_{idx}.jpg'
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        image.save(temp_image_path)

        temp_text_path = f'data/temp_text_{idx}.txt'
        with open(temp_text_path, 'w') as f:
            f.write(description)

        # add to df
        image_row = {'path': temp_image_path, 'media_type': 'image', 'embeddings': image_embedding}
        text_row = {'path': temp_text_path, 'media_type': 'text', 'embeddings': text_embedding}

        df = pd.concat([df, pd.DataFrame([image_row])], ignore_index=True)
        df = pd.concat([df, pd.DataFrame([text_row])], ignore_index=True)

    return df

def load_embeddings_into_vector_db():
    combined_dataset = load_from_disk(COMBINED_DATASET_DIR)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14-336").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14-336")

    columns = ['path', 'media_type', 'embeddings']
    df = pd.DataFrame(columns=columns)

    df = process_dataset_for_embeddings(combined_dataset['train'], df, clip_model, clip_processor, device)
    df = process_dataset_for_embeddings(combined_dataset['validation'], df, clip_model, clip_processor, device)
    df = process_dataset_for_embeddings(combined_dataset['test'], df, clip_model, clip_processor, device)

    # Vérifier la dimension des embeddings
    df['embeddings'] = df['embeddings'].apply(lambda x: x if isinstance(x, list) and len(x) == 768 else [0] * 768)

    # load embeddings
    if not KDBAI_ENDPOINT or not KDBAI_API_KEY:
        raise ValueError("KDB.AI endpoint and API key must be set in config.py or as environment variables.")

    session = kdbai.Session(api_key=KDBAI_API_KEY, endpoint=KDBAI_ENDPOINT)

    table_schema = {
        "columns": [
            {"name": "path", "pytype": "str"},
            {"name": "media_type", "pytype": "str"},
            {
                "name": "embeddings",
                "pytype": "float32",
                "vectorIndex": {"dims": 768, "metric": "CS", "type": "flat"},
            },
        ]
    }

    try:
        session.table("medical_images").drop()
        time.sleep(5)
    except kdbai.KDBAIException:
        pass

    table = session.create_table("medical_images", table_schema)

    n = 500
    for i in tqdm(range(0, df.shape[0], n)):
        table.insert(df[i:i+n].reset_index(drop=True))
    print("Embeddings loaded into KDB.AI vector database.")

if __name__ == "__main__":
    load_embeddings_into_vector_db()
