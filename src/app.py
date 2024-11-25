import os
from PIL import Image
import torch
from transformers import pipeline, CLIPProcessor, CLIPModel
import kdbai_client as kdbai
from sklearn.metrics.pairwise import cosine_similarity
from IPython.display import display, HTML
import ipywidgets as widgets
import time
from difflib import SequenceMatcher
from collections import defaultdict
from config import KDBAI_ENDPOINT, KDBAI_API_KEY

device = "cuda:0" if torch.cuda.is_available() else "cpu"

clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14-336").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14-336")

llama_model_name_or_path = "m42-health/Llama3-Med42-8B"
llama_pipeline = pipeline(
    "text-generation",
    model=llama_model_name_or_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# connect to kdb vector db
if not KDBAI_ENDPOINT or not KDBAI_API_KEY:
    raise ValueError("KDB.AI endpoint and API key must be set in config.py or as environment variables.")

session = kdbai.Session(api_key=KDBAI_API_KEY, endpoint=KDBAI_ENDPOINT)
table = session.table("medical_images")

def data_to_embedding(data_in, data_type):
    """
    Generate an embedding vector for the input data using CLIP.
    params :
        - data_in: input data (image or text).
        - data_type: Type of data ('image' or 'text').
    output : A vector embedding of the input data.
    """
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

def read_text_from_file(filename):
    """
    Read text content from a file.
    params : filename: Path to the file.
    output : content of the file as a string, or None if error.
    """
    try:
        with open(filename, 'r') as file:
            text = file.read()
        return text
    except IOError as e:
        print(f"An error occurred: {e}")
        return None

def mm_search(query_vector):
    """
    Perform a search in the kdb vector db using a query vector.
    params : query_vector: The vector representation of the query.
    output : descriptions and embeddings of the matching entries.
    """
    text_results = table.search(query_vector, n=10, filter=[("like", "media_type", "text")])
    descriptions = []
    embeddings = []
    for index, row in text_results[0].iterrows():
        if row['media_type'] == 'text':
            text_content = read_text_from_file(row['path'])
            descriptions.append(text_content)
            embeddings.append(row['embeddings'])
    return descriptions, embeddings

def evaluate_results(image_embedding, text_embeddings, descriptions):
    """
    Evaluate similarities between an image embedding and a set of text embeddings.
    params :
        - image_embedding: The embedding of the image.
        - text_embeddings: The embeddings of the text descriptions.
        - descriptions: The list of text descriptions.
    output : list of the top 5 matches with their similarity scores.
    """
    similarities = cosine_similarity([image_embedding], text_embeddings)
    top_indices = similarities[0].argsort()[-5:][::-1]
    best_matches = [(descriptions[i], similarities[0][i]) for i in top_indices]
    return best_matches

def normalize_text(text):
    """
    Normalize a text string by converting it to lowercase and removing extra spaces.
    params : text: input text.
    output : normalized text string.
    """
    return ' '.join(text.lower().split())

def are_similar(text1, text2, threshold=0.95):
    """
    Check if two texts are similar based on a given similarity threshold.
    params :
        - text1: First text.
        - text2: Second text.
        - threshold: similarity threshold (default is 0.95).
    output : True if the similarity ratio is above threshold, otherwise False.
    """
    return SequenceMatcher(None, normalize_text(text1), normalize_text(text2)).ratio() > threshold

def calculate_vote_score(best_match, suggestions):
    """
    Calculate a weighted consensus score for the suggestions based on similarity.
    params :
        - best_match: initial best match description.
        - suggestions: list of alternative suggestions.
    output : best suggestion based on consensus, and the sorted suggestions with scores.
    """
    best_match_embedding = data_to_embedding(best_match, 'text')
    suggestion_embeddings = [data_to_embedding(suggestion[0], 'text') for suggestion in suggestions]
    similarities = cosine_similarity([best_match_embedding], suggestion_embeddings)[0]
    similarity_dict = defaultdict(lambda: {'original_text': "", 'total_similarity': 0.0, 'count': 0, 'original_similarity': 0.0})

    for suggestion, similarity in zip(suggestions, similarities):
        found_similar = False
        for key in similarity_dict.keys():
            if are_similar(suggestion[0], key):
                similarity_dict[key]['total_similarity'] += similarity
                similarity_dict[key]['count'] += 1
                found_similar = True
                break

        if not found_similar:
            normalized_suggestion = normalize_text(suggestion[0])
            similarity_dict[normalized_suggestion]['original_text'] = suggestion[0]
            similarity_dict[normalized_suggestion]['total_similarity'] += similarity
            similarity_dict[normalized_suggestion]['original_similarity'] = similarity
            similarity_dict[normalized_suggestion]['count'] += 1

    weighted_scores = []
    for normalized_suggestion, values in similarity_dict.items():
        weighted_score = values['total_similarity'] * values['count']
        original_similarity = values['original_similarity']
        weighted_scores.append((values['original_text'], weighted_score, original_similarity))

    sorted_suggestions = sorted(weighted_scores, key=lambda x: x[1], reverse=True)
    return sorted_suggestions[0][0], sorted_suggestions

def generate_detailed_description(best_match):
    """
    Generate a detailed description for the best match using LLaMA.
    params : best_match: best match description.
    output : generated detailed description.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful, respectful and honest medical assistant. "
                "Always answer as helpfully as possible, while being safe. "
                "Focus only on the topic provided by the user. "
                "Your answers should directly address the user's query without introducing unrelated topics. "
            ),
        },
        {"role": "user", "content": f"Provide a detailed and focused description of the following medical condition only: {best_match}. Do not discuss any other topics and end your sentences."},
    ]

    prompt = llama_pipeline.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )

    stop_tokens = [llama_pipeline.tokenizer.eos_token_id]

    outputs = llama_pipeline(
        prompt,
        max_new_tokens=256,
        eos_token_id=stop_tokens[0],
        do_sample=True,
        temperature=0.2,
        top_k=150,
        top_p=0.75,
    )

    generated_text = outputs[0]["generated_text"][len(prompt):].strip()
    return generated_text

def generate_specialist_advice():
    """
    Generate advice to consult a specialist if no high-confidence matches are found.
    output : Generated advice as a string.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful, respectful, and honest medical assistant. "
                "Always answer as helpfully as possible, while being safe. "
                "Provide clear and direct advice for seeking professional medical help when necessary."
            ),
        },
        {
            "role": "user",
            "content": (
                "The provided image does not match any known medical conditions with high confidence. "
                "Advise the user to consult a healthcare professional."
            ),
        },
    ]

    prompt = llama_pipeline.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )

    outputs = llama_pipeline(
        prompt,
        max_new_tokens=150,
        eos_token_id=llama_pipeline.tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.2,
        top_k=150,
        top_p=0.75,
    )

    generated_text = outputs[0]["generated_text"][len(prompt):].strip()
    return generated_text

def search_text_by_image(image_path):
    """
    Perform a search for textual descriptions matching the given image.
    params : image_path: Path to the image file.
    output : detailed description and sorted suggestions.
    """
    image = Image.open(image_path)
    image_embedding = data_to_embedding(image, 'image')
    descriptions, text_embeddings = mm_search([image_embedding])

    best_matches = evaluate_results(image_embedding, text_embeddings, descriptions)
    initial_best_match_description = best_matches[0][0]

    new_best_match_description, sorted_suggestions = calculate_vote_score(initial_best_match_description, best_matches[1:5])

    if new_best_match_description != initial_best_match_description:
        best_match_message = f"Changement de Best Match '{initial_best_match_description}' ➔ '{new_best_match_description}'"
    else:
        best_match_message = f"Best Match : '{initial_best_match_description}'"

    display(HTML(f"<div style='font-family: Arial, sans-serif; color: #fff; background-color: #f7a795; padding: 10px; border-radius: 10px; border: 1px solid #ccc;'>{best_match_message}</div>"))

    if len(sorted_suggestions) == 0 or sorted_suggestions[0][1] < 0.64:
        detailed_description = generate_specialist_advice()
    else:
        detailed_description = generate_detailed_description(new_best_match_description)

    return detailed_description, sorted_suggestions

def display_suggestions(adjusted_suggestions):
    """
    Display the top suggestions in an HTML formatted box.
    params : adjusted_suggestions: List of suggestions with scores.
    """
    suggestion_html = "<div style='font-family: Arial, sans-serif; color: #fff; background-color: #6bbbe2; padding: 10px; border-radius: 10px; border: 1px solid #ccc;'>"

    for i, (desc, weighted_score, similarity_score) in enumerate(adjusted_suggestions[:3], 1):
        suggestion_html += (
            f"<b>Suggestion {i}:</b> {desc} "
            f"(<i>Weighted Consensus Score: {weighted_score:.2f} - Similarity Score: {similarity_score:.2f}</i>)<br>"
        )

    suggestion_html += "</div>"

    display(HTML(suggestion_html))

def main():
    """
    Main function to run the application. Allows users to upload an image and view results.
    """
    upload = widgets.FileUpload(accept='image/*', multiple=False)
    output = widgets.Output()
    result_label = widgets.HTML("")

    def on_upload_change(change):
        for name, file_info in upload.value.items():
            image_path = name
            with open(image_path, 'wb') as f:
                f.write(file_info['content'])

            output.clear_output(wait=True)
            with output:
                display(Image.open(image_path))

            detailed_description, best_matches = search_text_by_image(image_path)
            result_label.value = f"<div style='font-family: Arial, sans-serif; color: #333; background-color: #f9f9f9; padding: 10px; border-radius: 10px; border: 1px solid #ccc;'><b>Generated Description:</b><br>{detailed_description}</div>"
            display_suggestions(best_matches)

    upload.observe(on_upload_change, names='value')
    display(widgets.VBox([upload, output, result_label]))

if __name__ == "__main__":
    main()
