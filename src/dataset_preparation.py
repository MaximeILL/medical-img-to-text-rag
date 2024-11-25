# src/dataset_preparation.py

import os
import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset
import re
from config import SKIN_METADATA_CSV, COMBINED_DATASET_DIR

def load_and_process_images():
    metadata = pd.read_csv(SKIN_METADATA_CSV).reset_index(drop=True)
    metadata_dataset = Dataset.from_pandas(metadata[['description']])
    pathvqa_dataset = load_dataset("flaviagiammarino/path-vqa")

    def correct_sentence_order(sentence):
      
        if "not easily seen" in sentence:
            if "are seen" in sentence:
                sentence = re.sub(r'(\w+ not easily seen) are seen', r'\1', sentence)
            if "are" not in sentence:
                parts = sentence.split(' ')
                idx = parts.index('not')
                sentence = ' '.join(parts[:idx]) + ' are ' + ' '.join(parts[idx:])

        if "shows does" in sentence:
            sentence = re.sub(r'shows does', 'does show', sentence)

        sentence = re.sub(r'\b(from|in|on) is\b', r'is \1', sentence)

        return sentence

    def clean_and_reorganize_answer(answer):
        cleaned_answer = re.sub(r'\b(\w+)\s+\1\b', r'\1', answer)
        words = cleaned_answer.split()
        seen = set()
        result = []
        for word in words:
            if word not in seen:
                seen.add(word)
                result.append(word)
        cleaned_answer = ' '.join(result)
        cleaned_answer = correct_sentence_order(cleaned_answer)
        return cleaned_answer

    # generate formated question / answer
    def generate_formatted_question_and_response(example):
        question = example['question'].strip().lower()
        original_answer = example['answer'].strip().lower()
        formatted_response = original_answer

        if formatted_response in ['yes', 'no']:
            if 'does' in question and 'show' in question:
                entity = re.sub(r"does this .*? show", "", question).rstrip('?').strip()
                default_entity = 'image'
                if formatted_response == 'yes':
                    formatted_response = f"This {default_entity} shows {entity}."
                else:
                    formatted_response = f"This {default_entity} does not show {entity}."

            elif "present" in question:
                entity = re.sub(r"(is|are|present|what is|what are)", "", question).rstrip('?').strip()
                verb = "are" if ',' in entity else "is"
                if formatted_response == 'yes':
                    formatted_response = f"{entity} {verb} present in this image."
                else:
                    formatted_response = f"{entity} {verb} not present in this image."
            elif question.startswith("are"):
                entity = re.sub(r"are ", "", question).rstrip('?').strip()
                if "not easily seen" in question:
                    formatted_response = f"{entity.capitalize()} are not easily seen."
                else:
                    formatted_response = f"{entity.capitalize()} are present." if formatted_response == 'yes' else f"{entity.capitalize()} are not present."
            elif question.startswith("is"):
                entity = re.sub(r"is ", "", question).rstrip('?').strip()
                if formatted_response == 'yes':
                    formatted_response = f"{entity.capitalize()} is present."
                else:
                    formatted_response = f"{entity.capitalize()} is not present."
        elif question.startswith("where are"):
            entity = re.sub(r"where are ", "", question).rstrip('?').strip()
            formatted_response = f"{entity.capitalize()} are {original_answer}."
        elif question.startswith("where is"):
            entity = re.sub(r"where is ", "", question).rstrip('?').strip()
            formatted_response = f"{entity.capitalize()} is {original_answer}."
        elif question.startswith("how does") and 'show' in question:
            entity = re.sub(r"how does this (.+) show", "", question).rstrip('?').strip()
            formatted_response = f"This image shows {entity} {original_answer}."
        elif question.startswith("what is present"):
            entity = formatted_response
            verb = "are" if ',' in entity else "is"
            formatted_response = f"This image shows {entity}."
        elif question.startswith("what is"):
            entity = re.sub(r"what is", "", question).replace("being utilized to do", "").strip().rstrip('?').strip()
            formatted_response = f"{entity.capitalize()} is utilized to {formatted_response}."
        elif question.startswith("what do") and 'show' in question:
            entity = re.sub(r"what do", "", question).replace("represent", "").replace("show", "").strip().rstrip('?').strip()
            formatted_response = f"This image shows {formatted_response}" if formatted_response else f"{entity} represent {formatted_response}."
        elif question.startswith("what does"):
            entity = re.sub(r"what does", "", question).replace("represent", "").replace("show", "").strip().rstrip('?').strip()
            formatted_response = f"{entity.capitalize()} represents {formatted_response}."
        elif question.startswith("are"):
            entity = re.sub(r"are", "", question).rstrip('?').strip()
            verb = "are" if ',' in entity else "is"
            if formatted_response == 'yes':
                formatted_response = f"{entity} {verb} present."
            else:
                formatted_response = f"{entity} {verb} not present."

        formatted_response = clean_and_reorganize_answer(formatted_response)
        example['question'] = question.capitalize()
        example['answer'] = formatted_response
        return example

    # apply fct
    pathvqa_dataset = pathvqa_dataset.map(generate_formatted_question_and_response)

    def filter_short_answers(example):
        answer = example['answer'].strip().lower()
        return answer not in ['yes', 'no'] and len(answer.split()) > 2

    filtered_pathvqa_dataset = DatasetDict({
        split: ds.filter(filter_short_answers)
        for split, ds in pathvqa_dataset.items()
    })

    # format datasets
    def reformat_dataset(example):
        return {'image': example['image'], 'description': example['answer']}

    reformatted_pathvqa_dataset = DatasetDict({
        split: ds.map(reformat_dataset, remove_columns=['question', 'answer'])
        for split, ds in filtered_pathvqa_dataset.items()
    })

    def filter_does_not(example):
        description = example['description'].strip().lower()
        return "does not" not in description

    combined_filtered_dataset = DatasetDict({
        'train': reformatted_pathvqa_dataset['train'].filter(filter_does_not),
        'validation': reformatted_pathvqa_dataset['validation'].filter(filter_does_not),
        'test': reformatted_pathvqa_dataset['test'].filter(filter_does_not)
    })

    # shuffle dataset
    combined_filtered_dataset = DatasetDict({
        'train': combined_filtered_dataset['train'].shuffle(seed=42),
        'validation': combined_filtered_dataset['validation'].shuffle(seed=42),
        'test': combined_filtered_dataset['test'].shuffle(seed=42)
    })

    # save combined dataset
    os.makedirs(COMBINED_DATASET_DIR, exist_ok=True)
    combined_filtered_dataset.save_to_disk(COMBINED_DATASET_DIR)
    print(f"Combined dataset saved to: {COMBINED_DATASET_DIR}")

    return combined_filtered_dataset

if __name__ == "__main__":
    combined_dataset = load_and_process_images()
