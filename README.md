# Medical Image-to-Text RAG System

This project implements an image-to-text RAG system for medical imaging. It generates detailed descriptions of medical images using CLIP and a specialized instruct-tuned Llama model, leveraging data from the Hugging Face dataset PathVQA and PAD-UFES-20 from Kaggle.

## Project Overview

Data Processing: Cleans and enriches metadata from a CSV file (metadata.csv).

Dataset Preparation: Loads and processes images and data from the Hugging Face 'pathvqa' dataset.

Vector Database: Uses KDB.AI to store embeddings for efficient similarity search.

Application: Provides a user interface to upload images and receive generated medical descriptions.


## System Pipeline
<img src="/pipeline/schema_ragf.png" alt="Our RAG Pipeline" />


## Results

Here are some examples of descriptions generated by our system, and comparisons with LLaVa-1.6-Mistral-7B model :

<img src="/results_examples/lung.png" alt="Example 1: lungs" />
<img src="/results_examples/pa_liver_rag.png" alt="Example 2: pancreas and liver" />
<img src="/results_examples/br_f.png" alt="Example 3: brain subarachnoid hemorrhage" />
<img src="/results_examples/sp_f.png" alt="Example 4: spleen" />

## Data

### PAD-UFES-20
<img src="/pipeline/cap2.png" alt="PAD-UFES-20 data transformation" />

A dermatological dataset containing skin lesion images with associated tabular metadata. The dataset includes 26 features covering patient demographics, clinical diagnosis, and lesion characteristics. The original metadata is transformed from structured format to natural language descriptions this way

### PathVQA
<img src="/pipeline/pf1" alt="PathVQA data transformation" />

A diverse medical imaging dataset with various types of medical images (organs, tumors, scans, microscopy). Question-answer pairs are converted into descriptive statements

## Folder Structure

```
medical-image-to-text-rag/
├── data/
│   ├── metadata.csv          # metadata file (PAD-UFES-20)
│   └── skin_metadata.csv     # Generated after processing
├── src/
│   ├── config.py
│   ├── data_processing.py    # PathVQA
│   ├── dataset_preparation.py
│   ├── vector_database.py
│   └── app.py
├── requirements.txt
└── README.md
```

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/MaximeILL/medical-image-to-text-rag.git
cd medical-image-to-text-rag
```

### 2. Install Dependencies

Ensure you have Python 3.7 or higher installed.

```bash
pip install -r requirements.txt
```

### 3. Configure Paths

Review and adjust paths in src/config.py if necessary.

Alternatively, set environment variables to override default paths.


### 4. Run Data Processing

Process the metadata to generate skin_metadata.csv.

```bash
python src/data_processing.py
```

### 5. Prepare the Dataset

This step loads the Hugging Face 'PathVQA' dataset and processes it.

```bash
python src/dataset_preparation.py
```

### 6. Load Embeddings into Vector Database

Ensure you have access to a KDB.AI endpoint and have your API key ready.

Set your KDB.AI credentials:

```bash
export KDBAI_ENDPOINT='your_kdbai_endpoint'
export KDBAI_API_KEY='your_kdbai_api_key'
```

Load embeddings into the vector database:

```bash
python src/vector_database.py
```
*Note* : KDB.AI API evolves regularly and may be subject to change. Do not hesitate to consult the documentation in case of version or compatibility problems.

### 7. Run the Application

Launch the user interface to upload images and receive generated descriptions.

```bash
python src/app.py
```

This will start the application. Follow the on-screen instructions to upload an image and receive the generated medical description.

## Project Structure Details

### src/config.py

Contains configuration variables for the project, such as data paths and KDB AI credentials.

### src/data_processing.py

Processes the initial metadata.csv file to generate skin_metadata.csv with enriched descriptions.

### src/dataset_preparation.py

- Loads and processes the 'pathvqa' dataset from Hugging Face.
- Turn initial labels into meaningful declarative sentences
- Combines both datasets (PathVQA and PAD-UFES-20) and saves the processed dataset to disk.


### src/vector_database.py

- Loads the processed dataset.
- Computes embeddings for images and text descriptions using CLIP.
- Loads embeddings into the KDB.AI vector database for similarity search.


### src/app.py

Provides a user interface to:

- Upload an image.
- Process the image to find similar descriptions using the vector database.
- Generate a detailed medical description following our pipeline.


## Notes

- Data Download: Images and data are downloaded from the Hugging Face 'pathvqa' dataset automatically during dataset preparation.
- Memory Requirements : the use of models m42-health/Llama3-Med42-8B coupled with CLIP Large require an A100 GPU to run efficiently.

## Dependencies

The project requires the following Python packages:
```
pandas
numpy
torch
Pillow
transformers
datasets
kdbai_client
scikit-learn
tqdm
ipywidgets
ipython
matplotlib
```

Install them using:

```bash
pip install -r requirements.txt
```

## References

- PathVQA dataset: https://huggingface.co/datasets/flaviagiammarino/path-vqa
- PAD-UFES-20 dataset: https://www.kaggle.com/datasets/mahdavi1202/skin-cancer
- KDB AI: https://kdb.ai/

*Disclaimer: This project is for educational purposes. The generated medical descriptions should not be used for clinical diagnosis or treatment. Always consult a qualified healthcare professional for medical advice.*