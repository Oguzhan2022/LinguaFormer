# LinguaFormer: An English-to-Turkish Transformer Model

This repository contains the implementation of **LinguaFormer**, a Neural Machine Translation (NMT) model based on the Transformer architecture. The model is built from scratch using Keras 3 and TensorFlow to translate sentences from English to Turkish.

The entire workflow, from data preprocessing and custom tokenizer training to model building, training, and evaluation, is contained within the included Jupyter Notebook.

## Model Architecture & Hyperparameters

The model follows the classic encoder-decoder structure of a Transformer, optimized for this specific translation task.

-   **Encoder:** Processes the English input sentence.
    -   Consists of a `TokenAndPositionEmbedding` layer and a `TransformerEncoder` layer.
-   **Decoder:** Generates the Turkish translation based on the encoder's output.
    -   Consists of a `TokenAndPositionEmbedding` layer, a `TransformerDecoder` layer, and a final `Dense` layer for prediction.

### Key Hyperparameters
-   **Epochs:** 25
-   **Batch Size:** 128
-   **Max Sequence Length:** 25 tokens
-   **Embedding Dimension:** 256
-   **Number of Attention Heads:** 8
-   **Intermediate (Feed-Forward) Dimension:** 2048

---

## Performance & Results

The model was trained on a dataset of over 700,000 English-Turkish sentence pairs and evaluated both quantitatively (with ROUGE scores) and qualitatively.

### Training Performance
After 25 epochs, the model achieved:
-   **Training Accuracy:** ~91.92%
-   **Validation Accuracy:** ~90.63%

### Quantitative Evaluation (ROUGE Scores)
The model's translation quality was measured on 300 test samples using ROUGE metrics, which compare the generated translations to reference translations.

| Metric    | Precision | Recall | F1 Score |
|-----------|:---------:|:------:|:--------:|
| **ROUGE-1** |  0.6922   | 0.6935 |  0.6764  |
| **ROUGE-2** |  0.5176   | 0.5178 |  0.5128  |
| **ROUGE-L** |  0.6803   | 0.6815 |  0.6746  |

### Qualitative Assessment
-   **Strengths:** The model performs quite well on simple and direct sentences.
-   **Weaknesses:** There are minor structural or word-choice errors in some sentences, but the main idea is generally understandable.
-   **Challenges:** The model struggles with idioms or more complex structures/wordplay, tending towards word-for-word translations. The mistranslation of "free" as "ücretsizdir" (free of charge) instead of "serbest" (free/unconstrained), or the generation of a nonsensical word like "fer" for "horse", indicates situations where the model may not fully grasp the context.

---

## Technologies Used

-   Python 3
-   Keras 3 & TensorFlow
-   Keras-Hub (for custom layers and tokenizers)
-   TensorFlow-Text (for WordPiece vocabulary generation)
-   Rouge-Score (for evaluation)

---

## How to Use

### Prerequisites

-   Python 3.8+
-   `pip` and `venv` (recommended)

### Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Oguzhan2022/LinguaFormer.git
    cd LinguaFormer
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    # On Windows:
    .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install --upgrade rouge-score keras-hub keras tensorflow tensorflow-text
    ```

4.  **Download the Dataset:**
    -   Go to the dataset page on Kaggle: **[English-to-Turkish Sentence Pairs](https://www.kaggle.com/datasets/orvile/english-to-turkish-sentence-pairs/data)**.
    -   Download the file named `Sentence pairs in English-Turkish - 2025-05-06.tsv`.
    -   Place the downloaded `.tsv` file inside the `LinguaFormer` project folder you cloned.

5.  **Run the Jupyter Notebook:**
    -   Launch the `LinguaFormer_EN_TR_Translation.ipynb` notebook.
    -   **Important:** Before running, make sure the `file_path` variable in the notebook correctly points to the dataset file. For example:
        ```python
        # Update this path if you placed the file somewhere else
        file_path = "Sentence pairs in English-Turkish - 2025-05-06.tsv"
        ```
    -   Run all the cells sequentially to train the tokenizers, build the model, and run the training process.
