# LinguaFormer: An English-to-Turkish Transformer Model

This repository contains the implementation of **LinguaFormer**, a Neural Machine Translation (NMT) model based on the Transformer architecture. The model is built from scratch using Keras 3 and TensorFlow to translate sentences from English to Turkish.

## Features

-   **Transformer Architecture:** Implements the core encoder-decoder structure of a Transformer, which is the state-of-the-art for sequence-to-sequence tasks.
-   **Custom Tokenization:** Utilizes WordPiece tokenizers trained from scratch on the English and Turkish training data for optimal subword tokenization.
-   **Modern Tech Stack:** Built with Keras 3, TensorFlow, and `keras_hub` components for efficient and modular model construction.
-   **Efficient Data Handling:** Uses `tf.data.Dataset` to create a highly efficient, parallelized data pipeline for training.
-   **Inference with Greedy Sampling:** Implements a decoding function using a `GreedySampler` to generate translations for new input sentences.
-   **Performance Evaluation:** The model's performance is quantitatively measured using ROUGE scores (ROUGE-1, ROUGE-2, and ROUGE-L).

## Model Architecture

The model follows the classic encoder-decoder structure:

1.  **Encoder:** The encoder stack processes the English input sentence. It consists of:
    -   A `TokenAndPositionEmbedding` layer to create input embeddings.
    -   A `TransformerEncoder` layer to build contextual representations of the input sequence.

2.  **Decoder:** The decoder stack generates the Turkish translation based on the encoder's output. It consists of:
    -   A `TokenAndPositionEmbedding` layer for the target sequence.
    -   A `TransformerDecoder` layer that attends to both the encoder's output and the previously generated tokens to predict the next token.
    -   A final `Dense` layer with a softmax activation to produce the probability distribution over the Turkish vocabulary.

## Performance & Results

The model was trained for 25 epochs. The performance was evaluated both qualitatively (by observing translated examples) and quantitatively (using ROUGE metrics).

### ROUGE Score Evaluation

The model was evaluated on 300 samples from the test set.

| Metric    | Precision | Recall | F1 Score |
|-----------|:---------:|:------:|:--------:|
| **ROUGE-1** |  0.4907   | 0.4939 |  0.4851  |
| **ROUGE-2** |  0.2676   | 0.2694 |  0.2625  |
| **ROUGE-L** |  0.4655   | 0.4682 |  0.4599  |

### Qualitative Assessment

-   **Strengths:** The model performs well on simple, direct sentences, capturing the main idea correctly.
-   **Weaknesses:** It struggles with idioms, complex sentence structures, and wordplay, sometimes resorting to literal, word-for-word translations that can be awkward or incorrect. For example, it might translate "free" as "ücretsiz" (free of charge) instead of "serbest" (unconstrained) depending on the context.

## Technologies Used

-   Python 3
-   Keras 3
-   TensorFlow
-   Keras-Hub
-   TensorFlow-Text
-   Rouge-Score

---

## How to Use

### Prerequisites

-   Python 3.8+
-   `pip` and `venv`

### Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/LinguaFormer.git
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
    pip install -q --upgrade rouge-score keras-hub keras tensorflow tensorflow-text
    ```

4.  **Download the Dataset:**
    -   You need the dataset file: `Sentence pairs in English-Turkish - 2025-05-06.tsv`.
    -   Place this file in a location accessible by the notebook (e.g., inside a `/kaggle/input/veriseti/` directory as in the original code, or in the root project folder).
    -   **Important:** Make sure to update the `file_path` variable in the notebook to point to the correct location of your dataset file.

5.  **Run the Jupyter Notebook:**
    Launch the `LinguaFormer_EN_TR_Translation.ipynb` notebook and run the cells sequentially.