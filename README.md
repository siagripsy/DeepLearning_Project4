# DeepLearning Project 4

This project is a course assignment for **Deep Learning** and focuses on building a **Transformer-based model for sequence understanding**.

The main work is implemented in [Project4_Transformers.ipynb](c:\Users\sia\Desktop\DL\src\DeepLearning_Project4\Project4_Transformers.ipynb). The notebook uses the **IMDB Sentiment** dataset and walks through the full pipeline of preparing text data, implementing a Transformer classifier from scratch in PyTorch, training the model, evaluating it, and visualizing attention.

## Project Goals

The notebook is designed to:

- answer conceptual questions about Transformer models,
- prepare and tokenize the IMDB dataset,
- implement a Transformer encoder classifier,
- train and validate the model,
- evaluate final performance on the test set,
- visualize attention weights,
- provide short reflections on the strengths and challenges of Transformers.

## Notebook Contents

The notebook is organized into the following parts:

1. **Conceptual Questions**  
   Short explanations about:
   - why Transformers are useful compared to RNNs and LSTMs,
   - self-attention,
   - positional encoding,
   - multi-head attention.

2. **Dataset Preparation**  
   - loads the IMDB dataset using the Hugging Face `datasets` library,
   - splits data into train / validation / test sets,
   - tokenizes text with a simple custom tokenizer,
   - builds a vocabulary from the training set,
   - converts tokens to integer ids,
   - pads sequences and creates PyTorch dataloaders.

3. **Transformer Model Implementation**  
   The classifier includes:
   - token embeddings,
   - sinusoidal positional encoding,
   - stacked Transformer encoder blocks,
   - multi-head self-attention,
   - feed-forward layers,
   - mean pooling over valid tokens,
   - a classification head for sentiment prediction.

4. **Training and Evaluation**  
   - trains the model with cross-entropy loss,
   - tracks training loss and accuracy,
   - evaluates on validation and test sets,
   - plots training and validation curves.

5. **Attention Visualization**  
   - extracts attention weights from the first encoder layer,
   - visualizes one attention head as a heatmap,
   - helps interpret what the model focuses on.

6. **Reflection**  
   A short discussion of the advantages of Transformers and practical training challenges.

## Model Configuration

The current notebook uses the following default settings:

- `MAX_VOCAB_SIZE = 20000`
- `MAX_SEQ_LEN = 256`
- `BATCH_SIZE = 64`
- `EMBED_DIM = 128`
- `NUM_HEADS = 4`
- `FFN_DIM = 256`
- `NUM_LAYERS = 2`
- `DROPOUT = 0.2`
- `NUM_EPOCHS = 5`

To keep the notebook practical on CPU, it uses smaller default subsets of the IMDB dataset:

- `MAX_TRAIN_SAMPLES = 10000`
- `MAX_VAL_SAMPLES = 2500`
- `MAX_TEST_SAMPLES = 2500`

If you want to train on the full dataset, you can set these values to `None` inside the notebook.

## Requirements

You need Python 3 with the following libraries:

- `torch`
- `datasets`
- `matplotlib`

Example installation:

```bash
pip install torch datasets matplotlib
```

## How to Run

1. Open Jupyter Notebook or Jupyter Lab.
2. Open [Project4_Transformers.ipynb].
3. Run the cells in order from top to bottom.

Important notes:

- The IMDB dataset will be downloaded automatically the first time you run the notebook.
- Training time depends on whether you use CPU or GPU.
- The notebook saves the best model weights in memory during training and then evaluates that best version on the test set.

## Output

After running the notebook, you should see:

- example tokenized IMDB reviews,
- the printed Transformer model architecture,
- training loss and validation accuracy for each epoch,
- final test accuracy,
- training/validation plots,
- an attention heatmap for one review sample.

## Notes

- The code is heavily commented in English to make the implementation easier to understand.
- The Transformer model is implemented manually with PyTorch building blocks so the architecture is easier to study.
- This project is aimed at learning core Transformer ideas, not at maximizing benchmark performance.
