# NumPy Word2Vec: Skip-Gram with Negative Sampling (SGNS)
A pure Python/NumPy implementation of the Word2Vec training. This project was built from scratch without relying on high-level ML frameworks like PyTorch or TensorFlow.

## Quick Start
Prerequisites: Python 3.x and NumPy.
pip install -r requirements.txt

## Execution:
Ensure you have a plain text file named alice_dataset.txt in the root directory or change the name to that of your dataset in main.py
python main.py

The script will automatically parse the text, build the vocabulary, train the embeddings over a few epochs, and print the loss. At the end, it will run a Cosine Similarity test to demonstrate the learned semantic relationships and save the .npy matrix.

## Project Structure
- data.py: Handles basic text filtering, text tokenization, vocabulary building, and sliding-window pair generation.

- model.py: Contains the weight matrices, forward pass, Loss calculations, vectorized gradients, and similarity scoring.

- main.py: The orchestrator file that loads the data, runs the training loop, applies learning rate decay, and runs the evaluation.

## Dataset
The chosen dataset for training is the book "Alice's adventures in Wonderland" which has a reasonable size and consistent context. The dataset is taken from Project Gutenberg https://www.gutenberg.org/

## Data Engineering Pipeline
Real-world text contains formatting noise and punctuation that can ruin embedding spaces. The TextProcessor class in data.py handles this by:

- Replacing hyphens and underscores with spaces to preserve distinct words.

- Minimum Frequency Threshold: Words appearing fewer than min_freq times are excluded. This naturally filters out extreme typos and structural tags, ensuring the model only trains on words with sufficient contextual data.

## Mathematical & Architectural Choices
To ensure training is fast and computationally stable in pure NumPy, the followigng specific optimization choices were made:

### Skip-gram over CBOW
The main motivation of the choice of using Skip-gram architecture is curiosity, since before doing more research on how it actually works, the one-to-many approach of predicting a lot of context based on just one word seemed counterintuitive. Moreover, Skip-gram treats each context word as a new observation which is benificial when using small datasets, like we do. Also capturing rare characters (like the Dormouse or the Hatter) needed an architecture that wouldn't dilute their vectors by averaging them with surrounding frequent words (common limitation of CBOW).

### Negative Sampling vs. Softmax
A standard Softmax output layer requires calculating the dot product for every single word in the vocabulary for every training pair, which is excessively slow in CPU-bound Python. By implementing Negative Sampling, the architecture reduces a massive multi-class classification problem into a series of lightweight binary classification tasks. For each positive pair, the network only updates the weights for the target word, the 1 true context word, and k random negative samples.

### Gradient Derivation
The main source for learning about making a word2vec implementation comes from the research paper "word2vec Parameter Learning Explained" by Xin Rong. The presented formulas in the paper regarding gradients for negative sampling are implemented. Before implementatiion, the formulas were recalculated manually (with a pen and notebook) to understand them thoroughly. 

## Numerical Stability
To prevent exploding gradients and NaN propagation (a problem encountered in a previous version), gradient clipping is utilized. Furthermore, a global linear learning rate decay was implemented across the entire training cycle to allow the vectors to settle into precise semantic clusters during the final epochs.

