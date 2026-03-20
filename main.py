import numpy
from data import TextProcessor

def main():
    print("Loading and processing text...")
    #Load text data 
    with open("alice_dataset.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    processor = TextProcessor(window_size=2)
    training_data, vocabulary_size, word_to_id, id_to_word = processor.prepare_data(raw_text)

    print(f"Vocabulary size: {vocabulary_size}")
    print(f"Training pairs: {len(training_data)}")

    print("Initializing model...")
    #initialize model

    print("Start training loop...")
    #training loop

    print("Training complete. Testing embeddings...")
    #test embeddings
    pass

if __name__ == "__main__":
    main()