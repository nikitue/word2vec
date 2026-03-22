import random
import numpy
from data import TextProcessor
from model import SkipGramNegativeSampling

def main():
    print("Loading and processing text...")
    #Load text data 
    with open("alice_dataset.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    processor = TextProcessor(window_size=2)
    training_data, vocabulary_size, word_to_id, id_to_word = processor.prepare_data(raw_text)

    print(f"Vocabulary size: {vocabulary_size}")
    print(f"Training pairs: {len(training_data)}")

    #initialize model
    print("Initializing model...")
    embedding_dim = 50
    initial_lr = 0.025
    num_negative_samples = 5
    epochs = 5

    processed_pairs = 0
    total_pairs = len(training_data) * epochs

    model = SkipGramNegativeSampling(vocabulary_size, embedding_dim, initial_lr)
    print(f"Dimensions: {embedding_dim}")
    print(f"Negative Samples per pair: {num_negative_samples}\n")

    #training loop
    print("Start training loop...")
    for epoch in range(epochs):
        # Shuffle data at the start of each epoch for better SGD
        random.shuffle(training_data)
        total_loss = 0
        
        for target_id, context_id in training_data:
            #linearly reduce learning rate
            progress = processed_pairs / total_pairs
            model.l_rate = initial_lr * (1.0 - progress)
            
            # Don't reach absolute zero
            if model.l_rate < 0.0001:
                model.l_rate = 0.0001

            # Generate the random negative samples
            negative_ids = model.get_negative_samples(context_id, num_negative_samples)
            
            # Do a single train step (forward + backward pass and update weights) 
            loss = model.train_step(target_id, context_id, negative_ids)
            total_loss += loss
            processed_pairs += 1
            
        avg_loss = total_loss / len(training_data)
        print(f"Epoch {epoch + 1}/{epochs} ; Average Loss: {avg_loss:.4f}")

    #test embeddings
    print("Training complete. Testing embeddings...")
    test_words = ["alice", "rabbit", "queen"]
    
    for word in test_words:
        print(f"\nNearest neighbors to '{word}':")
        results = model.get_similar_words(word, word_to_id, id_to_word, top_k=4)
        if isinstance(results, str):
            print(results)
        else:
            for sim_word, score in results:
               print(f"   - {sim_word} (Score: {score:.3f})")
    
    #save model
    model.save_embeddings(word_to_id)

if __name__ == "__main__":
    main()