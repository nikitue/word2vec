import re
from collections import Counter

class TextProcessor:
    def __init__(self, window_size=2, min_freq=3):
        self.window_size = window_size
        self.min_freq = min_freq

    def clean_text(self, raw_text):
        # Convert to lowercase first
        text = raw_text.lower()
        
        # Replace hyphens and underscores with spaces to split compound words
        text = re.sub(r'[-_]', ' ', text)
        
        # Remove ALL remaining punctuation (quotes, commas, apostrophes, etc.)
        # "i'll" becomes "ill", "don't" becomes "dont", statistically shouldn't affect the embeddings
        text = re.sub(r'[^a-z\s]', '', text)
        
        # Split by whitespace into a list of words
        words = text.split()
        
        return words
    
    def build_vocab(self, words):
        # Count frequencies of all words
        word_counts = Counter(words)
        
        # Filter out words that don't meet the minimum frequency
        valid_words = [word for word, count in word_counts.items() if count >= self.min_freq]
        
        # Sort for consistency
        unique_words = sorted(list(set(valid_words))) #don't sort if the words are too many
        vocab_size = len(unique_words)
        
        word_to_id = {word: i for i, word in enumerate(unique_words)}
        id_to_word = {i: word for i, word in enumerate(unique_words)}
        
        return vocab_size, word_to_id, id_to_word

    def generate_training_pairs(self, words, word_to_id):
        # Filter the original text sequence to ONLY include valid words
        word_ids = [word_to_id[w] for w in words if w in word_to_id]
        
        training_pairs = []
        for i, target_id in enumerate(word_ids):
            # Ensure we don't go below index 0 or above last index
            start = max(0, i - self.window_size)
            end = min(len(word_ids), i + self.window_size + 1) 
            
            for j in range(start, end):
                if i != j:
                    training_pairs.append((target_id, word_ids[j]))
                    
        return training_pairs
    
    def prepare_data(self, raw_text):
        """
        The main function to call from main.py
        """
        words = self.clean_text(raw_text)
        vocab_size, word_to_id, id_to_word = self.build_vocab(words)
        training_pairs = self.generate_training_pairs(words, word_to_id)
        
        return training_pairs, vocab_size, word_to_id, id_to_word