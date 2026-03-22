import numpy as np
import math 

def sigmoid(x):
    x = np.clip(x, -10, 10)  # Prevents overflow if dot product gets too large/small
    return 1 / (1 + np.exp(-x))

class SkipGramNegativeSampling:
    def __init__(self, vocabulary_size, embedding_dim=50,learning_rate=0.025):
        self.vocabulary_size = vocabulary_size
        self.embedding_dim = embedding_dim
        self.l_rate = learning_rate

        #Weight matrices
        self.W_target = np.random.uniform(-0.1, 0.1, (vocabulary_size, embedding_dim)) #the actual word embedding (input to hidden layer)
        self.W_context = np.random.uniform(-0.1, 0.1, (vocabulary_size, embedding_dim)) #the context word embedding (hidden to output layer)
    
    def train_step(self, target_id, context_id, negative_ids):
        v_w = self.W_target[target_id] #embedding for target word, i.e. hidden layer vector (h in the research paper)
        v_c = self.W_context[context_id] #vector for true context word
        v_n = self.W_context[negative_ids] #vectors for negative samples

        #forward pass
        z_pos = np.dot(v_w, v_c) #similiarity of target and true context word
        z_neg = np.dot(v_n, v_w) #similarity of negative samples to target word

        #squash values to probabilities
        p_pos = sigmoid(z_pos) 
        p_neg = sigmoid(z_neg) 

        err_pos = p_pos - 1
        err_neg = p_neg # - 0

        #loss
        loss = -np.log(p_pos) - np.sum(np.log(1 - p_neg)) #not really needed, helpful for monitoring

        if not (math.isnan(p_pos) and np.isnan(p_neg).all() and np.isnan(err_pos) and np.isnan(err_neg).all() and np.isnan(z_pos) and np.isnan(z_neg).all()):
            print("z_pos:", z_pos, "p_pos:", p_pos)
            print("z_neg:", z_neg, "p_neg:", p_neg)
            print("err_pos:", err_pos, "err_neg:", err_neg)
            print(f"Loss: {loss:.4f}")

        #gradients
        grad_target = err_pos * v_c * np.dot(err_neg, v_n) 
        grad_true_context_word  = err_pos * v_c
        grad_negative_sample = np.outer(err_neg, v_w)

        #clip gradients to prevent exploding gradients
        grad_target = np.clip(grad_target, -1.0, 1.0)
        grad_true_context_word = np.clip(grad_true_context_word, -1.0, 1.0)
        grad_negative_sample = np.clip(grad_negative_sample, -1.0, 1.0)

        #backward pass
        self.W_target[target_id] -= self.l_rate * grad_target
        self.W_context[context_id] -= self.l_rate * grad_true_context_word
        self.W_context[negative_ids] -= self.l_rate * grad_negative_sample

        return loss

    def get_negative_samples(self, context_id, num_samples=5):
        #there is a slight chance of picking another context word as a negative sample
        #but in huge datasets the chance and effect it may have are statistically small
        negative_ids = []
        while len(negative_ids) < num_samples:
            rand_id = np.random.randint(0, self.vocabulary_size)
           # Don't pick the true context word as a negative sample 
            if rand_id != context_id:
                negative_ids.append(rand_id)
        return negative_ids
    
    def get_similar_words(self, word, word_to_id, id_to_word, top_k=5):
        """
        Finds the closest words in the embedding space using Cosine Similarity.
        """

        if word not in word_to_id:
            return f"Word '{word}' not in vocabulary."
            
        word_id = word_to_id[word]
        target_vec = self.W_target[word_id]
        
        # Calculate dot product of target with all embedding vectors
        dot_products = np.dot(self.W_target, target_vec)
        
        # Calculate norms
        matrix_norms = np.linalg.norm(self.W_target, axis=1)
        target_norm = np.linalg.norm(target_vec)
        
        # Divide dot products by the product of the norms
        similarities = dot_products / (matrix_norms * target_norm + 1e-8) #prevent division by zero
        
        #get top_k closest words, excluding the word itself at index 0
        closest_ids = np.argsort(similarities)[::-1][1:top_k+1]
        
        # Format the output for easy reading
        results = [(id_to_word[idx], similarities[idx]) for idx in closest_ids]
        return results