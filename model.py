import numpy as np

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

        print("z_pos:", z_pos, "p_pos:", p_pos)
        print("z_neg:", z_neg, "p_neg:", p_neg)
        print("err_pos:", err_pos, "err_neg:", err_neg)
    
        #loss
        loss = -np.log(p_pos) - np.sum(np.log(1 - p_neg)) #not really needed, helpful for monitoring
        print(f"Loss: {loss:.4f}")

        #gradients
        grad_target = err_pos * v_c * np.dot(err_neg, v_n) 
        grad_true_context_word  = err_pos * v_c
        grad_negative_sample = np.outer(err_neg, v_w)

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
            rand_id = np.random.randint(0, self.vocab_size)
           # Don't pick the true context word as a negative sample 
            if rand_id != context_id:
                negative_ids.append(rand_id)
        return negative_ids