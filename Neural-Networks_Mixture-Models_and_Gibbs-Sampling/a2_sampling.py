import enum
import numpy as np 
import string 
import random 

class GibbsSampler:
    def __init__(self, K, alpha, beta1, beta2):
        self.K = K
        self.alpha = alpha # hyperparam for pi_i
        self.beta1 = beta1 # hyperparam for theta_k^b
        self.beta2 = beta2 # hyperparam for theta_k^h
        self.N = 0 # number of articles  
        self.V = 0 # vocab size

        self.bodies_data_ids = []
        self.headlines_data_ids = []
        self.stopwords = []
        self.word2id = {}
        self.vocab = []

        # topic assignment for each word 
        self.z_b = [] 
        self.z_h = []

        self.n_ik = None # count of words in article i assigned to topic k 
        self.n_kv_b = None # count of word v in body assigned to k 
        self.n_k_b = None # total count of words in body assigned to topic k 
        self.n_kv_h = None # count of word v in head assigned to topic k 
        self.n_k_h = None # total count of words in head assigned to topic k

        # estimated params after sampling 
        self.pi_estimate = None 
        self.theta_b_estimate = None 
        self.theta_h_estimate = None 

    def load_stopwords(self, file_name):
        self.stopwords = []
        with open(file_name, 'r', encoding='utf-8') as infile:
            for line in infile:
                line = line.rstrip()
                self.stopwords.append(line)

    def load_data(self, file_name):
        self.word2id = {}
        self.vocab = []
        bodies_as_words = []
        headlines_as_words = [] 

        with open(file_name, 'r', encoding='utf-8') as infile:
            for line in infile:
                line = line.rstrip()
                cols = line.split('\t')
                if len(cols) < 3:
                    continue 
                body, headline, _ = cols[0], cols[1], cols[2]
                body_words = body.lower().translate(str.maketrans('', '', string.punctuation)).split()
                headline_words = headline.lower().translate(str.maketrans('', '', string.punctuation)).split()
                bodies_as_words.append(body_words)
                headlines_as_words.append(headline_words)

        bodies_as_ids = []
        
        for body in bodies_as_words:
            body_as_ids = []
            for word in body:
                if word.isalpha() and word not in self.stopwords:
                    if word not in self.word2id:
                        self.word2id[word] = len(self.vocab)
                        self.vocab.append(word)
    
                    word_id = self.word2id[word]
                    body_as_ids.append(word_id)
            bodies_as_ids.append(body_as_ids)

        headlines_as_ids = []

        for headline in headlines_as_words:
            headline_as_ids = []
            for word in headline:
                if word.isalpha() and word not in self.stopwords:
                    if word not in self.word2id:
                        self.word2id[word] = len(self.vocab)
                        self.vocab.append(word)
    
                    word_id = self.word2id[word]
                    headline_as_ids.append(word_id)
            headlines_as_ids.append(headline_as_ids)

        # Add code here to store the bodies and the headlines
        self.bodies_data_ids = [] 
        self.headlines_data_ids = []
        for i in range(len(bodies_as_ids)):
            if bodies_as_ids[i]: # if not empty
                self.bodies_data_ids.append(bodies_as_ids[i])
                self.headlines_data_ids.append(headlines_as_ids[i])
                
        self.N = len(self.bodies_data_ids)
        self.V = len(self.vocab)

        print(f"Loaded {self.N} articles with non-empty processed bodies.")
        print(f"Vocabulary size: {self.V}")

    def init_sampler_state(self):
        if self.N == 0 or self.V == 0:
            raise ValueError("Data not loaded or vocab/document count is zero")

        self.z_b = [np.zeros(len(doc_words), dtype=int) for doc_words in self.bodies_data_ids]
        self.z_h = [np.zeros(len(doc_words), dtype=int) for doc_words in self.headlines_data_ids]

        self.n_ik = np.zeros((self.N, self.K), dtype=int)
        self.n_kv_b = np.zeros((self.K, self.V), dtype=int)
        self.n_k_b = np.zeros(self.K, dtype=int)
        self.n_kv_h = np.zeros((self.K, self.V), dtype=int)
        self.n_k_h = np.zeros(self.K, dtype=int)

        for i in range(self.N):
            # init body word topics and counts 
            if self.bodies_data_ids[i]:
                for j, word_id in enumerate(self.bodies_data_ids[i]):
                    topic = random.randrange(self.K)
                    self.z_b[i][j] = topic 
                    self.n_ik[i, topic] += 1 
                    self.n_kv_b[topic, word_id] += 1 
                    self.n_k_b[topic] += 1 

            # init headline word topics and counts 
            if self.headlines_data_ids[i]:
                for l, word_id in enumerate(self.headlines_data_ids[i]):
                    topic = random.randrange(self.K)
                    self.z_h[i][l] = topic
                    self.n_ik[i, topic] += 1
                    self.n_kv_h[topic, word_id] += 1
                    self.n_k_h[topic] += 1 

        print("sampler state initialized with random topic assignments and counts")

    def run_gibbs_iter(self):
        # Iterate through all body words
        for i in range(self.N):
            if not self.bodies_data_ids[i]: 
                continue 
            M_i = len(self.bodies_data_ids[i])
            L_i = len(self.headlines_data_ids[i])

            for j in range(M_i):
                word_id = self.bodies_data_ids[i][j]
                old_topic = self.z_b[i][j]

                # decrement counts for the current word 
                self.n_ik[i, old_topic] -= 1 
                self.n_kv_b[old_topic, word_id] -= 1
                self.n_k_b[old_topic] -= 1 

                # calc sampling probs for each topic k 
                sampling_probs = np.zeros(self.K) 
                for k_prime in range(self.K):
                    # term A - word topic for body 
                    term_A_num = self.n_kv_b[k_prime, word_id] + self.beta1
                    term_A_den = self.n_k_b[k_prime] + self.V * self.beta1
                    term_A = term_A_num / term_A_den if term_A_den > 0 else 0

                    # term B - topic article 
                    term_B_num = self.n_ik[i, k_prime] + self.alpha
                    term_B_den = (M_i - 1) + L_i + self.K * self.alpha
                    term_B = term_B_num / term_B_den if term_B_den > 0 else 0

                    sampling_probs[k_prime] = term_A * term_B 

                # Normalize probs 
                sum_probs = np.sum(sampling_probs)
                if sum_probs > 1e-9:
                    # avoid div by 0 if all probabilities are small
                    sampling_probs /= sum_probs 
                else: 
                    # fallback to uniform if all probabilities are zero
                    sampling_probs = np.ones(self.K) / self.K 

                # sample new topic 
                new_topic = np.random.choice(self.K, p=sampling_probs)
                self.z_b[i][j] = new_topic 

                # iincrement counts with the new topic 
                self.n_ik[i, new_topic] += 1
                self.n_kv_b[new_topic, word_id] += 1
                self.n_k_b[new_topic] += 1 
        
        # Iterate through all headline words
        for i in range(self.N):
            if not self.headlines_data_ids[i]:
                continue 
            M_i = len(self.bodies_data_ids[i])
            L_i = len(self.headlines_data_ids[i])

            for l in range(L_i):
                word_id = self.headlines_data_ids[i][l]
                old_topic = self.z_h[i][l]

                # decrement counts 
                self.n_ik[i, old_topic] -= 1
                self.n_kv_h[old_topic, word_id] -= 1
                self.n_k_h[old_topic] -= 1 

                # calculate sampling probabilities 
                sampling_probs = np.zeros(self.K)
                for k_prime in range(self.K):
                    # Term C - word topic for headline 
                    term_C_num = self.n_kv_h[k_prime, word_id] + self.beta2
                    term_C_den = self.n_k_h[k_prime] + self.V * self.beta2
                    term_C = term_C_num / term_C_den if term_C_den > 0 else 0

                    # term D - topic article 
                    term_D_num = self.n_ik[i, k_prime] + self.alpha
                    term_D_den = M_i + (L_i - 1) + self.K * self.alpha
                    term_D = term_D_num / term_D_den if term_D_den > 0 else 0

                    sampling_probs[k_prime] = term_C * term_D 

                sum_probs = np.sum(sampling_probs)
                if sum_probs > 1e-9:
                    sampling_probs /= sum_probs 
                else:
                    sampling_probs = np.ones(self.K) / self.K 

                new_topic = np.random.choice(self.K, p=sampling_probs)
                self.z_h[i][l] = new_topic 

                # Increment coutns 
                self.n_ik[i, new_topic] += 1
                self.n_kv_h[new_topic, word_id] += 1
                self.n_k_h[new_topic] += 1 
        
    def estimate_parameters(self):
        if self.n_ik is None: 
            raise ValueError("Sampler state not initialized or run.")

        # Estimate pi_estiamte (N, K)
        self.pi_estimate = np.zeros((self.N, self.K))
        for i in range(self.N):
            total_words_in_article_i = np.sum(self.n_ik[i, :])
            if total_words_in_article_i + self.K * self.alpha > 0:
                self.pi_estimate[i, :] = (self.n_ik[i, :] + self.alpha) / (total_words_in_article_i + self.K * self.alpha)

            else:
                self.pi_estimate[i, :] = 1.0 / self.K 

        # Estimate theta_b_estimate (K, V)
        self.theta_b_estimate = np.zeros((self.K, self.V))
        denominator_b = self.n_k_b + self.V * self.beta1 
        for k in range(self.K):
            if denominator_b[k] > 0:
                self.theta_b_estimate[k, :] = (self.n_kv_b[k, :] + self.beta1) / denominator_b[k] 

            else:
                self.theta_b_estimate[k, :] = 1.0 / self.V 

        # Estimate theta_h_estimate (K, V)
        self.theta_h_estimate = np.zeros((self.K, self.V))
        denominator_h = self.n_k_h + self.V * self.beta2 
        for k in range(self.K):
            if denominator_h[k] > 0:
                self.theta_h_estimate[k, :] = (self.n_kv_h[k, :] + self.beta2) / denominator_h[k] 

            else:
                self.theta_h_estimate[k, :] = 1.0 / self.V 
        print("Parameters estimated from Gibbs sample")

    def train(self, num_iterations):
        if self.N == 0 or self.V == 0: 
            print("Data not loaded.")
            return 

        self.init_sampler_state()

        print("Starting Gibbws sampling...")
        for iter_num in range(num_iterations):
            print(f"Iteration {iter_num + 1}/{num_iterations}")
            self.run_gibbs_iter()

        print("Gibbs sampling finished.")
        print("Estimating parameters from the final sample...")
        self.estimate_parameters()


if __name__ == '__main__':
    K = 10 
    alpha = 1.0
    beta1 = 0.01 
    beta2 = 0.01 
    num_iters = 100 

    stopwards_file = "stopwords.txt"
    data = "articles.txt"

    gibbs_model = GibbsSampler(K, alpha, beta1, beta2)
    print("--- Loading stopwards ---")
    gibbs_model.load_stopwords(stopwards_file)
    print("--- Loading and processing data ---")
    gibbs_model.load_data(data)

    if gibbs_model.N > 0 and gibbs_model.V > 0:
        gibbs_model.train(num_iters)

        if gibbs_model.theta_b_estimate is not None:
            print("\nTop 10 words for each Body topic (from Gibbs sampler estimate):")
            for k in range(gibbs_model.K):
                top_indices_b = np.argsort(gibbs_model.theta_b_estimate[k, :])[::-1][:10]
                top_words_b = [gibbs_model.vocab[idx] for idx in top_indices_b]
                print(f"Topic {k+1}: {', '.join(top_words_b)}")

        if gibbs_model.theta_h_estimate is not None:
            print("\nTop 10 words for each Headline topic (from Gibbs sample estimates):")
            for k in range(gibbs_model.K):
                top_indices_h = np.argsort(gibbs_model.theta_h_estimate[k, :])[::-1][:10]
                top_words_h = [gibbs_model.vocab[idx] for idx in top_indices_h]
                print(f"Topic {k+1}: {', '.join(top_words_h)}")

    else:
        print("Model could not be trained due to no data or vocab")



