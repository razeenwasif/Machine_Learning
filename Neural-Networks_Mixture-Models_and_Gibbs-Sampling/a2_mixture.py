import numpy as np
import string

class MyMixtureModel:

    def __init__(self, K):
        ################
        #YOUR CODE HERE#
        ################        
        self.K = K 
        self.N = 0 
        self.V = 0 

        # Model params 
        self.pi = None 
        self.theta_b = None 
        self.theta_h = None 

        # Data storage
        self.bodies_data_ids = [] # word ids for bodies 
        self.headlines_data_ids = [] # word ids for headlines 

        # For E=step 
        self.gamma_b = [] 
        self.gamma_h = [] 
        self.stopwards = [] 
        self.word2id = {} 
        self.vocab = []

    def load_stopwords(self, file_name):
        """
        This function stores the stopwords in a given file into the stopword list.
        """
        # The following code is provided and does not need to be modified.
        self.stopwords = []
        with open(file_name, 'r') as input_file:
            for line in input_file:
                line = line.rstrip()
                self.stopwords.append(line)

    def load_data(self, file_name):
        """
        This function processes the input data file and stores the words in the bodies and the headlines
        of the news articles. 
        """
        
        ################
        #YOUR CODE HERE#
        ################

        # Modify the code below to store the words in the bodies and the headlines (separately).
        # The words should be stored as IDs.
        # Starting code has been provided to read the input data file, split the body or the headline into
        # individual words, remove the stop words, and map the words into unique IDs.


        # This dictionary maps words to IDs (starting from 0).
        self.word2id = {}
        # This list stores the words corresponding to the IDs, e.g., self.vocab[0] stores the word whose
        # ID is 0.
        self.vocab = []
        bodies_as_words = []
        headlines_as_words = []

        with open(file_name, 'r') as input_file:
            for line in input_file:
                line = line.rstrip()
                columns = line.split('\t')
                if len(columns) < 3:
                    continue
                body,headline,category = columns[0], columns[1], columns[2]
                # Turn the body of the article into a list of words in lowercase
                body = body.lower().translate(str.maketrans('', '', string.punctuation)).split() 
                # Turn the headline of the article into a list of words in lowercase
                headline = headline.lower().translate(str.maketrans('', '', string.punctuation)).split()
                bodies_as_words.append(body)
                headlines_as_words.append(headline)
        
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


    def initialise_parameters(self):
        ################
        #YOUR CODE HERE#
        ################
        if self.N == 0 or self.V == 0:
            print("Data not loaded properly or vocab/doc count is zero")
            return 
        
        # init pi - article specific topic mixtures
        self.pi = np.random.rand(self.N, self.K) + 1e-9 # epsilon for stability
        self.pi = self.pi / np.sum(self.pi, axis=1, keepdims=True)
        
        # init theta b - body word distributions per topic
        self.theta_b = np.random.rand(self.K, self.V) + 1e-9 
        self.theta_b = self.theta_b / np.sum(self.theta_b, axis=1, keepdims=True)
        
        # init theta h - headline word distributions per topic 
        self.theta_h = np.random.rand(self.K, self.V) + 1e-9 
        self.theta_h = self.theta_h / np.sum(self.theta_h, axis=1, keepdims=True)
        
        # init gamma lists 
        self.gamma_b = [np.zeros((len(doc_words), self.K)) if doc_words else np.array([]) for doc_words in self.bodies_data_ids]
        self.gamma_h = [np.zeros((len(doc_words), self.K)) if doc_words else np.array([]) for doc_words in self.headlines_data_ids]

    def run_E_step(self):
        ################
        #YOUR CODE HERE#
        ################
        if self.pi is None or self.theta_b is None or self.theta_h is None:
            raise ValueError("Parameters not initialized.")

        for i in range(self.N):
            # Body words
            if self.bodies_data_ids[i]: # if body not empty
                M_i = len(self.bodies_data_ids[i])
                numerator_b_terms = np.zeros((M_i, self.K))
                for j, word_id in enumerate(self.bodies_data_ids[i]):
                    numerator_b_terms[j, :] = self.pi[i, :] * self.theta_b[:, word_id]
                
                denominator_b = np.sum(numerator_b_terms, axis=1, keepdims=True)
                denominator_b[denominator_b < 1e-9] = 1e-9 # Avoid division by zero
                self.gamma_b[i] = numerator_b_terms / denominator_b
            else:
                self.gamma_b[i] = np.array([]) # Empty array if no body words

            # Headline words
            if self.headlines_data_ids[i]: # Check if headline is not empty
                L_i = len(self.headlines_data_ids[i])
                numerator_h_terms = np.zeros((L_i, self.K))
                for l, word_id in enumerate(self.headlines_data_ids[i]):
                     numerator_h_terms[l, :] = self.pi[i, :] * self.theta_h[:, word_id]

                denominator_h = np.sum(numerator_h_terms, axis=1, keepdims=True)
                denominator_h[denominator_h < 1e-9] = 1e-9
                self.gamma_h[i] = numerator_h_terms / denominator_h
            else:
                self.gamma_h[i] = np.array([])

    def run_M_step(self):
        ################
        #YOUR CODE HERE#
        ################
        if not any(g.size > 0 for g in self.gamma_b) and not any(g.size > 0 for g in self.gamma_h):
             print("Gamma values are empty or not computed. Skipping M-step.")
             return

        # Update pi
        for i in range(self.N):
            sum_gamma_b_for_article_i = np.sum(self.gamma_b[i], axis=0) if self.gamma_b[i].size > 0 else np.zeros(self.K)
            sum_gamma_h_for_article_i = np.sum(self.gamma_h[i], axis=0) if self.gamma_h[i].size > 0 else np.zeros(self.K)
            
            M_i = self.gamma_b[i].shape[0] if self.gamma_b[i].size > 0 else 0
            L_i = self.gamma_h[i].shape[0] if self.gamma_h[i].size > 0 else 0
            
            total_words_in_article_i = M_i + L_i
            if total_words_in_article_i > 0:
                self.pi[i, :] = (sum_gamma_b_for_article_i + sum_gamma_h_for_article_i) / total_words_in_article_i
            else: # Handle articles with no words
                self.pi[i, :] = 1.0 / self.K # Re-initialize to uniform if article became empty

        # Update theta_b 
        new_theta_b_numerator = np.zeros((self.K, self.V)) + 1e-9 # Laplace smoothing
        new_theta_b_denominator = np.zeros(self.K) + (self.V * 1e-9) # For smoothing

        for k in range(self.K):
            for i in range(self.N):
                if self.bodies_data_ids[i] and self.gamma_b[i].size > 0:
                    for j, word_id in enumerate(self.bodies_data_ids[i]):
                        new_theta_b_numerator[k, word_id] += self.gamma_b[i][j, k]
                    new_theta_b_denominator[k] += np.sum(self.gamma_b[i][:, k])
        
        self.theta_b = new_theta_b_numerator / new_theta_b_denominator[:, np.newaxis]


        # Update theta_h 
        new_theta_h_numerator = np.zeros((self.K, self.V)) + 1e-9 
        new_theta_h_denominator = np.zeros(self.K) + (self.V * 1e-9) 

        for k in range(self.K):
            for i in range(self.N):
                if self.headlines_data_ids[i] and self.gamma_h[i].size > 0:
                    for l, word_id in enumerate(self.headlines_data_ids[i]):
                        new_theta_h_numerator[k, word_id] += self.gamma_h[i][l, k]
                    new_theta_h_denominator[k] += np.sum(self.gamma_h[i][:, k])

        self.theta_h = new_theta_h_numerator / new_theta_h_denominator[:, np.newaxis]
        
    def compute_log_likelihood(self):
        ################
        #YOUR CODE HERE#
        ################
        log_likelihood = 0.0
        if self.pi is None or self.theta_b is None or self.theta_h is None:
            print("Parameters not initialized for log-likelihood computation.")
            return -np.inf

        for i in range(self.N):
            article_ll = 0.0
            # Body log-likelihood
            if self.bodies_data_ids[i]:
                # P(word_id | pi_i, theta_b) = sum_k pi_ik * theta_k_word_id
                # log P(word_id | pi_i, theta_b)
                # For each word in body_i
                for word_id in self.bodies_data_ids[i]:
                    prob_word_b_k = self.pi[i, :] * self.theta_b[:, word_id]
                    prob_word_b = np.sum(prob_word_b_k)
                    if prob_word_b > 1e-12:
                        article_ll += np.log(prob_word_b)
                    else:
                        article_ll += np.log(1e-12) # Avoid -inf

            # Headline log-likelihood
            if self.headlines_data_ids[i]:
                for word_id in self.headlines_data_ids[i]:
                    prob_word_h_k = self.pi[i, :] * self.theta_h[:, word_id]
                    prob_word_h = np.sum(prob_word_h_k)
                    if prob_word_h > 1e-12:
                        article_ll += np.log(prob_word_h)
                    else:
                        article_ll += np.log(1e-12)
            log_likelihood += article_ll
        return log_likelihood

    def train(self, num_iterations = 100, threshold=0.001):
        # This function has been implemented for you and does not need to be modified.
        old_log_likelihood = None
        for t in range(num_iterations):
            self.run_E_step()
            self.run_M_step()
            log_likelihood = self.compute_log_likelihood()
            print("Iteration " + str(t) + ": " + str(log_likelihood))
            if (old_log_likelihood != None and log_likelihood - old_log_likelihood < threshold):
                break
            old_log_likelihood = log_likelihood


# You can set K to different values
K = 5
num_iterations = 50
threshold = 0.001

stopwords_file = "stopwords.txt"
data_file = "articles.txt"

model = MyMixtureModel(K)
model.load_stopwords(stopwords_file)
model.load_data(data_file)
model.initialise_parameters()
model.train(num_iterations, threshold)

# Write code to print out the top-10 words with the highest probabilities of each topic
print("\nTop 10 words for each BODY topic:")
for k in range(model.K):
    top_indices_b = np.argsort(model.theta_b[k, :])[::-1][:10]
    top_words_b = [model.vocab[idx] for idx in top_indices_b]
    print(f"Topic {k+1}: {', '.join(top_words_b)}")

print("\nTop 10 words for each HEADLINE topic:")
for k in range(model.K):
    top_indices_h = np.argsort(model.theta_h[k, :])[::-1][:10]
    top_words_h = [model.vocab[idx] for idx in top_indices_h]
    print(f"Topic {k+1}: {', '.join(top_words_h)}")
