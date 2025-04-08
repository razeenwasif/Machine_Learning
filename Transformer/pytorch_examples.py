import torch 
import numpy as np 
import time 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device) 


start_time = time.time()
# matrix operations here 
zeros = torch.zeros(1,1)
end_time = time.time()

elapsed_time = end_time - start_time
print(f"{elapsed_time: .8f}")

########### Probability Tensor ####################################

probabilities = torch.tensor([0.1, 0.9])
# 10% or 0.1 => 0, 90% or 0.9 => 1. Each probability points to the
# index of the probability in the tensor
# Draw 5 samples from the multinomial distribution
samples = torch.multinomial(probabilities, num_samples=10, replacement=True)
print(samples)
# 90% chance of getting 1 (since 0.9 is idx 1)

###################################################################

# timestamp - 50:41
