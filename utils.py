# utils.py
import torch
import numpy as np
import random


###########################################################################################################

def set_seed(seed=42):  # Default seed value is 42
    random.seed(seed)  # Set the seed for Python's built-in random module
    np.random.seed(seed)  # Set the seed for NumPy's random number generator
    torch.manual_seed(seed)  # Set the seed for PyTorch's CPU random number generator

    # Check if CUDA (GPU support) is available and set the seed for CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # Set the seed for the current GPU
        torch.cuda.manual_seed_all(seed)  # Set the seed for all available GPUs
 
    torch.backends.cudnn.deterministic = True # Make the CuDNN backend deterministic to ensure consistent results
    torch.backends.cudnn.benchmark = False # Disable CuDNN's auto-tuner to avoid non-deterministic behavior
    
###########################################################################################################