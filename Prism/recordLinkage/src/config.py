# =============================================================================
# Configuration file for the Record Linkage Project
# =============================================================================

# The length of q-grams to be used for Jaccard and Dice similarity
Q_GRAM_LENGTH = 2

# The number of trees to use in the Random Forest classifier
ML_N_ESTIMATORS = 10

# Logging configuration
LOG_LEVEL = 'INFO' # Can be 'DEBUG', 'INFO', 'WARNING', etc.
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'

# Flag to enable or disable GPU comparison
USE_GPU_COMPARISON = True
