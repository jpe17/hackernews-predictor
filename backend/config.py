"""
Configuration file for the HackerNews Score Prediction project.
"""

# Data Constants
NUMBER_OF_SAMPLES = 4000000
MINIMUM_SCORE = 10
MAXIMUM_SCORE = 1000
MIN_TRESHOLD = 10000
MAX_AUGMENT_PER_BIN = 15000
TOTAL_BUDGET = 100000
NUM_DOMAINS = 200
NUM_USERS = 1000

# Random Seeds for reproducibility
RANDOM_STATE = 42

# Model Architecture Constants - SIMPLIFIED ARCHITECTURE (Numerical + Title)
TITLE_EMB_DIM = 200         # GloVe title embeddings (200D)
NUMERICAL_DIM = 36          # Enhanced: 34 original + domain_mean + user_mean

# Neural Network Configuration - SINGLE CONFIG ONLY
LEARNING_RATE = 3e-3        # Learning rate (increased for faster convergence)
HIDDEN_DIM = 128            # Hidden layer size
DROPOUT_RATE = 0.1          # Dropout rate
WEIGHT_DECAY = 1e-5         # Weight decay for regularization
PATIENCE = 25               # Early stopping patience (reduced for faster training)
MAX_EPOCHS = 100             # Maximum epochs to train (increased to allow for convergence)

# Training Configuration - Optimized for speed while maintaining convergence
VALIDATION_FREQUENCY = 2    # Validate every N epochs (reduced frequency for speed)
PRINT_FREQUENCY = 5         # Print progress every N epochs (more frequent for monitoring)
COMPUTE_R2_FREQUENCY = 5    # Compute R² every N epochs (more frequent for better monitoring)
WANDB_LOG_FREQUENCY = 5     # Log to wandb every N epochs (added missing variable)
VERBOSE_TRAINING = False    # Whether to show verbose training progress (added missing variable)
GRADIENT_CLIP_NORM = 1.0    # Gradient clipping max norm
SCHEDULER_FACTOR = 0.7      # LR scheduler reduction factor (less aggressive reduction)
SCHEDULER_PATIENCE = 10     # LR scheduler patience (reduced for faster adaptation)

# Performance Thresholds
TARGET_R2 = 0.2             # Target R² score for the model
CLOSE_PERFORMANCE_THRESHOLD = 0.003  # Threshold for "close" performance comparison

# Data Split Constants
VAL_SIZE = 0.2              # 20% for test set
TEST_SIZE = 0.25            # 25% of remaining for validation (so 60% train, 20% val, 20% test)

# File Paths
DATA_PATH = "data/hackernews_full_data.parquet"
GLOVE_FILE = "data/glove.6B.200d.txt"
MODEL_PATH = f"artifacts/best_numerical_title_model.pth"
MODEL_CONFIG_PATH = f"artifacts/model_config.pkl"
SCALER_PATH = f"artifacts/scaler.pkl"