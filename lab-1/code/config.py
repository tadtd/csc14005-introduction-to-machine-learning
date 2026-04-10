import numpy as np

# Hyperparameters
LEARNING_RATE = 0.01
EPS = 1e-6
MAX_ITER = 10_000

# Seed
SEED = 42

def set_seed(seed: int = SEED) -> None:
  np.random.seed(seed)