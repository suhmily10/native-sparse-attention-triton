# Optimize these parameters
BLOCK_SIZE_Q = 128  # Try different sizes: 64, 128, 256
BLOCK_SIZE_K = 64   # Experiment with 32, 64, 128
BLOCK_SIZE_D = triton.next_power_of_2(head_dim)
num_warps = 4 if head_dim <= 64 else 8  # Try 8/16 for larger dimensions
num_stages = 3  # Test with 2-4 stages