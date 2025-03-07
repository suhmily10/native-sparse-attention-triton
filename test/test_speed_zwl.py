import torch
import time
from flash_attn import flash_attn_func,flash_attn_varlen_func

# Constants
B = 16       # Batch size
L = 8192     # Sequence length
NH = 32      # Number of attention heads
NH_KV = 8    # Number of key/value heads (for grouped query attention)
HD = 128     # Head dimension

BLOCK_SIZE = 64
SLIDING_WINDOW_SIZE = 512
TOPK=16

def bench(name, func, num_warmup=2, num_runs=10):
    for _ in range(num_warmup):
        out = func()
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_runs):
        out = func()
    torch.cuda.synchronize()
    print(f"{name}: {(time.time() - start) / num_runs * 1000:.2f} ms per run")
    return out

def stage1_func(q, k, v):
    k = k.view(B, L//BLOCK_SIZE, BLOCK_SIZE, NH_KV, HD).mean(dim=2).view(B, L//BLOCK_SIZE, NH_KV, HD)
    v = v.view(B, L//BLOCK_SIZE, BLOCK_SIZE, NH_KV, HD).mean(dim=2).view(B, L//BLOCK_SIZE, NH_KV, HD)
    stage_1_output, lse, attn_probs = flash_attn_func(q, k, v, causal=True, return_attn_probs=True)
    attn_probs = torch.log(attn_probs) - lse.unsqueeze(-1)
    topk = torch.topk(attn_probs, k=TOPK, dim=-1)

    return stage_1_output, topk

def stage2_func(q, k, v, topk):
    pass


if __name__ == "__main__":
    q = torch.randn(B, L, NH, HD, device="cuda", dtype=torch.float16)
    k = torch.randn(B, L, NH_KV, HD, device="cuda", dtype=torch.float16)
    v = torch.randn(B, L, NH_KV, HD, device="cuda", dtype=torch.float16)

    baseline = lambda: flash_attn_func(q, k, v, causal=True)
    bench("baseline", baseline)

    # stage1 = lambda: stage1_func(q, k, v)
    # _, topk = bench("stage 1", stage1)

    # stage2 = lambda: stage2_func(q, k, v, topk)
    # bench("stage 2", stage2)

    # stage3 = lambda: flash_attn_func(q, k, v, causal=True, window_size=(SLIDING_WINDOW_SIZE, -1))
    
    # Create cumulative sequence lengths for fixed-length sequences
    cu_seqlens_q = torch.arange(0, (B + 1) * L, step=L, dtype=torch.int32, device="cuda")
    cu_seqlens_k = cu_seqlens_q.clone()

    stage4 = lambda: flash_attn_varlen_func(
        q.reshape(-1, NH, HD),
        k.reshape(-1, NH_KV, HD),
        v.reshape(-1, NH_KV, HD),
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=L,
        max_seqlen_k=L,
        causal=True,
        window_size=(SLIDING_WINDOW_SIZE, -1),
        dropout_p=0.0
    )
    bench("stage 4", stage4)
