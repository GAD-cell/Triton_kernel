import triton
import triton.language as tl


@triton.jit
def Conv1D_kernel(
    A_ptr, B_ptr, C_ptr,
    N, K,
    BLOCK_K: tl.constexpr 
):
    pid = tl.program_id(0)  


    offsets = tl.arange(0, BLOCK_K) - (K // 2)

    indices = pid + offsets


    valid_mask = (indices >= 0) & (indices < N)


    safe_indices = tl.where(valid_mask, indices, 0)
    a_ptrs = A_ptr + safe_indices
    a_vals = tl.load(a_ptrs, mask=valid_mask, other=0.0)


    b_ptrs = B_ptr + tl.arange(0, BLOCK_K)
    b_mask = tl.arange(0, BLOCK_K) < K  
    b_vals = tl.load(b_ptrs, mask=b_mask, other=0.0)


    c = tl.sum(a_vals * b_vals)
    tl.store(C_ptr + pid, c)


    


import math
# Note: A, B, C are all float32 device tensors
def solution(A, B, C, N: int, K: int):
  
    assert K % 2 == 1, "Le filtre K doit être impair."
    BLOCK_K = 2 ** math.ceil(math.log2(K)) 

    grid = lambda meta: (N,)

    Conv1D_kernel[grid](
        A, B, C,
        N, K,
        BLOCK_K=BLOCK_K
    )
