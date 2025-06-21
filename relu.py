import triton
import triton.language as tl

@triton.jit
def relu_kernel_2d(
    in_ptr, out_ptr,
    n, m,  # shape of the 2D tensor
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr
):

    pid_m = tl.program_id(0)  # lignes
    pid_n = tl.program_id(1)  # colonnes

    i = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]  # (BLOCK_M, 1)
    j = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]  # (1, BLOCK_N)


    mask = (i < n) & (j < m)
    offsets = i * m + j


    in_vals = tl.load(in_ptr + offsets, mask=mask, other=0.0)
    out_vals = tl.maximum(in_vals, 0.0)
    tl.store(out_ptr + offsets, out_vals, mask=mask)


def solution(input, output, n: int, m: int):
    BLOCK_M = 128
    BLOCK_N = 32

    grid = lambda meta: (
        triton.cdiv(n, BLOCK_M),
        triton.cdiv(m, BLOCK_N)
    )

    relu_kernel_2d[grid](
        input, output,
        n, m,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=4,
        num_stages=2
    )
