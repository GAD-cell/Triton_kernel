import triton
import triton.language as tl


@triton.jit
def add_kernel(
    in1_ptr,
    in2_ptr,
    out_ptr,
    n,
    BLOCK_SIZE : tl.constexpr
    ):

    PID = tl.program_id(0)

    start = BLOCK_SIZE*PID
    elements = tl.arange(0,BLOCK_SIZE) + start
    mask = elements < n

    a = tl.load(in1_ptr + elements,mask = mask, other=0.0)
    b = tl.load(in2_ptr + elements,mask = mask, other=0.0)

    out =  a + b
    tl.store(out_ptr + elements,out,mask = mask)

# Note: d_input1, d_input2, d_output are all float32 device tensors
def solution(d_input1, d_input2, d_output, n: int):
  
    grid = lambda meta : (triton.cdiv(n,meta['BLOCK_SIZE']),)

    add_kernel[grid](
        d_input1,
        d_input2,
        d_output,
        n,
        BLOCK_SIZE = 1024
    )
