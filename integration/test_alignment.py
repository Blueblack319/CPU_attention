import numpy as np
import ctypes

def aligned_array(size, dtype=np.float32, alignment=32):
    # Determine the number of bytes per element
    itemsize = np.dtype(dtype).itemsize
    # Allocate raw memory with alignment
    buf = ctypes.create_string_buffer(size * itemsize + alignment)
    start = ctypes.addressof(buf)
    offset = (alignment - start % alignment) % alignment
    # Create a numpy array that views this aligned memory
    aligned_array = np.frombuffer(buf, dtype=dtype, count=size, offset=offset)
    
    return aligned_array

# Usage
size = 1024
array = aligned_array(size, dtype=np.float32, alignment=32)

# Verify alignment
if array.ctypes.data % 32 == 0:
    print("Array is correctly aligned to 32 bytes.")
else:
    print("Array is not aligned.")
