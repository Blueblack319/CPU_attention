from multiprocessing import shared_memory
import ctypes

# Load the C++ shared library
lib = ctypes.CDLL('./libshared_memory.so')

# Define the C++ function's signature
lib.toggle_shared_memory_bool.argtypes = [ctypes.c_char_p, ctypes.c_size_t]

# Create shared memory in Python
SHARED_MEM_NAME = "shared_bool"
SHARED_MEM_SIZE = 1  # 1 byte for a boolean

# Create a shared memory block
shm = shared_memory.SharedMemory(create=True, name=SHARED_MEM_NAME, size=SHARED_MEM_SIZE)
buffer = shm.buf

# Initialize the shared memory with a boolean value (False)
buffer[0] = 0
print(f"Python: Initialized shared memory with boolean value: {bool(buffer[0])}")

# Call the C++ function to toggle the boolean
lib.toggle_shared_memory_bool(SHARED_MEM_NAME.encode('utf-8'), SHARED_MEM_SIZE)

# Read the modified value from shared memory
modified_value = bool(buffer[0])
print(f"Python: Read modified boolean value from shared memory: {modified_value}")

# Call the C++ function again to toggle the boolean back
lib.toggle_shared_memory_bool(SHARED_MEM_NAME.encode('utf-8'), SHARED_MEM_SIZE)

# Read the modified value from shared memory
modified_value = bool(buffer[0])
print(f"Python: Read modified boolean value from shared memory: {modified_value}")

# Cleanup
shm.close()
shm.unlink()
