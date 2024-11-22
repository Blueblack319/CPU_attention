#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <cstring>
#include <iostream>

extern "C" {

    // Function to check and modify a boolean in shared memory
    void toggle_shared_memory_bool(const char* name, size_t size) {
        // Open the shared memory
        int fd = shm_open(name, O_RDWR, 0666);
        if (fd == -1) {
            std::cerr << "Error opening shared memory" << std::endl;
            return;
        }

        // Map the shared memory
        void* ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (ptr == MAP_FAILED) {
            std::cerr << "Error mapping shared memory" << std::endl;
            return;
        }

        // Access the boolean (1 byte)
        uint8_t* bool_ptr = static_cast<uint8_t*>(ptr);
        std::cout << "C++: Current boolean value: " << (bool)(*bool_ptr) << std::endl;

        // Toggle the boolean value
        *bool_ptr = (*bool_ptr == 0) ? 1 : 0;
        std::cout << "C++: Toggled boolean value to: " << (bool)(*bool_ptr) << std::endl;

        // Clean up
        munmap(ptr, size);
        close(fd);
    }
}
