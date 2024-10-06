// shared.h
#ifndef SHARED_H
#define SHARED_H

#include <atomic>
#include <condition_variable>
#include <mutex>

// Declare global synchronization variables
extern std::atomic<bool> ready_flag;        // Used to signal threads to start
extern std::atomic<bool> finished_flags[];  // Flags for thread completion

// Atomic counter to count completed threads
extern std::atomic<int> complete_counter;
#endif  // SHARED_H
