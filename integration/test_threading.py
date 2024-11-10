import threading
import os
import psutil

# Sample function for each thread (replace with actual logic)
def attn_output_threaded_half(*args):
    # Example of work done by the thread
    print(f"Thread {args[-3]} working on indices {args[-2]} to {args[-1]}")

def set_thread_priority(pid, priority):
    """Attempt to set process priority for the specified pid."""
    try:
        p = psutil.Process(pid)
        p.nice(priority)
    except Exception as e:
        print(f"Failed to set priority: {e}")

def set_cpu_affinity(pid, cpus):
    """Attempt to set CPU affinity for the specified pid."""
    try:
        p = psutil.Process(pid)
        p.cpu_affinity(cpus)
    except Exception as e:
        print(f"Failed to set CPU affinity: {e}")

# Initialize threads
num_threads = 8  # Example thread count
work_per_thread = 10  # Example workload per thread
total_work = 100  # Total workload

threads = []
priority = -10  # Example priority; ranges vary by system

for t in range(num_threads):
    start_idx = t * work_per_thread
    end_idx = min(start_idx + work_per_thread, total_work)
    
    # Create and start the thread
    thread = threading.Thread(
        target=attn_output_threaded_half,
        args=(None, None, None, None, None, None, None, None, None, None, None, None, None, t, num_threads, start_idx, end_idx)
    )
    thread.start()
    
    # Set priority and CPU affinity after the thread starts
    thread_pid = psutil.Process(thread.native_id)  # Get native thread ID for affinity

    # Set thread priority (try-except due to cross-platform compatibility)
    set_thread_priority(thread_pid.pid, priority)
    
    # Set CPU affinity based on `t`
    if t > 23:
        cpu = 48 + (t - 23)
    else:
        cpu = t
    set_cpu_affinity(thread_pid.pid, [cpu])
    
    threads.append(thread)

# Wait for all threads to complete
for thread in threads:
    thread.join()
