import multiprocessing
import os
import psutil


# Function for each process (replace with your actual processing logic)
def attn_output_process_half(start_idx, end_idx, process_index, num_processes):
    pid = os.getpid()
    print(
        f"Process {process_index} with PID {pid} working on indices {start_idx} to {end_idx}"
    )

    # Set process priority
    set_process_priority(pid, priority=-10)  # Example priority

    # Set CPU affinity based on the process index
    cpu = 48 + (process_index - 23) if process_index > 23 else process_index
    set_cpu_affinity(pid, [cpu])

    # Simulate processing (replace with actual workload)
    # Example workload here
    # ...


def set_process_priority(pid, priority):
    """Set process priority for the specified pid."""
    try:
        p = psutil.Process(pid)
        p.nice(priority)
    except Exception as e:
        print(f"Failed to set priority for process {pid}: {e}")


def set_cpu_affinity(pid, cpus):
    """Set CPU affinity for the specified pid."""
    try:
        p = psutil.Process(pid)
        p.cpu_affinity(cpus)
    except Exception as e:
        print(f"Failed to set CPU affinity for process {pid}: {e}")


# Define number of processes and work distribution
num_processes = 8  # Example process count
work_per_process = 10  # Example workload per process
total_work = 100  # Total workload

# Create a process pool
processes = []
for p in range(num_processes):
    start_idx = p * work_per_process
    end_idx = min(start_idx + work_per_process, total_work)

    # Create a new process
    process = multiprocessing.Process(
        target=attn_output_process_half, args=(start_idx, end_idx, p, num_processes)
    )
    process.start()

    # Add process to the list for tracking
    processes.append(process)

# Wait for all processes to complete
for process in processes:
    process.join()
