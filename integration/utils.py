import psutil


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
