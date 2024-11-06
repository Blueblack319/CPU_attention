import threading
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.pool import ThreadPool
import time



# Sample task for each thread
def task(data0, data1, data2, data3, data4, data5, data6, data7, data8, data9, data10, data11, data12, data13, ready_flag):
    # Busy-wait until the ready_flag is set
    while not ready_flag.is_set():
        pass  # Keep checking the flag status without releasing CPU
    
    # return data[0] + data[1]
    return data0 + data1 + data2 + data3 + data4 + data5 + data6 + data7 + data8 + data9 + data10 + data11 + data12 + data13

# Main function to control the threads
def test_thread_pool():
    # Initialize the ready_flag as a threading.Event
    ready_flag = threading.Event()
    # Number of threads
    thread_num = 10
    data_list = [(i+1, i+2, i+1, i+2, i+1, i+2, i+1, i+2, i+1, i+2, i+1, i+2, i+1, i+2, ready_flag) for i in range(thread_num)]

    # Initialize ThreadPool
    with ThreadPool(thread_num) as pool:
        # Submit tasks to ThreadPool
        result = pool.starmap_async(task, data_list)

        print("Main thread sleeping for 2 seconds before setting ready_flag...")
        time.sleep(2)  # Simulate some setup time

        # Signal all threads to proceed
        ready_flag.set()
        print("Main thread set ready_flag, all threads should proceed.")

        for result in result.get():
            print(f"Got result: {result}")


if __name__ == "__main__":
    test_thread_pool()

