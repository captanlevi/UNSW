import multiprocessing
import time

start = time.perf_counter()
def doSomething():
    time.sleep(1)
    print("done sleeping")

processes = []
for _ in range(10):
    p = multiprocessing.Process(target= doSomething)
    p.start()
    processes.append(p)

for p in processes:
    print("hello")
    p.join()
    

end = time.perf_counter()

print(f'Finished in {round(end- start,2)} seconds')