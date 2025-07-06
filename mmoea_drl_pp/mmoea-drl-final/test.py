from tqdm import tqdm
import time

for i in tqdm(range(10)):
    # Perform some task
    time.sleep(0.1)
    tqdm.write("message: Reached iteration".format(i))
