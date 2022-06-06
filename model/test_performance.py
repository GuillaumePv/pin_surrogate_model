
from deepsurrogate import *
#adding directory to path
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)

sys.path.append(parentdir)
from pin_model_simulation import *
from common import *

opt = Optimizer()

sim = 1000

num_cores = multiprocessing.cpu_count()
print(f"== number of CPU: {num_cores} ==")
start = time.time()
for i in tqdm(range(sim)):
    simulation(i,False)
end = time.time()
print(end-start)
