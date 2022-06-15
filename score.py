from model.deepsurrogate import DeepSurrogate
import pandas as pd
import time

start = time.time()
deepsurrogate = DeepSurrogate()
print("=== compute score of model ===")
print("=== wait some time ===")
score = deepsurrogate.c_model.score(deepsurrogate.X,deepsurrogate.Y)
print(score)
end = time.time()
print("Time taken: " + (end-start))
score.to_csv('./score_'+ deepsurrogate.par_c.name +".csv")