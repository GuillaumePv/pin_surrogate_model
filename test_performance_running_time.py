from model.deepsurrogate import DeepSurrogate

from pin_model_simulation import *
from common import *
import matplotlib.pyplot as plt
from matplotlib import style
style.use('fivethirtyeight')

deepsurrogate = DeepSurrogate()

nums = [10,100,1000]
df = deepsurrogate.X
COL = ["alpha","delta","epsilon_b","epsilon_s","mu","buy","sell"]

data_perf = {
    "PIN likelihood":[],
    'Surrogate likelihood':[]
}
for n in nums:
    print(f"=== number of simulations: {n} ===")
    df_num = df[COL].head(n)
    start_m = time.time()
    deepsurrogate.c_model.predict(df_num)
    end_m = time.time()
    duration_m = end_m - start_m
    data_perf['Surrogate likelihood'].append(duration_m)

    start_b = time.time()
    for i in range(df_num.shape[0]):
        v = df_num.iloc[i].values
        array_MLE = ll(v[0],v[1],v[2],v[3],v[4],pd.Series(v[5]),pd.Series(v[6]))
        MLE = logsumexp(array_MLE,axis=0)[0]
        
    end_b = time.time()
    duration_b = end_b-start_b
    data_perf['PIN likelihood'].append(duration_b)


df_result = pd.DataFrame(data_perf,index=nums)
df_result.to_latex("./results/table/speed_perf_comparison.tex")
df_result.plot()
plt.title(r"PIN vs Surrogate (Running time)")
plt.ylabel(r"Time taken in sec")
plt.xlabel(r"Number of simulations")
plt.tight_layout()
plt.savefig("./results/graphs/speed_perf_comparison.png")
plt.close()

