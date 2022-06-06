from model.deepsurrogate import DeepSurrogate
from tqdm import tqdm
deepsurrogate = DeepSurrogate()

df_pin = deepsurrogate.X[['buy','sell']]
df_pin = df_pin.head(3)

PIN_values = []
for i in tqdm(range(df_pin.shape[0])):
    pin = deepsurrogate.get_pin(df_pin.loc[i].values)
    PIN_values.append(pin)

df_pin['PIN'] = PIN_values
print("=== Save simulation result ===")
df_pin.to_latex("./results/table/simulation_result_pin.tex",index=False)
df_pin.to_csv("./results/table/simulation_result_pin.csv",index=False)
print(df_pin)