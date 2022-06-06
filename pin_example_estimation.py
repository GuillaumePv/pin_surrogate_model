from model.deepsurrogate import DeepSurrogate

deepsurrogate = DeepSurrogate()

df_b_s = deepsurrogate.X[['buy','sell']]
df_b_s = df_b_s.head(3)

for i in range(df_b_s.shape[0]):
    pin = deepsurrogate.get_pin(df_b_s.loc[i].values)
    print(pin)