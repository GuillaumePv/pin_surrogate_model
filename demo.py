# %%
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('fivethirtyeight')

import seaborn as sns
from model.deepsurrogate import DeepSurrogate
from tqdm import tqdm
deepsurrogate = DeepSurrogate()

columns_pin = ["buy","sell"]

# %%

if os.path.isfile("./results/PIN_real.txt") == False:
    print("=== creating real file ===")
    f = open("./results/PIN_real.txt", "a")
    f.write("PIN\n")
    f.close()
dataset = ["parro","vetropak","vontonbel","sig","vaudoise","bcv"]
for name in tqdm(dataset):
    print(f"=== algo for {name} ===")
    path = f"./data/data_{name}.xlsx"
    folder_results = './results/'+name
    if os.path.exists(folder_results) == False:
        os.makedirs(folder_results)

    trade = pd.read_excel(path,sheet_name="trade_price")

    # %%
    bid = pd.read_excel(path,sheet_name="bid")

    # %%
    ask = pd.read_excel(path,sheet_name="ask")


    # %%
    bid_ask_merge = pd.merge(bid,ask,on=["Timestamp"])


    # %%
    bid_ask_merge["midpoint"]= (bid_ask_merge["Bid Close"]+bid_ask_merge["Ask Close"])/2


    # %%
    final_merged = pd.merge(bid_ask_merge,trade,on=["Timestamp"])
    final_merged["bid-ask spread"] = final_merged["Ask Close"] - final_merged["Bid Close"]


    # %%
    final_merged['direction_trade'] = np.where(final_merged['Trade Close']>= final_merged['midpoint'], "buy", "sell")

    # %%
    data_final = final_merged[["Timestamp","Trade Close","Trade Count","direction_trade","midpoint"]]

    # %%

    # %%
    test = data_final.copy()
    test["Timestamp"] = pd.to_datetime(test["Timestamp"]) # only if it isn't already
    test = test.set_index("Timestamp")
    agre_data = test.groupby("direction_trade").resample('D').sum()

    # %%
    buy = agre_data["Trade Count"].loc["buy"].T
    sell = agre_data["Trade Count"].loc["sell"].T

    # %%

    columns = ["#buys","#sells"]
    buy_and_sell = pd.merge(buy,sell,on=["Timestamp"])
    buy_and_sell.columns = columns
    buy_and_sell = buy_and_sell.loc[(buy_and_sell!=0).any(axis=1)]

    # %%
    buy_and_sell.columns = columns_pin
    PIN_values = []
    for i in tqdm(range(buy_and_sell.shape[0])):
        pin = deepsurrogate.get_pin(buy_and_sell.iloc[i].values)
        f = open("./results/PIN_real.txt", "a")
        f.write(f"{pin}\n")
        f.close()
        #print(pin)
        PIN_values.append(pin)

    buy_and_sell['PIN'] = PIN_values
    #buy_and_sell[["buy","sell"]].plot()

    # %%
    close_price = final_merged[['Timestamp','midpoint','bid-ask spread','Bid Close','Ask Close']]
    close_price.index = close_price['Timestamp']
    close_price = close_price.iloc[:,1:]
    close_price = close_price.resample('D').mean()
    close_price = close_price.dropna()
    graph_merge = pd.merge(close_price,buy_and_sell,left_index=True,right_index=True)
    graph_merge["buy-sell difference"] = np.abs(graph_merge['buy'] - graph_merge["sell"])

    fig,ax = plt.subplots(figsize=(10,5))
    l1, = ax.plot(graph_merge.index,graph_merge["Bid Close"],label="Bid")
    ax.set_ylabel(r"Price (Bid & Ask)",fontsize=12)
    l2 = ax.plot(graph_merge.index,graph_merge["Ask Close"],label="Ask")
    ax2 = ax.twinx()
    l3, = ax2.plot(graph_merge.index, graph_merge.PIN,color="orange",label="PIN")
    ax2.set_ylabel(r"PIN value",fontsize=12)
    ax.legend(loc="lower left")
    ax2.legend(loc="upper left")
    plt.xlabel(r"Date")
    plt.tight_layout()

    plt.grid(False)
    plt.savefig(folder_results+f"/{name}_bid_ask_pin_evo.png")
    plt.close()
    # %%
    fig,ax = plt.subplots(figsize=(10,5))
    l1, = ax.plot(graph_merge.index,graph_merge.midpoint)
    ax.set_ylabel(r"Close price",fontsize=12)
    ax2 = ax.twinx()
    l2, = ax2.plot(graph_merge.index, graph_merge.PIN,color="orange")
    ax2.set_ylabel(r"PIN value",fontsize=12)

    plt.legend([l1, l2],["Close price", "PIN"],bbox_to_anchor=(0.6, -0.1))
    plt.xlabel(r"Date")
    plt.tight_layout()
    plt.grid(False)
    plt.savefig(folder_results+f"/{name}_close_price_pin.png")
    plt.close()

    # %%
    fig,ax = plt.subplots(figsize=(10,5))
    l1, = ax.plot(graph_merge.index,graph_merge["bid-ask spread"])
    ax.set_ylabel(r"bid-ask spread",fontsize=12)
    ax2 = ax.twinx()
    l2, = ax2.plot(graph_merge.index, graph_merge.PIN,color="orange")
    ax2.set_ylabel(r"PIN value",fontsize=12)

    plt.legend([l1, l2],["bid-ask spread", "PIN"],bbox_to_anchor=(0.6, -0.1))
    plt.xlabel(r"Date")
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(folder_results+f"/{name}_bid_ask_spread_pin.png")
    plt.close()


    # problem here
    fig,ax = plt.subplots(figsize=(10,5))
    l1, = ax.plot(graph_merge.index,graph_merge["buy-sell difference"])
    ax.set_ylabel(r"buy-sell absolute difference",fontsize=15)
    ax2 = ax.twinx()
    l2, = ax2.plot(graph_merge.index, graph_merge.PIN,color="orange")
    ax2.set_ylabel(r"PIN value",fontsize=15)

    plt.legend([l1, l2],["buy-sell difference", "PIN"],bbox_to_anchor=(0.6, -0.1))
    plt.xlabel(r"Date")
    plt.tight_layout()
    plt.grid(False)
    plt.savefig(folder_results+f"/{name}_buy_sell_diff_pin.png")
    plt.close()

    sns.histplot(graph_merge["PIN"],kde=True)
    plt.tight_layout()
    plt.savefig(folder_results+f"/{name}_dist_pin.png")
    plt.close()

    plt.boxplot(graph_merge["PIN"])
    plt.xticks([1], [name])
    plt.ylabel(r"PIN value",fontsize=12)
    plt.tight_layout()
    plt.savefig(folder_results+f"/{name}_boxplot_pin.png")
    plt.close()

    graph_merge["PIN"].describe().to_latex(folder_results+f"/{name}_pin_stat_desc.tex")

### analyse of the total results ###

df = pd.read_csv("./results/PIN_real.txt")
df.describe().to_latex("./results/table/stat_descrip_PIN_real.tex")




