# %%
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('fivethirtyeight')

from model.deepsurrogate import DeepSurrogate
from tqdm import tqdm
deepsurrogate = DeepSurrogate()

columns_pin = ["buy","sell"]

#buy_and_sell = buy_and_sell.head(50)



# %%
dataset = ["parro","vetropak","vontonbel","sig","vaudoise","bcv"]
for name in tqdm(dataset):

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
        #print(pin)
        PIN_values.append(pin)

    buy_and_sell['PIN'] = PIN_values
    buy_and_sell.to_csv("test_results.csv")
    #buy_and_sell[["buy","sell"]].plot()

    # %%
    close_price = final_merged[['Timestamp','midpoint','bid-ask spread']]
    close_price.index = close_price['Timestamp']
    close_price = close_price.iloc[:,1:]
    close_price = close_price.resample('D').mean()
    close_price = close_price.dropna()
    graph_merge = pd.merge(close_price,buy_and_sell,left_index=True,right_index=True)
    graph_merge["buy-sell difference"] = np.abs(graph_merge['buy'] - graph_merge["sell"])

    # %%
    fig,ax = plt.subplots(figsize=(10,5))
    l1, = ax.plot(graph_merge.index,graph_merge.midpoint)
    ax.set_ylabel(r"Close price")
    ax2 = ax.twinx()
    l2, = ax2.plot(graph_merge.index, graph_merge.PIN,color="orange")
    ax2.set_ylabel(r"PIN value")

    plt.legend([l1, l2],["Close price", "PIN"])
    plt.xlabel(r"Date")
    plt.tight_layout()
    plt.grid(False)
    plt.savefig(folder_results+"/close_price_pin.png")
    plt.close()

    # %%
    fig,ax = plt.subplots(figsize=(8,5))
    l1, = ax.plot(graph_merge.index,graph_merge["bid-ask spread"])
    ax.set_ylabel(r"bid-ask spread")
    ax2 = ax.twinx()
    l2, = ax2.plot(graph_merge.index, graph_merge.PIN,color="orange")
    ax2.set_ylabel(r"PIN value")

    plt.legend([l1, l2],["bid-ask spread", "PIN"])
    plt.xlabel(r"Date")
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(folder_results+"/bid_ask_spread_pin.png")
    plt.close()


    # %%
    fig,ax = plt.subplots(figsize=(8,3))
    l1, = ax.plot(graph_merge.index,graph_merge["buy-sell difference"])
    ax.set_ylabel(r"buy-sell absolute difference")
    ax2 = ax.twinx()
    l2, = ax2.plot(graph_merge.index, graph_merge.PIN,color="orange")
    ax2.set_ylabel(r"PIN value")

    plt.legend([l1, l2],["buy-sell difference", "PIN"])
    plt.xlabel(r"Date")
    plt.tight_layout()
    plt.grid(False)
    plt.savefig(folder_results+"/buy_sell_diff_pin.png")
    plt.close()

# %%



