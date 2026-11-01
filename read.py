import pandas as pd


def read_datasets(path="3/creditcard.csv"):
    """
    csvからデータを読んでdataframeで返す。
    Timeの前80%を学習用、後ろ20%をテスト用とする
    """
    df = pd.read_csv(path)

    # NOTE: データセットはもともとTimeでソートされてる。
    rows_cnt = len(df)
    pos = int(rows_cnt * 0.8)
    return df.iloc[:pos], df[pos:]
