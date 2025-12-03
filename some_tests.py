import pandas as pd

result_df = pd.DataFrame(columns=["Class_Pair", "TinyCNN", "SmallCNN", "SeparableSmallCNN", "MediumCNN", "SeparableMediumCNN"])

data_dict = {
    "Class_Pair": "1 v 2",
    "TinyCNN": 0.9730,
    "SmallCNN": 0.9850,
    "SeparableSmallCNN": 0.9800,
    "MediumCNN": 0.9900,
    "SeparableMediumCNN": 0.9880
}

print(result_df)

result_df.loc[len(result_df)] = data_dict
print(result_df)
result_df.loc[len(result_df)] = data_dict
print(result_df)