import numpy as np
import pandas as pd

df = pd.read_csv("cleaned_dev.csv")


df_1 = df[df["bad_flag"] == 1.0]
df_0 = df[df["bad_flag"] == 0.0]


df_0 = df_0.sample(frac=1, random_state=42).reset_index(drop=True)

splits = np.array_split(df_0, 6)


dfs = [pd.concat([df_1, split]).sample(frac=1, random_state=42).reset_index(drop=True) for split in splits]

output_file = "balanced_datasets.xlsx"

with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
    for i, df in enumerate(dfs):
        sheet_name = f"df_{i + 1}"
        df.to_excel(writer, index=False, sheet_name=sheet_name)
