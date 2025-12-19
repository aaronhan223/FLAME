import pandas as pd

# Read, skipping title + blank row
df = pd.read_csv(
    "multimodal_multitask_enc_ranks.csv",
    skiprows=2
)

# Rename columns explicitly
df.columns = ["layer", "IHM+LOS+PHENO", "IHM", "LOS", "PHENO"]

# Keep only meaningful columns
df = df[["layer", "IHM", "LOS", "PHENO", "IHM+LOS+PHENO"]]

# Replace "-" with NaN and convert to numeric
df = df.replace("-", pd.NA)
df.iloc[:, 1:] = df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')

print(df.head())
import matplotlib.pyplot as plt
import numpy as np

data = df.set_index("layer")

# Convert to float and handle NaN values
data = data.astype(float)

plt.figure(figsize=(10, 12))

im = plt.imshow(
    data.values,
    aspect="auto",
    interpolation="nearest"
)

plt.colorbar(im, label="Rank")

plt.xticks(
    np.arange(len(data.columns)),
    data.columns,
    rotation=45,
    ha="right"
)

plt.yticks(
    np.arange(len(data.index)),
    data.index
)

plt.title("Layer-wise Rank (SV>0.001) Heatmap Across Tasks")
plt.tight_layout()
plt.savefig("layer_enc_rank_heatmap.png", dpi=300)
