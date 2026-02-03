import pandas as pd
import os

def combine_csv_files(file_list, output_file):
    combined_df = pd.DataFrame()

    for file in file_list:
        df = pd.read_csv(file)
        
        # Drop index column if it exists (like 'Unnamed: 0')
        if 'Unnamed: 0' in df.columns:
            df = df.drop('Unnamed: 0', axis=1)
        
        # Get the base name without extension for column naming
        base_name = os.path.splitext(os.path.basename(file))[0]
        
        # Rename 'rank' column to the file name
        if 'rank' in df.columns:
            df = df.rename(columns={'rank': base_name})
        
        if combined_df.empty:
            combined_df = df
        else:
            # Merge on 'layer' column
            combined_df = pd.merge(combined_df, df, on="layer", how="outer")

    combined_df.to_csv(output_file, index=False)
    return combined_df

data_dir = "/cis/home/schaud35/clinical-highmmt/layerwise_ranks/multitask/fusemoe"
# df_enc = combine_csv_files(
#     [
#         f"{data_dir}/{f}" for f in os.listdir(data_dir) if f.endswith("_enc_ranks.csv")
#     ],
#     f"{data_dir}/multimodal_multitask_enc_ranks_combined.csv"
# )
# df = combine_csv_files(
#     [
#         f"{data_dir}/{f}" for f in os.listdir(data_dir) if "_enc_ranks" not in f
#     ],
#     f"{data_dir}/multimodal_multitask_ranks_combined.csv"
# )
# import pdb; pdb.set_trace()


# # Read, skipping title + blank row
df = pd.read_csv(
    f"{data_dir}/multimodal_multitask_ranks_combined.csv",
    skiprows=1
)

# Rename columns explicitly
# CrossAttnTransformer multimodal multitask
# df.columns = ["layer", "IHM+LOS+PHENO", "IHM", "LOS", "PHENO"]
# CrossAttnTransformer fleximodal multitask
# df.columns = ["layer", "IHM-LOS_TS-Text-CXR_Text-CXR", "IHM-LOS_TS-Text-CXR_TS-CXR", "IHM-LOS_TS-Text-CXR_TS-Text",
#               "IHM-PHENO_TS-Text-CXR_Text-CXR", "IHM-PHENO_TS-Text-CXR_TS-CXR", "IHM-PHENO_TS-Text-CXR_TS-Text",
#               "LOS-PHENO_TS-Text-CXR_Text-CXR", "LOS-PHENO_TS-Text-CXR_TS-CXR", "LOS-PHENO_TS-Text-CXR_TS-Text",
#               ]
# FuseMoe fleximodal multitask
df.columns = ["layer", "IHM_TS-Text-CXR", "LOS_TS-Text-CXR", "PHENO_TS-Text-CXR",
                "IHM-LOS_TS-Text-CXR_TS-Text-CXR", "IHM-LOS_TS-Text_TS-Text",
                "IHM_TS-Text", "LOS_TS-Text", "PHENO_TS-Text",
             ]

# Keep only meaningful columns
# df = df[["layer", "IHM", "LOS", "PHENO", "IHM+LOS+PHENO"]]
modalities = "TS-Text"
df = df[["layer", f"IHM_{modalities}", f"LOS_{modalities}", f"PHENO_{modalities}", f"IHM-LOS_{modalities}_{modalities}"]]

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
plt.savefig(f"{data_dir}/{modalities}_layer_rank_heatmap.png", dpi=300)
