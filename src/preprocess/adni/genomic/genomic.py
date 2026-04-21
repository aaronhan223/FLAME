import pandas as pd
from pandas_plink import read_plink1_bin, write_plink1_bin
import numpy as np
import anndata
import os

df = pd.read_csv('/export/io79/data/schaud35/datasets/adni/genomic/ADNI_1_GWAS_Plink/ADNI_cluster_01_forward_757LONI' + '.bim', header=None, sep='\t')

print(df.head())
