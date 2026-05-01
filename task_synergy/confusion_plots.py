import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

tasks = ["IHM","LOS","MOR","RAD","Birads","Risk","Density"]

# create empty matrix
matrix = pd.DataFrame(np.nan, index=tasks, columns=tasks)

pairs = {
("IHM","LOS"):(0.77,0.80),
("MOR","RAD"):(0.00,0.00),
("Birads","Risk"):(0.90,0.70),
("Birads","Density"):(0.80,0.93),
("Risk","Density"):(0.71,0.94),
("IHM","MOR"):(0.75,0.83),
("IHM","RAD"):(0.76,0.76),
("IHM","Birads"):(0.76,0.90),
("IHM","Risk"):(0.77,0.48),
("IHM","Density"):(0.77,0.93),
("LOS","MOR"):(0.79,0.83),
("LOS","RAD"):(0.79,0.76),
("LOS","Birads"):(0.81,0.85),
("LOS","Risk"):(0.79,0.49),
("LOS","Density"):(0.79,0.78),
("MOR","Birads"):(0.83,0.72),
("MOR","Risk"):(0.83,0.50),
("MOR","Density"):(0.83,0.90),
("RAD","Birads"):(0.75,0.90),
("RAD","Risk"):(0.75,0.51),
("RAD","Density"):(0.75,0.90),
}

# fill matrix
for (a,b),(v1,v2) in pairs.items():
    matrix.loc[a,b] = v1
    matrix.loc[b,a] = v2

# optional: diagonal values
np.fill_diagonal(matrix.values,1.0)

# plot heatmap
plt.figure(figsize=(8,6))

sns.heatmap(
    matrix.astype(float),
    annot=True,
    fmt=".2f",
    cmap="viridis",
    linewidths=0.5
)

plt.title("AUROC Confusion Heatmap Across Tasks")
plt.xlabel("Evaluation Task")
plt.ylabel("Training Task")

plt.tight_layout()
plt.savefig("pairwise_confusion_heatmap_auroc.png", dpi=300)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

tasks = ["IHM","LOS","MOR","RAD","Birads","Risk","Density"]

# create empty matrix
matrix_f1 = pd.DataFrame(np.nan, index=tasks, columns=tasks)

pairs_f1 = {
("IHM","LOS"):(0.35,0.68),
("MOR","RAD"):(0.00,0.00),
("Birads","Risk"):(0.47,0.20),
("Birads","Density"):(0.44,0.70),
("Risk","Density"):(0.17,0.69),
("IHM","MOR"):(0.37,0.17),
("IHM","RAD"):(0.37,0.42),
("IHM","Birads"):(0.34,0.45),
("IHM","Risk"):(0.24,0.07),
("IHM","Density"):(0.45,0.67),
("LOS","MOR"):(0.72,0.15),
("LOS","RAD"):(0.71,0.43),
("LOS","Birads"):(0.72,0.43),
("LOS","Risk"):(0.71,0.07),
("LOS","Density"):(0.65,0.40),
("MOR","Birads"):(0.22,0.34),
("MOR","Risk"):(0.20,0.07),
("MOR","Density"):(0.24,0.62),
("RAD","Birads"):(0.46,0.44),
("RAD","Risk"):(0.44,0.07),
("RAD","Density"):(0.46,0.45),
}

# fill matrix
for (a,b),(v1,v2) in pairs_f1.items():
    matrix_f1.loc[a,b] = v1
    matrix_f1.loc[b,a] = v2

# optional diagonal values
np.fill_diagonal(matrix_f1.values,1.0)

# plot heatmap
plt.figure(figsize=(8,6))

sns.heatmap(
    matrix_f1.astype(float),
    annot=True,
    fmt=".2f",
    cmap="viridis",
    linewidths=0.5
)

plt.title("F1 Confusion Heatmap Across Tasks")
plt.xlabel("Evaluation Task")
plt.ylabel("Training Task")

plt.tight_layout()
plt.savefig("pairwise_confusion_heatmap_f1.png", dpi=300)