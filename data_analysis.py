import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from data_prepocessing import DataPreprocessing

data_class = DataPreprocessing()

f, ax = plt.subplots(figsize=(10, 8))
corr = data_class.raw_data.corr()
print(corr)

sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)
