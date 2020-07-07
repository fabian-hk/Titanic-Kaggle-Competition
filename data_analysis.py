import seaborn as sn
import matplotlib.pyplot as plt

from data_prepocessing import DataPreprocessing

data_class = DataPreprocessing()

corr = data_class.raw_data.corr()
print(corr)

fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
sn.heatmap(corr, annot=True, ax=ax)
plt.show()
