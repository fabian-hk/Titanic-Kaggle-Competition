import seaborn as sn
import matplotlib.pyplot as plt

from data_prepocessing import DataPreprocessing


def compute_correlation_matrix(data_class: DataPreprocessing):
    corr = data_class.raw_data.corr()

    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    sn.heatmap(corr, annot=True, ax=ax)
    plt.show()


def min_max_mean(data_class: DataPreprocessing):
    df = data_class.raw_data
    for column in df:
        print(f"Column: {column}")
        print(f"{df[column].describe()}\n")


if __name__ == "__main__":
    data_class = DataPreprocessing()

    min_max_mean(data_class)
    compute_correlation_matrix(data_class)
