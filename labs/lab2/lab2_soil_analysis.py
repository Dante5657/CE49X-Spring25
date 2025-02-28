import pandas as pandas
import numpy as numpy
from pathlib import Path

# Pull the dataset from .csv file
def load_data(input_file):
    try:
        dataset = pandas.read_csv(input_file)
        return dataset
    except FileNotFoundError:
        print(f"The file '{input_file}' does  not exist.")
        quit()

# Clean the data, replace NaN values and remove outliers
def clean_data(input_dataset, input_outliers):
    if input_dataset is None:
        print("The dataset is null")
        quit()

    input_dataset.fillna(round(input_dataset.mean(numeric_only = True), 1), inplace = True)

    if input_outliers:
        for column in range(1, input_dataset.columns.size):
            column_mean = input_dataset[input_dataset.columns[column]].mean()
            column_deviation = input_dataset[input_dataset.columns[column]].std()
            input_dataset = input_dataset[
                numpy.abs(input_dataset[input_dataset.columns[column]] - column_mean) <= (3 * column_deviation)]

    return input_dataset

# Compute statistics such as mean, median etc. for all columns
def compute_statistics(input_dataset):
    if input_dataset is None:
        print("The dataset is null")
        quit()

    total_statistics = {}
    for column in range(1, input_dataset.columns.size):
        column_statistics = {
            "Minimum": input_dataset[input_dataset.columns[column]].min(),
            "Maximum": input_dataset[input_dataset.columns[column]].max(),
            "Mean": input_dataset[input_dataset.columns[column]].mean(),
            "Median": input_dataset[input_dataset.columns[column]].median(),
            "Standard Deviation": input_dataset[input_dataset.columns[column]].std(),
        }
        total_statistics[input_dataset.columns[column]] = column_statistics

    for column in range(len(total_statistics)):
        print(" ")
        print(f"Descriptive Statistics for '{list(total_statistics.keys())[column]}':")
        for key, value in total_statistics[list(total_statistics.keys())[column]].items():
            print(f"{key}: {value:.2f}")

    return


def main():
    input_clean_outliers = bool(input("Clean the outliers?: "))

    file_path = Path(__file__).parents[2] / "datasets/soil_test.csv"

    # Do all the work
    compute_statistics(clean_data(load_data(file_path), input_clean_outliers))

main()
