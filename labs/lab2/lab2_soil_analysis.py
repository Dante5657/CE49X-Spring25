import pandas as pandas
import numpy as numpy
from pathlib import Path



def load_data(input_file):
    try:
        dataset = pandas.read_csv(input_file)
        return dataset
    except FileNotFoundError:
        print(f"The file '{input_file}' does  not exist.")
        quit()

def clean_data(input_dataset, input_outliers):
    if input_dataset is None:
        print("The dataset is null")
        quit()

    input_dataset.fillna(round(input_dataset.mean(numeric_only=True),1), inplace=True)

    if input_outliers and 'soil_ph' in input_dataset.columns:
        mean_ph = input_dataset['soil_ph'].mean()
        deviation_ph = input_dataset['soil_ph'].std()
        input_dataset = input_dataset[numpy.abs(input_dataset['soil_ph'] - mean_ph) <= (3 * deviation_ph)]

    return input_dataset


def compute_statistics(input_dataset, column='soil_ph'):
    if input_dataset is None or column not in input_dataset.columns:
        print(f"Error: Column '{column}' not found in dataset.")
        return

    stats = {
        'Minimum': input_dataset[column].min(),
        'Maximum': input_dataset[column].max(),
        'Mean': input_dataset[column].mean(),
        'Median': input_dataset[column].median(),
        'Standard Deviation': input_dataset[column].std(),
    }

    print(f"Descriptive Statistics for '{column}':")
    for key, value in stats.items():
        print(f"{key}: {value:.2f}")

    return stats


def main():
    root_folder = Path(__file__).parents[2]
    file_path = root_folder / "datasets/soil_test.csv"
    dataset = load_data(file_path)
    cleaned_data = clean_data(dataset, True)
    print(cleaned_data)
    compute_statistics(cleaned_data)

main()
