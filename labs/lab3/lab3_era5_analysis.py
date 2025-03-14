from pathlib import Path
import sys
import numpy
import pandas
import matplotlib.pyplot as pyplot

"""
    Pull the dataset from .csv file
"""
def load_data(input_file: str) -> pandas.DataFrame:
    try:
        dataset = pandas.read_csv(input_file, parse_dates=['timestamp'])
        return dataset
    except FileNotFoundError:
        print(f"The file '{input_file}' does  not exist.")
        sys.exit(1)
    except Exception as exception:
        print(f"Following exception occurred while trying to load file '{input_file}' {exception}:")
        sys.exit(1)

"""
    Clean the data with fillna and remove outliers depending on std on request.
"""
def clean_data(input_dataset: pandas.DataFrame, input_outliers: bool) -> pandas.DataFrame:
    if input_dataset is None:
        print("The dataset is null")
        sys.exit(1)

    input_dataset.fillna(round(input_dataset.mean(numeric_only = True), 4), inplace = True)

    if input_outliers:
        """
            Only Iterates the numeric columns
        """
        for column in input_dataset.select_dtypes(include=[numpy.number]).columns:
            column_deviation = input_dataset[column].std()
            column_mean = input_dataset[column].mean()

            if not column_deviation == 0 or column_deviation is None:
                input_dataset = input_dataset[
                    numpy.abs(input_dataset[column] - column_mean) <= (3 * column_deviation)]

    return input_dataset

def calculate_wind_speed(input_dataset: pandas.DataFrame, input_wind_column_name: str) -> pandas.DataFrame:
    """
        wind speed = sqrt(u^2 + v^2)
        Wind speed function is taken as the square root of u and v components of wind.
        Pretty much like a vector summation
    """
    wind_speed = numpy.sqrt((input_dataset['u10m'] ** 2) + (input_dataset['v10m'] ** 2))
    input_dataset[input_wind_column_name] = wind_speed
    return input_dataset

def calculate_monthly_wind_speed_average(input_dataset: pandas.DataFrame, input_wind_column_name: str) -> dict:
    input_dataset.set_index('timestamp', inplace=True)
    monthly_averages = input_dataset[input_wind_column_name].resample('ME').mean()
    input_dataset.reset_index(inplace=True)
    monthly_averages.index = monthly_averages.index.strftime('%B')
    return monthly_averages

def calculate_seasonal_wind_speed_average(input_dataset: pandas.DataFrame, input_wind_column_name: str) -> dict:
    dataset_by_seasons = input_dataset.groupby(
        input_dataset['timestamp'].apply(lambda datetime: util_get_season(datetime, north_hemisphere=True))
    )
    seasonal_averages = dataset_by_seasons[input_wind_column_name].mean()
    return seasonal_averages

"""
    Calculates the 99th percentile of extreme weather conditions
"""
def calculate_extreme_weather_conditions(input_dataset: pandas.DataFrame, input_wind_column_name: str) -> pandas.DataFrame:
    threshold = input_dataset[input_wind_column_name].quantile(0.99)
    extreme_sorted_dataset = (input_dataset[
        input_dataset[input_wind_column_name] >= threshold]
                  .sort_values(by=input_wind_column_name, ascending=False))
    return extreme_sorted_dataset


def calculate_daily_wind_patterns(input_dataset: pandas.DataFrame, wind_speed_column_name: str) -> dict:
    input_dataset['hour'] = input_dataset['timestamp'].dt.hour
    hourly_averages = input_dataset.groupby('hour')[wind_speed_column_name].mean()
    return hourly_averages

"""
    Plots monthly wind speed averages on a scatter diagram
"""
def plot_monthly_wind_speed(city_entry_1: tuple, city_entry_2: tuple):
    pyplot.figure(figsize=(10, 5))
    pyplot.xlabel("Month")
    pyplot.ylabel("Average Wind Speed (m/s)")
    pyplot.title("Monthly Average Wind Speeds")
    pyplot.grid(True)
    pyplot.tight_layout()
    pyplot.legend([city_entry_1[0], city_entry_2[0]], loc="upper left")
    pyplot.plot(city_entry_1[1].index, city_entry_1[1].values, label=city_entry_1[0], marker='o')
    pyplot.plot(city_entry_2[1].index, city_entry_2[1].values, label=city_entry_2[0], marker='s')
    pyplot.show()


"""
    Plots seasonal wind speed averages on a bar diagram
"""
def plot_seasonal_wind_speed(city_entry_1: tuple, city_entry_2: tuple):
    pyplot.figure(figsize=(10, 5))
    pyplot.xlabel("Season")
    pyplot.ylabel("Average Wind Speed (m/s)")
    pyplot.title("Seasonal Average Wind Speeds")
    pyplot.xticks(numpy.arange(len(city_entry_1[1])), city_entry_1[1].keys())
    pyplot.grid(True)
    pyplot.tight_layout()
    pyplot.legend([city_entry_1[0], city_entry_2[0]], loc="upper right")
    pyplot.bar(numpy.arange(len(city_entry_1[1])) - 0.2, city_entry_1[1].values, 0.4, label=city_entry_1[0])
    pyplot.bar(numpy.arange(len(city_entry_1[1])) + 0.2, city_entry_2[1].values, 0.4, label=city_entry_2[0])
    pyplot.show()

"""
    Plots wind directions on a polar plot
"""
def plot_wind_direction(input_dataset: pandas.DataFrame, city: str):
    """
        Wind direction is calculated by finding the angle between horizontal axis and the vector
    """
    wind_direction = (numpy.degrees(
        numpy.arctan2(input_dataset['v10m'], input_dataset['u10m'])) + 360) % 360
    """
        Divides plot into 22.5 degree subsections
    """
    bins = numpy.arange(0, 361, 22.5)
    counts, _ = numpy.histogram(wind_direction, bins=bins)

    """
        Creates a polar plot
    """
    theta = numpy.deg2rad(bins[:-1] + 11.25)  # center of each bin
    widths = numpy.deg2rad([22.5] * len(theta))

    pyplot.tight_layout()
    (pyplot.subplot(111, polar=True)
     .bar(theta, counts, width=widths, bottom=0.0, edgecolor='gray', align='center'))
    pyplot.title(f"Wind Direction in {city}")
    pyplot.show()

def display_menu():
    print("\nERA5 Analysis Menu:")
    print("1. Print DataFrame Information")
    print("2. Print Basic Statistics")
    print("3. Print Extreme Weather Conditions (99th Percentile)")
    print("4. Print Hourly Wind Patterns")
    print("5. Plot Wind Direction")
    print("6. Plot Monthly Wind Speed (comparison between cities)")
    print("7. Plot Seasonal Wind Speed (comparison between cities)")
    print("8. Exit")

def main():
    input_clean_outliers = input('Clean the outliers? (y/n): ').lower().strip() == 'y'
    wind_speed_column_name = "wind_speed"

    """
        Define the file path for the datasets
    """
    file_paths = {
        "Berlin": (Path(__file__).parents[2] / "datasets/berlin_era5_wind_20241231_20241231.csv"),
        "Munich": (Path(__file__).parents[2] / "datasets/munich_era5_wind_20241231_20241231.csv")
    }

    """
        This for loop is used instead of defining for 2 cities separately to make the code scalable
        for additions of future city data
    """
    # Dictionaries to store processed data
    datasets = {}
    monthly_averages = {}
    seasonal_averages = {}
    extreme_weather_conditions = {}
    hourly_patterns = {}

    for name,file_path in file_paths.items():
        """
            Load and fill in NaN values for the dataset
                if input_clean_outliers is true also clean the data from outliers
            Definitions are made on same variable name to save memory space
        """
        dataset = (
            clean_data(load_data(file_path), input_clean_outliers))

        """
            Wind speed is calculated and added as a new column to the dataframe
        """
        dataset = (
            calculate_wind_speed(dataset, wind_speed_column_name))

        datasets[name] = dataset.copy()

        """
            Monthly wind speed average is calculated straightforward and added to dictionary
        """
        monthly_averages[name] = (
            calculate_monthly_wind_speed_average(dataset, wind_speed_column_name))

        """
            Seasonal wind speed average is calculated using util function and added to dictionary
        """
        seasonal_averages[name] = (
            calculate_seasonal_wind_speed_average(dataset, wind_speed_column_name))

        """
            Calculate the highest wind speeds during the year
        """
        extreme_weather_conditions[name] = calculate_extreme_weather_conditions(
            dataset.copy(), wind_speed_column_name)

        """
            Calculate hourly average wind speed
        """
        hourly_patterns[name] = calculate_daily_wind_patterns(
            dataset.copy(), wind_speed_column_name)

    """
        While loop is used for the menu for repeatable menu behaviour
    """
    while True:
        display_menu()
        choice = input("Please select one of the options (1-8): ").strip()
        match choice:
            case "1":
                for city_name, dataset in datasets.items():
                    print(f"\n'{city_name}' DataFrame Information:")
                    print(dataset.info())
            case "2":
                for city_name, dataset in datasets.items():
                    print(f"\n'{city_name}' Summary Statistics:")
                    print(dataset.describe())
            case "3":
                for city_name, extremes in extreme_weather_conditions.items():
                    print(f"\n'{city_name}' Extreme Weather Conditions:")
                    print(extremes)
            case "4":
                for city_name, hourly in hourly_patterns.items():
                    print(f"\n'{city_name}' Hourly Wind Patterns:")
                    print(hourly)
            case "5":
                for city_name, hourly in datasets.items():
                    plot_wind_direction(datasets[city_name].copy(), city_name)
            case "6":
                plot_monthly_wind_speed(
                    ("Berlin", monthly_averages["Berlin"]),
                    ("Munich", monthly_averages["Munich"]))
            case "7":
                plot_seasonal_wind_speed(
                    ("Berlin", seasonal_averages["Berlin"]),
                    ("Munich", seasonal_averages["Munich"]))
            case "8":
                print("Terminating...")
                break
            case _:
                print("Invalid operation. Please select a valid option (1-8).")

"""
    A utility function to get season from a datetime
"""
def util_get_season(input_datetime, north_hemisphere: bool = True):
    now = (input_datetime.date().month, input_datetime.date().day)
    if (3, 21) <= now < (6, 21):
        season = 'Spring' if north_hemisphere else 'Fall'
    elif (6, 21) <= now < (9, 23):
        season = 'Summer' if north_hemisphere else 'Winter'
    elif (9, 23) <= now < (12, 21):
        season = 'Fall' if north_hemisphere else 'Spring'
    else:
        season = 'Winter' if north_hemisphere else 'Summer'

    return season

if __name__ == "__main__":
    main()


"""
Skyrim makes it possible to run large weather models on a common GPU rather than depending on extremely high-end costly setups. 
It also accommodates cloud-based (Modal) and on-prem GPU configurations, easing forecasting and saving costs. 
Setup is easy, and you can save results in AWS or compute them in a Jupyter notebook. 
With model support for the likes of FourcastNet, Pangu, and Graphcast (ML models for weather forecasting), 
and upcoming additions like real-time NWP (Numerical weather forecast) and model quantization, 
Skyrim is all about making leading-edge weather forecasting accessible to everyone.
"""