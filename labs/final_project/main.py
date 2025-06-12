from src.data_input import DataInput
from src.calculations import LCACalculator
from src.visualization import LCAVisualizer

def main():
    try:
        # Load and validate input data
        data_input = DataInput("data/raw/sample_data.csv")
        df = data_input.load_and_validate()

        # Perform LCA calculations
        calculator = LCACalculator("data/raw/impact_factors.json")
        results = calculator.calculate_impacts(df)

        # Visualize results
        visualizer = LCAVisualizer()
        visualizer.plot_stage_impacts(results)
        print("Analysis complete. Visualizations saved.")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
