
from src.data_input import DataInput
from src.calculations import LCACalculator
from src.visualization import LCAVisualizer
import matplotlib.pyplot as plt
import os

def display_menu():
    print("""
LCA Tool - Life Cycle Assessment
--------------------------------
1. Load data
2. Calculate impacts
3. Show impact breakdown (pie chart)
4. Show life cycle impacts (bar chart)
5. Compare products (radar chart)
6. Show end-of-life breakdown (stacked bar)
7. Show impact correlations (heatmap)
0. Exit
    """)

def main():
    data_input = DataInput()
    visualizer = LCAVisualizer()
    product_data = None
    impacts = None
    calculator = None

    1base_path = os.path.dirname(__file__)
    default_data_path = os.path.join(base_path, 'data', 'raw', 'sample_data.csv')
    default_factors_path = os.path.join(base_path, 'data', 'raw', 'impact_factors.json')

    while True:
        display_menu()
        choice = input("Enter choice (0-7): ").strip()

        if choice == "1":
            try:
                data_path = input(f"Enter path to data file (default: {default_data_path}): ").strip()
                if not data_path:
                    data_path = default_data_path

                factors_path = input(f"Enter path to impact factors file (default: {default_factors_path}): ").strip()
                if not factors_path:
                    factors_path = default_factors_path

                product_data = data_input.read_data(data_path)
                calculator = LCACalculator(impact_factors_path=factors_path)
                print("âï¸ Data successfully loaded.")
            except Exception as e:
                print(f"â Error loading data: {e}")

        elif choice == "2":
            if product_data is None or calculator is None:
                print("⚠️ Please load data first.")
                continue
            try:
                impacts = calculator.calculate_impacts(product_data)
                print("✔️ Impacts calculated.")
            except Exception as e:
                print(f"❌ Error calculating impacts: {e}")

        elif choice == "3":
            if impacts is None:
                print("⚠️ Please calculate impacts first.")
                continue
            try:
                group_by = input("Group by ('material_type' or 'life_cycle_stage')? ").strip() or "material_type"
                fig = visualizer.plot_impact_breakdown(impacts, 'carbon_impact', group_by)
                plt.show()
            except Exception as e:
                print(f"❌ Error generating pie chart: {e}")

        elif choice == "4":
            if impacts is None:
                print("⚠️ Please calculate impacts first.")
                continue
            try:
                product_id = input("Enter product ID: ").strip() or impacts['product_id'].iloc[0]
                fig = visualizer.plot_life_cycle_impacts(impacts, product_id)
                plt.show()
            except Exception as e:
                print(f"❌ Error generating life cycle bar chart: {e}")

        elif choice == "5":
            if impacts is None:
                print("⚠️ Please calculate impacts first.")
                continue
            try:
                ids = input("Enter comma-separated product IDs (e.g. P001,P002): ").strip().split(",")
                fig = visualizer.plot_product_comparison(impacts, ids)
                plt.show()
            except Exception as e:
                print(f"❌ Error generating radar chart: {e}")

        elif choice == "6":
            if impacts is None:
                print("⚠️ Please calculate impacts first.")
                continue
            try:
                product_id = input("Enter product ID: ").strip() or impacts['product_id'].iloc[0]
                fig = visualizer.plot_end_of_life_breakdown(impacts, product_id)
                plt.show()
            except Exception as e:
                print(f"❌ Error generating EOL bar chart: {e}")

        elif choice == "7":
            if impacts is None:
                print("⚠️ Please calculate impacts first.")
                continue
            try:
                fig = visualizer.plot_impact_correlation(impacts)
                plt.show()
            except Exception as e:
                print(f"❌ Error generating correlation heatmap: {e}")

        elif choice == "0":
            print("👋 Exiting LCA Tool.")
            break

        else:
            print("❌ Invalid choice. Try again.")

if __name__ == "__main__":
    main()

