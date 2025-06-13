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
8. Show total impacts by product\n9. Show normalized impacts\n0. Exit
    """)

def ensure_plot_dir():
    plots_dir = os.path.join(os.path.dirname(__file__), 'results', 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    return plots_dir

def main():
    data_input = DataInput()
    visualizer = LCAVisualizer()
    product_data = None
    impacts = None
    calculator = None
    plot_dir = ensure_plot_dir()

    base_path = os.path.dirname(__file__)
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
                print("â ï¸ Please load data first.")
                continue
            try:
                impacts = calculator.calculate_impacts(product_data)
                print("âï¸ Impacts calculated. Here is a summary:")
                summary_cols = ['product_id', 'life_cycle_stage', 'material_type',
                                'carbon_impact', 'energy_impact', 'water_impact']
                print(impacts[summary_cols].head(10).to_string(index=False))
            except Exception as e:
                print(f"â Error calculating impacts: {e}")

        elif choice == "3":
            if impacts is None:
                print("â ï¸ Please calculate impacts first.")
                continue
            try:
                group_by = input("Group by ('material_type' or 'life_cycle_stage')? ").strip() or "material_type"
                fig = visualizer.plot_impact_breakdown(impacts, 'carbon_impact', group_by)
                save_path = os.path.join(plot_dir, f"impact_breakdown_{group_by}.png")
                fig.savefig(save_path)
                print(f"âï¸ Chart saved to {save_path}")
            except Exception as e:
                print(f"â Error generating pie chart: {e}")

        elif choice == "4":
            if impacts is None:
                print("â ï¸ Please calculate impacts first.")
                continue
            try:
                product_id = input("Enter product ID: ").strip() or impacts['product_id'].iloc[0]
                fig = visualizer.plot_life_cycle_impacts(impacts, product_id)
                save_path = os.path.join(plot_dir, f"lifecycle_impacts_{product_id}.png")
                fig.savefig(save_path)
                print(f"âï¸ Chart saved to {save_path}")
            except Exception as e:
                print(f"â Error generating life cycle bar chart: {e}")

        elif choice == "5":
            if impacts is None:
                print("â ï¸ Please calculate impacts first.")
                continue
            try:
                ids = input("Enter comma-separated product IDs (e.g. P001,P002): ").strip().split(",")
                fig = visualizer.plot_product_comparison(impacts, ids)
                filename = "_".join(ids).replace(" ", "")
                save_path = os.path.join(plot_dir, f"product_comparison_{filename}.png")
                fig.savefig(save_path)
                print(f"âï¸ Chart saved to {save_path}")
            except Exception as e:
                print(f"â Error generating radar chart: {e}")

        elif choice == "6":
            if impacts is None:
                print("â ï¸ Please calculate impacts first.")
                continue
            try:
                product_id = input("Enter product ID: ").strip() or impacts['product_id'].iloc[0]
                fig = visualizer.plot_end_of_life_breakdown(impacts, product_id)
                save_path = os.path.join(plot_dir, f"eol_breakdown_{product_id}.png")
                fig.savefig(save_path)
                print(f"âï¸ Chart saved to {save_path}")
            except Exception as e:
                print(f"â Error generating EOL bar chart: {e}")

        elif choice == "7":
            if impacts is None:
                print("â ï¸ Please calculate impacts first.")
                continue
            try:
                fig = visualizer.plot_impact_correlation(impacts)
                save_path = os.path.join(plot_dir, "impact_correlation.png")
                fig.savefig(save_path)
                print(f"âï¸ Chart saved to {save_path}")
            except Exception as e:
                print(f"â Error generating correlation heatmap: {e}")

        
        elif choice == "8":
            if impacts is None:
                print("â ï¸ Please calculate impacts first.")
                continue
            try:
                total_impacts = calculator.calculate_total_impacts(impacts)
                print("ð Total Impacts by Product:")
                print(total_impacts.to_string(index=False))
                # Optionally save to CSV
                csv_path = os.path.join(plot_dir, "..", "total_impacts.csv")
                total_impacts.to_csv(csv_path, index=False)
                print(f"ð¾ Saved to {csv_path}")
            except Exception as e:
                print(f"â Error aggregating total impacts: {e}")

        elif choice == "9":
            if impacts is None:
                print("â ï¸ Please calculate impacts first.")
                continue
            try:
                normalized = calculator.normalize_impacts(impacts)
                print("ð Normalized Impacts (0â1 scale):")
                print(normalized[['product_id', 'carbon_impact', 'energy_impact', 'water_impact', 'waste_generated_kg']].head(10).to_string(index=False))
                norm_path = os.path.join(plot_dir, "..", "normalized_impacts.csv")
                normalized.to_csv(norm_path, index=False)
                print(f"ð¾ Saved to {norm_path}")
            except Exception as e:
                print(f"â Error normalizing impacts: {e}")


        elif choice == "0":
            print("ð Exiting LCA Tool.")
            break

        else:
            print("â Invalid choice. Try again.")

if __name__ == "__main__":
    main()

