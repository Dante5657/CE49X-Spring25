
# LCA Tool – Life Cycle Assessment Framework

![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-stable-brightgreen.svg)

A modular and scalable Python-based tool designed for analyzing and visualizing environmental impacts across product life cycles using structured data and defined impact factors.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Project Layout](#project-layout)
- [Installation](#installation)
- [Running the Tool](#running-the-tool)
- [Input Formats](#input-formats)
- [Output Details](#output-details)
- [API Reference](#api-reference)
  - [Data Input (`data_input.py`)](#data-input-datainputpy)
  - [LCA Calculations (`calculations.py`)](#lca-calculations-calculationspy)
  - [Visualization (`visualization.py`)](#visualization-visualizationpy)
  - [Utilities (`utils.py`)](#utilities-utilspy)
- [Error Handling](#error-handling)

---

## Overview

This tool processes structured input datasets representing product stages, materials, and consumption data. It allows calculating environmental impacts such as carbon emissions, energy use, and water usage, then visualizes these results through various charts (pie, radar, bar, and heatmaps).

---

## Key Features

- 🚀 **Multi-format input support** (CSV, Excel, JSON)
- 🛡️ **Validation of structure and semantics**
- 📈 **Modular impact computation** using flexible JSON-based factor models
- 📊 **Charts for carbon/energy/water waste** breakdowns and product comparisons
- 🔄 **End-of-life analysis** (recycling, landfill, incineration)
- ♻️ **Impact correlation metrics** to identify key factors
- 🧪 **Integrated unit conversion and normalization utilities**

---

## Project Layout

```
final_project/
├── data/raw/
│   ├── sample_data.csv
│   └── impact_factors.json
├── notebooks/
│   └── lca_analysis_example.ipynb
├── src/
│   ├── data_input.py
│   ├── calculations.py
│   ├── visualization.py
│   ├── utils.py
│   └── __init__.py
├── tests/
│   ├── test_calculations.py
│   └── test_visualization.py
├── main.py
└── requirements.txt
```

---

## Installation

```bash
git clone <repo-url>
cd final_project
pip install -r requirements.txt
```

---

## Running the Tool

To perform a full LCA workflow:

```bash
python main.py
```

---

## Input Formats

### Product Data (CSV/JSON/Excel)
Each row represents a stage in the product life cycle. Required columns include:

- `product_id`, `product_name`, `life_cycle_stage`
- `material_type`, `quantity_kg`, `energy_consumption_kwh`, etc.

### Impact Factors (JSON)
Hierarchical structure: `material -> stage -> {carbon, energy, water}`

---

## Output Details

- Impact calculations (DataFrames)
- Breakdown and comparison plots
- Summary reports (optional extension)
- Figures returned as Matplotlib objects

---

## API Reference

### Data Input (`data_input.py`)

**Class `DataInput`**
- `read_data(path)`: Load CSV, Excel, or JSON input files.
- `read_impact_factors(path)`: Load nested JSON with environmental coefficients.

### LCA Calculations (`calculations.py`)

**Class `LCACalculator`**
- `__init__(impact_factors_path)`: Load and parse the JSON impact model.
- `calculate_impacts(data)`: Compute per-row impacts using stage-specific multipliers.

### Visualization (`visualization.py`)

**Class `LCAVisualizer`**
- `plot_impact_breakdown(data, impact_type, group_by)`: Pie chart by material or stage.
- `plot_life_cycle_impacts(data, product_id)`: Stacked bar of impacts over stages.
- `plot_product_comparison(data, product_ids)`: Radar chart of multi-product impact profiles.
- `plot_end_of_life_breakdown(data, product_id)`: Stacked bar showing EOL strategies.
- `plot_impact_correlation(data)`: Heatmap showing cross-impact relationships.

### Utilities (`utils.py`)

**Function `convert_units(value, from_unit, to_unit)`**
- Supports mass, volume, and energy categories.
- Automatically identifies categories and performs conversion via intermediate base units.

---

## Error Handling

- `FileNotFoundError`, `ValueError` raised for missing or malformed input
- Impact calculation raises exceptions on empty or invalid DataFrames
- Visualization methods safely handle missing subsets (e.g., no end-of-life data)

---

## Final Notes

This tool is ready to extend with CLI options, API interfaces, or dashboard integrations. A well-structured and testable base makes it ideal for experimentation and academic LCA modeling.

---

