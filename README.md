# 📦 Supply Chain Optimization (ML Pipeline)

[![Made with Jupyter](https://img.shields.io/badge/Made%20with-Jupyter-orange)](#)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](#)

A complete, notebook-driven machine-learning pipeline to tackle four core supply-chain problems:

- **Demand Forecasting** (predict units to sell)
- **Inventory Optimization** (predict optimal stock levels)
- **Lead-Time Prediction** (predict delivery time)
- **Revenue Forecasting** (predict revenue)

The repository centers around a single notebook: **`supply chain.ipynb`**, which performs EDA → feature engineering → model training & evaluation → artifact saving.

---

## 🔎 Project Overview
This project builds practical ML models that help supply-chain teams plan better:

- **Forecast future demand** to reduce stock-outs/overstock
- **Set inventory targets** per SKU/location
- **Predict lead times** to improve delivery SLAs
- **Forecast revenue** for finance & ops planning

Each task is modeled independently with curated features, cross-validated baselines, and clear test metrics (R² / MAE / MAPE). Best models are persisted as a single **`optimized_ml_models.pkl`** for downstream use (apps, APIs, dashboards).

---

## 🗂 Data & Features

### Base inputs (typical columns)
Common operational fields such as:
- `Price`, `Number of products sold`, `Availability`, `Stock levels`
- `Manufacturing costs`, `Order quantities`, `Lead times`, `Production volumes`
- Product/Location/Mode encodings (e.g., `product_type_*`, `location_*`, `transportation_modes_*`)

### Engineered features (examples)
- `revenue_per_unit = Revenue / Units`
- `price_to_cost_ratio = Price / Manufacturing costs`
- `inventory_turnover = Number of products sold / Stock levels`
- `profit_margin = (Price - Manufacturing costs) / Price`
- `cost_efficiency`, `supply_chain_velocity`, `demand_intensity`, `market_share_proxy`, `price_competitiveness`
- One-hot encodings for product type, location, and transport modes

> **Feature set used by a trained model (example):**  
> `['Price','Number of products sold','Availability','Stock levels','revenue_per_unit','cost_efficiency','inventory_turnover','profit_margin','supply_chain_velocity','premium_product_flag','product_type_haircare','product_type_skincare','transportation_modes_Rail','transportation_modes_Road','transportation_modes_Sea']`

> **Demand-forecasting engineered set (example):**  
> `['Price','Availability','Stock levels','Lead times','Order quantities','Production volumes','Manufacturing costs','price_to_cost_ratio','inventory_turnover','cost_efficiency','profit_margin','demand_intensity','market_share_proxy','price_competitiveness','product_type_haircare','product_type_skincare','location_Chennai','location_Delhi','location_Kolkata','location_Mumbai']`

> Your notebook computes/uses these consistently before training.

---

## 🤖 Models & Metrics

We benchmark multiple algorithms per task and select the best on the test split:

- **Models:** Linear Regression, **Ridge**, Decision Tree, **Random Forest**, **Gradient Boosting**
- **Metrics:** R² (↑ better), MAE (↓ better), MAPE (↓ better)

**Typical snapshot (from this notebook run):**
- **Demand Forecasting:** Ridge (R² ≈ **0.90**)
- **Inventory Optimization:** Random Forest (R² ≈ **0.64**)
- **Lead-Time Prediction:** Linear Regression (R² ≈ **1.00**)
- **Revenue Forecasting:** Random Forest (R² ≈ **0.68**)

Overall average R² ≈ **0.81** → **2 models production-ready**, **2 models pilot-ready**.

> Exact numbers print in the notebook and are exported to CSV.

---

## 📁 Project Structure
```bash 📦 supply-chain-ml
├─ supply chain.ipynb # Main notebook (EDA → FE → Train → Evaluate → Save)
├─ supply_chain_data.csv # Sample dataset (rename/path if using your own)
├─ optimized_ml_performance_summary.csv# Auto-saved: best model metrics per task
├─ optimized_ml_models.pkl # Auto-saved: trained models & metadata
├─ requirements.txt # (Optional) pinned deps
└─ README.md # This file


