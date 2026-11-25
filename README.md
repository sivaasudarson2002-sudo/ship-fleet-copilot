
# 🚢 Ship Fleet Copilot — Data Analysis, ML & Agentic AI

An end-to-end project on ship operations data: cleaning, exploratory analysis, machine learning,
and an **AI agent (Google Gemini)** that answers fleet questions using tools built on the data.

## 📊 Dataset
- [Ship Performance Clustering Dataset (Kaggle)](https://www.kaggle.com/datasets/jeleeladekunlefijabi/ship-performance-clustering-dataset) — 2,736 weekly voyage records, 18 columns
- Synthetic but realistic data: ship type, route, engine, maintenance, weather, speed, power, cost, revenue, efficiency

## 📁 Files
| File | What it contains |
|---|---|
| `Ship_project.ipynb` | Data cleaning, feature engineering, EDA, regression, leakage demo, KMeans clustering |
| `Fleet_Copilot_Agent.ipynb` | Gemini AI agent with 5 tools built on the data and the clustering model |
| `ship_clusters.joblib` | Trained StandardScaler + KMeans model (4 voyage profiles) |

## 🔧 What I did
1. **Data cleaning** — filled 136 missing values per category with "unknown", fixed a mislabelled engine type
   (HFO is a fuel used in diesel engines), converted dates, checked duplicates and outliers (IQR rule)
2. **Feature engineering** — Profit, Energy Used (kWh), Cost per nautical mile
3. **EDA** — histograms, weather/maintenance vs efficiency, correlation heatmap
4. **Machine learning**
   - Regression (Linear, Random Forest) to predict efficiency → R² ≈ 0: the data is synthetic with no real signal
   - Showed **data leakage**: a target-derived feature gave a fake R² of 0.99
   - **KMeans clustering** → 4 operating profiles with business actions
5. **Agentic AI** — Gemini agent with 5 tools (fleet summary, group comparison, loss analysis,
   cluster profiles, new-voyage classifier using the KMeans model)

## 🧭 Voyage profiles (KMeans, k = 4)
| Profile | Voyages | Avg efficiency (nm/kWh) | Avg profit | Action |
|---|---|---|---|---|
| Efficient money-makers | 651 | 1.19 | $431k | Copy their practices |
| Full but costly | 680 | 0.67 | $120k | Review fuel, port and charter costs |
| Half-empty and costly | 715 | 0.89 | $181k | Improve cargo planning |
| Cheap but fuel-hungry | 690 | 0.46 | $343k | Hull cleaning and engine checks |

## 🤖 Agent design
- All numbers come from Python tools; the LLM never calculates or invents data
- Low temperature (0.2), 60-second timeout, max 6 tool calls per question
- Honest when data is missing (e.g. says it cannot give a CII rating)

## 🛠️ Tech stack
Python · pandas · matplotlib · seaborn · scikit-learn · joblib · Google Gemini API (google-genai) · Google Colab

## ▶️ How to run
1. Download the dataset from Kaggle (link above)
2. Open `Ship_project.ipynb` in Colab and run all cells (creates `ship_clustered.csv` and `ship_clusters.joblib`)
3. Add your Gemini API key in Colab Secrets as `GEMINI_API_KEY`
4. Open `Fleet_Copilot_Agent.ipynb`, upload `ship_clustered.csv` and `ship_clusters.joblib`, and run cell by cell

## 👤 Author
Sivaa Sudarson.S · [LinkedIn](https://www.linkedin.com/in/sivaa-sudarson-548271341) · sivaasudarson2002@gmail.com
