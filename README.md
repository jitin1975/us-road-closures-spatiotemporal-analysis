# US Road Construction & Closures — Spatio-Temporal Data Analytics

> **AID843: Spatio-Temporal Data Analytics-I** |
> International Institute of Information Technology Bangalore  
> Programming Assignments A3

---

## Dataset

US Road Construction and Closures (2016–2021) | [Kaggle — sobhanmoosavi](https://www.kaggle.com/datasets/sobhanmoosavi/us-road-construction-and-closures) |

**A3 Dataset Stats:**
- 6,170,627 raw records → 1,644,242 after preprocessing
- 47 attributes: GPS coordinates, timestamps, severity, weather, 13 POI flags
- Coverage: All 50 US states, Feb 2016 – Dec 2021

---

## Repository Structure

```
us-road-closures-spatiotemporal-analysis/
│                       
│   ├── stda-assgn-3.ipynb       
│   ├── report.pdf               # Compiled report
│   │
│   ├── outputs/
│   │   ├── latex_tables.tex             # Auto-generated LaTeX tables (38 tables)
│   │   │
│   │   ├── eda/                         # Exploratory Data Analysis figures
│   │   │   ├── fig1_missing_values.png
│   │   │   ├── fig2_temporal_distributions.png
│   │   │   ├── fig3_closure_severity.png
│   │   │   ├── fig4_duration_distribution.png
│   │   │   ├── fig5_top_states.png
│   │   │   ├── fig6_weather_conditions.png
│   │   │   ├── fig7_correlation_heatmap.png
│   │   │   ├── fig8_feature_variance.png
│   │   │   └── fig9_pca_features.png
│   │   │
│   │   ├── temporal/                    # Temporal analysis figures
│   │   │   ├── fig_t1_time_series.png
│   │   │   ├── fig_t2_decomposition.png
│   │   │   ├── fig_t3_acf_pacf.png
│   │   │   ├── fig_t4_arima_forecast.png
│   │   │   ├── fig_t5_ets_models.png
│   │   │   ├── fig_t6_ets_comparison.png
│   │   │   ├── fig_t7_classification.png
│   │   │   ├── fig_T7_sarima.png
│   │   │   ├── fig_T8_prophet.png
│   │   │   ├── fig_T9_lstm_temporal.png
│   │   │   ├── fig_t9_regression.png
│   │   │   ├── fig_t10_heatmap_hourXdow.png
│   │   │   └── fig_t11_heatmap_yearXmonth.png
│   │   │
│   │   ├── spatial/                     # Spatial analysis figures
│   │   │   ├── fig_s1_density_map.png
│   │   │   ├── fig_s2_kde.png
│   │   │   ├── fig_s3_morans_i.png
│   │   │   ├── fig_s3b_gearys_c.png
│   │   │   ├── fig_s3c_lisa.png
│   │   │   ├── fig_s4_clustering.png
│   │   │   ├── fig_s5_spatial_classification.png
│   │   │   ├── fig_s6_gwr.png
│   │   │   ├── fig_s7_state_choropleth.png
│   │   │   └── fig_s8_variogram.png
│   │   │
│   │   ├── mining/                      # Data mining figures
│   │   │   ├── fig_M1_colocation_PI.png
│   │   │   ├── fig_M2_temporal_patterns.png
│   │   │   ├── fig_M3_association_rules.png
│   │   │   ├── fig_M4_anomaly_detection.png
│   │   │   ├── fig_M5_temporal_anomalies.png
│   │   │   ├── fig_M6_spatial_sequences.png
│   │   │   ├── fig_M7_hotspots.png
│   │   │   └── fig_M8_cluster_profiling.png
│   │   │
│   │   ├── joint/                       # Joint inference figures
│   │   │   ├── fig_J1_spatiotemporal_hotspot.png
│   │   │   ├── fig_J2_feature_set_comparison.png
│   │   │   ├── fig_J3_spatiotemporal_interaction.png
│   │   │   ├── fig_J4_3d_dbscan.png
│   │   │   ├── fig_J5_hotspot_prediction.png
│   │   │   ├── fig_J6_comprehensive_summary.png
│   │   │   ├── fig_JX1_xgboost_spatiotemporal.png
│   │   │   └── fig_JX2_lstm_spatiotemporal.png
│   │   │
│   │   └── tables/                      # All CSV result tables (38 files)
│   │       ├── dataset_overview.csv
│   │       ├── summary_statistics.csv
│   │       ├── engineered_features.csv
│   │       ├── adf_test_results.csv
│   │       ├── arima_model_selection.csv
│   │       ├── arima_metrics.csv
│   │       ├── ets_model_comparison.csv
│   │       ├── sarima_metrics.csv
│   │       ├── prophet_metrics.csv
│   │       ├── lstm_temporal_metrics.csv
│   │       ├── temporal_classification_results.csv
│   │       ├── temporal_regression_results.csv
│   │       ├── morans_i_results.csv
│   │       ├── gearys_c_results.csv
│   │       ├── lisa_results.csv
│   │       ├── clustering_quality.csv
│   │       ├── spatial_classification_results.csv
│   │       ├── gwr_results.csv
│   │       ├── state_level_stats.csv
│   │       ├── M1_colocation_PI.csv
│   │       ├── M2_monthly_motifs.csv
│   │       ├── M2_weekly_motifs.csv
│   │       ├── M2_season_transition.csv
│   │       ├── M3_association_rules.csv
│   │       ├── M4_anomaly_comparison.csv
│   │       ├── M5_temporal_anomalies.csv
│   │       ├── M6_state_coactivation.csv
│   │       ├── M6_state_transition.csv
│   │       ├── M7_hotspots.csv
│   │       ├── M8_cluster_profiles.csv
│   │       ├── J1_hotspot_matrix.csv
│   │       ├── J2_classification_comparison.csv
│   │       ├── J2_regression_comparison.csv
│   │       ├── J4_3d_cluster_profiles.csv
│   │       ├── J5_hotspot_prediction.csv
│   │       ├── J6_key_findings.csv
│   │       ├── xgboost_spatiotemporal.csv
│   │       └── lstm_spatiotemporal_metrics.csv
│
└── README.md
```

---

## A3: Analysis Pipeline

The notebook is structured into **4 parts**, each self-contained:

### Part 1 — EDA & Feature Engineering
- Loads and cleans the raw 6.17M record CSV
- Computes 19 engineered features across temporal, spatial, and weather categories
- Produces 9 EDA figures and 3 summary tables

### Part 2 — Temporal Analysis & Machine Learning
- **Time Series:** STL decomposition, ADF stationarity test
- **Forecasting:** ARIMA(0,1,1), SARIMA(1,1,1)(1,1,0)₁₂, ETS (4 variants), Prophet, LSTM
- **Classification:** Predicting severity (1–4) from temporal features — Gradient Boosting, Random Forest, Decision Tree, Logistic Regression
- **Regression:** Predicting event duration — Ridge, Random Forest Regressor
- **Temporal Mining:** L/M/H motif discovery, hour×DOW heatmaps

### Part 3 — Spatial Analysis & Machine Learning
- **Density:** Hexbin density map, 2D KDE
- **Autocorrelation:** Moran's I, Geary's C, LISA (Monte Carlo, 999 permutations)
- **Clustering:** K-Means (k=2–10, best k=4, silhouette=0.587), DBSCAN
- **Classification:** Spatial severity prediction — Gradient Boosting, Random Forest, Logistic Regression
- **Regression:** Geographically Weighted Regression (GWR) vs Global OLS
- **Other:** Empirical variogram, state choropleth, variogram range ~900 km

### Part 4 — Data Mining & Joint Inference
| Code | Technique | Key Result |
|---|---|---|
| M1 | Spatial Colocation (Participation Index) | Station↔Amenity PI=0.817 |
| M2 | Temporal Motif Mining (L/M/H) | HH support=0.296, HHHH=0.254 |
| M3 | Apriori Association Rules | TrafficSignal→Crossing lift=2.834 |
| M4 | Isolation Forest + LOF Anomaly Detection | 334 consensus anomalies, avg 202h duration |
| M5 | Temporal Anomaly (Z-score + IQR + Rolling) | COVID-19 dip flagged Apr–Jun 2020 |
| M6 | State Co-activation & Transition Sequences | AZ↔OR highest co-activation lift=1.191 |
| M7 | Getis-Ord G* Hotspot Analysis | S. Florida z=7.90 (99% confidence) |
| M8 | Regional Cluster Profiling (6 regions) | Southeast: highest count; Midwest: longest duration |
| J1 | Spatio-Temporal Hotspot Heatmap | Philadelphia corridor peaks Sep–Oct |
| J2 | Joint Feature Set Comparison (7 combos) | Spatial+Temporal best: acc=0.716, R²=0.393 |
| J3 | Spatial-Temporal Interaction Analysis | Urban+rush events 2× longer duration |
| J4 | 3D DBSCAN (lat, lon, time) | Midwest summer cluster, 19,899 events |
| J5 | Predictive Hotspot Model | GB accuracy=0.898, F1-macro=0.866 |

---

## Key Results Summary

### Temporal Forecasting (15-month test set)
| Model | RMSE | MAPE (%) | Best? |
|---|---|---|---|
| ARIMA(0,1,1) | 45,271 | 77.24 | |
| Simple ES | 26,009 | 43.28 | |
| Holt-Winters (mult.) | 29,883 | 34.83 | |
| Prophet (multiplicative) | 25,405 | 20.62 | |
| **SARIMA(1,1,1)(1,1,0)₁₂** | **18,293** | **14.19** | ✅ |
| LSTM (look-back=6) | 28,652 | 29.33 | |

### Spatial Classification — Severity Prediction
| Model | Accuracy | F1-Macro |
|---|---|---|
| Gradient Boosting | 0.876 | 0.475 |
| Random Forest | 0.651 | 0.430 |
| Logistic Regression | 0.543 | 0.317 |

### Joint Feature Set — Duration Regression (R²)
| Feature Set | R² |
|---|---|
| Weather Only | 0.033 |
| Spatial Only | 0.145 |
| Temporal Only | 0.331 |
| **Spatial + Temporal** | **0.393** |
| Joint (All) | 0.390 |

---

## Tech Stack

```
Python 3.12 (Kaggle)
├── pandas, numpy, pyarrow
├── matplotlib, seaborn
├── scikit-learn          — ML models, clustering, PCA
├── statsmodels           — ARIMA, SARIMA, ETS, ADF, ACF/PACF
├── prophet               — Facebook Prophet forecasting
├── tensorflow/keras      — LSTM models
├── xgboost               — XGBoost joint model
├── scipy                 — KDE, variogram, stats
└── mlxtend               — Apriori association rule mining
```


