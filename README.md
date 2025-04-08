# Restaurant Success Prediction

Welcome to the **Restaurant Success Prediction** project! This repository uses the Zomato dataset to predict restaurant ratings, a proxy for success, based on features like votes, 
cost, online ordering, table booking, location, and restaurant type. The project employs Linear Regression and Random Forest models, with comprehensive exploratory data analysis (EDA) 
and visualizations to interpret the results.

## Project Overview

- **Dataset**:restaurant data (`zomato.csv`).
- **Goal**: Predict restaurant ratings (`rate`) using machine learning.
- **Features**:
  - `votes`: Number of votes received.
  - `approx_cost(for two people)`: Estimated cost for two.
  - `online_order`: Availability of online ordering (Yes/No).
  - `book_table`: Table booking option (Yes/No).
  - `location`: Restaurant location.
  - `rest_type`: Type of restaurant (e.g., Casual Dining, Cafe).
- **Models**: Linear Regression, Random Forest Regressor.
- **Metrics**: R² Score, RMSE (Root Mean Squared Error), MAE (Mean Absolute Error).
- **Tech Stack**: Python, Pandas, Scikit-learn, Matplotlib, Seaborn.

## Repository Structure
restaurant-success-prediction/
├── data/
│   ├── zomato.csv              # Original Zomato dataset
│   └── zomato_preprocessed.csv # Preprocessed dataset
├── images/
│   ├── cuisine_distribution.png    # Top 10 cuisines bar chart
│   ├── restaurant_locations.png    # Top 10 locations bar graph
│   ├── restaurant_types_pie.png    # Top 5 restaurant types pie chart
│   ├── correlation_heatmap.png     # Numeric features correlation
│   ├── metrics_comparison.png      # Model metrics bar plot
│   ├── confusion_matrices.png      # Confusion matrices for both models
│   ├── predicted_vs_actual.png     # Predicted vs actual scatter plots
│   ├── residual_plots.png          # Residual plots for both models
│   └── error_distribution.png      # Error distribution histogram
├── src/
│   ├── preprocessing.py        # Data cleaning and encoding
│   ├── eda.py                  # Exploratory Data Analysis
│   ├── modeling.py             # Model training and evaluation
│   └── visualization.py        # Model performance visualizations
├── restaurant_model_metrics.csv # Model performance metrics
├── README.md                   # This file
└── requirements.txt            # Python dependencies
