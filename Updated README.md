# Restaurant Success Prediction

## Overview
This project aims to predict the success of restaurants based on features like ratings, cost, location, and more. "Success" is defined as a restaurant achieving a rating of 4.0 or higher.
Using machine learning, this project identifies key factors driving restaurant success and provides predictions that could help restaurant owners, investors, or analysts.

## Dataset
The dataset is sourced from Zomato and includes detailed restaurant information. Key features are:
- **Ratings**: The restaurant’s rating (e.g., "4.1/5").
- **Approximate Cost**: Cost for two people dining.
- **Location**: Geographic location of the restaurant.
- **Restaurant Type**: Category (e.g., casual dining, cafe).
- **Cuisines**: Types of cuisine offered.
- **Votes**: Number of votes/reviews (indicating popularity).
- Additional features: Online ordering and table booking availability.

## Approach

### Data Cleaning
- **Ratings**: The `rate` column is processed to extract numerical values (e.g., "4.1/5" to 4.1). Invalid entries (e.g., "NEW") are replaced with NaN and filled with the mean.
- **Cost and Votes**: `approx_cost(for two people)` and `votes` are converted to numeric, with non-numeric values set to NaN and filled with mean or zero.
- **Categorical Variables**: Features like `online_order`, `book_table`, `location`, `rest_type`, and `listed_in(type)` are encoded using `LabelEncoder`.

### Feature Engineering
- **Primary Cuisine**: The `cuisines` column is simplified by taking the first cuisine and encoding it numerically.
- **Success Label**: A binary target `success` is created:
  - `1` = Rating >= 4.0 (successful)
  - `0` = Rating < 4.0 (not successful)

 ### Model
 - **Random Forest Classifier**: This algorithm is used due to its ability to handle mixed data types (numeric and categorical), its resistance to overfitting, and its capability
   to rank feature importance. The model is configured with `class_weight='balanced'` to address potential imbalances in the success labels.
 

### Evaluation
The model’s performance is assessed using:
- **Confusion Matrix**: Visualizes true positives, false positives, true negatives, and false negatives.
- **Classification Report**: Provides precision, recall, and F1-score for both classes.
- **Accuracy Score**: Measures overall prediction accuracy.
- **Cross-Validation**: 5-fold cross-validation ensures robust performance evaluation across multiple data splits.
- **Feature Importance**: Identifies key drivers of success.

## Results
- The Random Forest model achieves an accuracy of **[insert your actual accuracy here, e.g., 87.5%]** (based on test set and cross-validation).
- Feature importance highlights `votes` (popularity) and `location` as top predictors, suggesting customer engagement and geographic placement are critical.
- Cross-validation confirms model reliability with an average accuracy of **[insert your Day 3 cross-val mean here, e.g., 86.8%]** and low variability (standard deviation **[insert std here, e.g., 2.1%]**).
- Visualizations (e.g., feature importance plots, confusion matrix heatmaps) provide deeper insights.

## Dependencies
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

## Outputs
- Model performance metrics (confusion matrix, classification report, accuracy, cross-validation scores).
- Feature importance rankings.
- Visualizations displayed via Matplotlib/Seaborn.

## Visualizations
- **Feature Importance Plot**: Highlights top factors influencing success.
- **Confusion Matrix Heatmap**: Shows prediction breakdown.
- **Distribution of Ratings**: Illustrates rating spread.
- **Cost vs. Success**: Examines cost-success relationship.
- **Success Rate by Location**: Compares success across locations.
