## Restaurant Success Prediction ##
 
 # Overview
 This project aims to predict the success of restaurants based on various features such as ratings, cost, location, and more. In this context, "success" is defined 
 as a restaurant achieving a rating of 4.0 or higher. By leveraging machine learning techniques, this project identifies key factors that drive restaurant success and 
 provides predictions that could benefit restaurant owners, investors, or analysts.
 
 # Dataset
 The dataset used in this project is sourced from Zomato and includes detailed information about restaurants. Key features in the dataset are:
 - Ratings: The rating of the restaurant (e.g., "4.1/5").
 - Approximate Cost: The cost for two people dining at the restaurant.
 - Location: The geographic location of the restaurant.
 - Restaurant Type: The type or category of the restaurant (e.g., casual dining, cafe).
 - Cuisines: The types of cuisine offered.
 - Votes: The number of votes or reviews received (indicating popularity).
 - And additional features like online ordering and table booking availability.
  
 ## Approach
 # Data Cleaning
 - Ratings: The `rate` column is processed to extract numerical values (e.g., converting "4.1/5" to 4.1). Non-numeric or invalid entries (e.g., "NEW") are replaced with NaN.
 - Cost and Votes: The `approx_cost(for two people)` and `votes` columns are converted to numeric types, with non-numeric values set to NaN and removed.
 - Categorical Variables: Features such as `online_order`, `book_table`, `location`, `rest_type`, and `listed_in(type)` are encoded into numeric format using `LabelEncoder`.
 
 # Feature Engineering
 - Primary Cuisine**: The `cuisines` column is simplified by extracting the first listed cuisine and encoding it numerically.
 - Success Label**: A binary target variable `success` is created, where:
   - `1` = Rating >= 4.0 (successful)
   - `0` = Rating < 4.0 (not successful)
 
 # Model
 - Random Forest Classifier: This algorithm is used due to its ability to handle mixed data types (numeric and categorical), its resistance to overfitting, and its capability
   to rank feature importance. The model is configured with `class_weight='balanced'` to address potential imbalances in the success labels.
 
 # Evaluation
 The model's performance is assessed using:
 - Confusion Matrix: To visualize true positives, false positives, true negatives, and false negatives.
 - Classification Report: Provides precision, recall, and F1-score for both classes.
 - Accuracy Score: Measures overall prediction accuracy.
 - Feature Importance: Identifies which features most strongly influence restaurant success.
 
 # Results
 - The Random Forest model achieves a solid accuracy (e.g., approximately 87% in sample runs), with balanced precision and recall for both successful and unsuccessful restaurants.
 - Feature importance analysis highlights `votes` (popularity) and `location` as key drivers of success, suggesting that customer engagement and geographic placement are critical factors.
 - Visualizations (e.g., feature importance plots and confusion matrix heatmaps) are included in the script output for deeper insights.
 
 - Required Libraries: 
   - pandas
   - numpy
   - scikit-learn
   - matplotlib
   - seaborn
 
 - The script outputs:
   - Model performance metrics (confusion matrix, classification report, accuracy).
   - Feature importance rankings.
   - Visualizations saved or displayed (depending on script configuration).
 
 # Visualizations
 The project generates several visualizations to aid interpretation:
 - Feature Importance Plot: Highlights the top factors influencing success.
 - Confusion Matrix Heatmap: Shows the breakdown of model predictions.
 - Distribution of Ratings: Illustrates how ratings are spread across the dataset.
 - Cost vs. Success: Examines the relationship between dining cost and success.
 - Success Rate by Location: Compares success rates across different locations.
