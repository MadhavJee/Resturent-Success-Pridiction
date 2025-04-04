# Cross-Validation (moved to start for validation context)
scores = cross_val_score(model, X, y, cv=5)
print("\nCross-Validation Results:")
print(f"Individual Fold Scores: {scores}")
print(f"Average Accuracy: {scores.mean() * 100:.2f}%")
print(f"Standard Deviation: {scores.std() * 100:.2f}%")

# Feature Importance Plot
feature_importance = pd.DataFrame({
    'Feature': features,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Feature Importance in Predicting Restaurant Success')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.show()

# Confusion Matrix Heatmap
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Not Successful', 'Successful'], 
            yticklabels=['Not Successful', 'Successful'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix Heatmap')
plt.show()

# Distribution of Ratings
plt.figure(figsize=(8, 6))
sns.histplot(df['rate'].dropna(), bins=20, kde=True)  # Fixed 'rating' to 'rate'
plt.title('Distribution of Restaurant Ratings')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.show()

# Cost vs. Success Box Plot
plt.figure(figsize=(8, 6))
sns.boxplot(x='success', y='cost', data=df)
plt.title('Cost vs. Restaurant Success')
plt.xlabel('Success (0 = No, 1 = Yes)')
plt.ylabel('Cost for Two People')
plt.show()

# Success Rate by Location
location_success = df.groupby('location')['success'].mean().sort_values(ascending=False)
plt.figure(figsize=(12, 6))
location_success.plot(kind='bar')
plt.title('Success Rate by Location')
plt.xlabel('Location')
plt.ylabel('Success Rate')
plt.show()
