# Encodeing categorical variables
categorical_cols = ['online_order', 'book_table', 'location', 'rest_type', 'listed_in(type)']
for col in categorical_cols:
    df[col] = df[col].astype(str)  # Ensure all are strings
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# Handleing 'cuisines' by taking the first cuisine listed and encoding it
df['primary_cuisine'] = df['cuisines'].apply(
    lambda x: x.split(',')[0].strip() if pd.notna(x) and x.strip() != '' else 'Unknown'
)
le = LabelEncoder()
df['primary_cuisine'] = le.fit_transform(df['primary_cuisine'])

# Selecting features
features = ['online_order', 'book_table', 'location', 'rest_type', 'primary_cuisine', 'cost', 'votes']
X = df[features]
y = df['success']

# Ensure all features in X are numeric
for col in X.columns:
    if not pd.api.types.is_numeric_dtype(X[col]):
        print(f"Non-numeric column found: {col}")
        X[col] = pd.to_numeric(X[col], errors='coerce')
        X = X.dropna(subset=[col])
        y = y[X.index]  # Align y with X

# Checking target distribution
print("Target Distribution:")
print(y.value_counts())

# Spliting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

# Predicting and evaluate
y_pred = model.predict(X_test)
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Checking Accuracy Score
score =accuracy_score(y_test,y_pred)*100
print("\nAccuracy Score:")
print(f"{score:.2f}%")

# Feature Importance
feature_importance = pd.DataFrame({
    'Feature': features,
    'Importance': model.feature_importances_
})
print("\nFeature Importance:")
print(feature_importance.sort_values(by='Importance', ascending=False))
