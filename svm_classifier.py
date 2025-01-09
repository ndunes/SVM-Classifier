import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# loading Breast Cancer dataset from scikit.learn
dat = load_breast_cancer()
feature_names = dat.feature_names
x, y = dat.data, dat.target
print(f"Features: {dat.feature_names}")
print(f"Target: {dat.target_names}")

# test train split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

# feature standardization
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# define hyperparameter grid for SVM
param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': [1, 0.1, 0.001],
        'kernel': ['linear', 'rbf']
}

# finding optimal parameters using GridSearchCV
grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 2, cv = 5)
grid.fit(x_train, y_train)

print("Best Parameters:", grid.best_params_)

grid_predictions = grid.predict(x_test)
print("Classification Report (Test Set):", classification_report(y_test, grid_predictions))

# confusion matrix
conf_matrix = confusion_matrix(y_test, grid_predictions)
disp = ConfusionMatrixDisplay(confusion_matrix = conf_matrix, display_labels = dat['target_names'])
disp.plot(cmap = 'viridis')
plt.title("Confusion Matrix (Test Set)")
plt.savefig('visuals/confusion_matrix.png')
plt.show()

# cross-validation to evaluate model performance and assess for overfitting
cv_scores = cross_val_score(grid.best_estimator_, x_train, y_train, cv = 5)
print("Cross-Validation Scores:", cv_scores)
print("Average Cross-Validation Score:", np.mean(cv_scores))

# examining coefficients for feature importance
model = grid.best_estimator_

coefficients = model.coef_[0]
x_train_df = pd.DataFrame(x_train, columns = feature_names)
features = x_train_df.columns

coef_df = pd.DataFrame({
    'Feature': features,
    'Coefficient': coefficients
 })

coef_df['Abs_Coefficient'] = coef_df['Coefficient'].abs()
coef_df = coef_df.sort_values(by = 'Abs_Coefficient', ascending = False)

# bar plot to visualize the top 10 features
import seaborn as sns

sorted_idx = np.argsort(np.abs(coefficients))[::-1]
top_idx = sorted_idx[:10]
top_features = x_train_df.columns[top_idx]
top_coefs = coefficients[top_idx]

plt.figure(figsize =  (10, 6))
sns.barplot(x = np.abs(top_coefs), y = top_features, palette = 'viridis')
plt.xlabel('Absolute Coefficient Value')
plt.title('Top Features Based on Absolute Coefficients')
plt.savefig('visuals/top_features_plot.png')
plt.show()









