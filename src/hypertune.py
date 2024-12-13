from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
import pandas as pd
import mlflow
import mlflow.sklearn

# Load the dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='target')

# Splitting into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the RandomForest classifier model
rf = RandomForestClassifier(random_state=42)

# Define the parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [10, 50, 100],
    'max_depth': [None, 10, 20, 30]
}

# Apply GridSearchCV
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

# Set the MLflow tracking URI
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment('breat_cancer')


# Start an MLflow run  ( It will track only the best model that we get at the end of the training)
# with mlflow.start_run():
#     # Fit the GridSearchCV model
#     grid_search.fit(X_train, y_train)

#     # Get the best parameters and score
#     best_params = grid_search.best_params_
#     best_score = grid_search.best_score_

#     # Log parameters and metrics
#     mlflow.log_params(best_params)
#     mlflow.log_metric("accuracy", best_score)

#     # Log training and testing datasets as artifacts
#     train_df = X_train.copy()
#     train_df['target'] = y_train
#     train_df.to_csv("training_data.csv", index=False)
#     mlflow.log_artifact("training_data.csv")

#     test_df = X_test.copy()
#     test_df['target'] = y_test
#     test_df.to_csv("testing_data.csv", index=False)
#     mlflow.log_artifact("testing_data.csv")

#     # Log the best model
#     mlflow.sklearn.log_model(grid_search.best_estimator_, "random_forest_model")

#     # Set tags
#     mlflow.set_tag("author", "Alok")

#     print("Best Parameters:", best_params)
#     print("Best Score:", best_score)



# to track all the params that we pass in the training phase
with mlflow.start_run() as parent:
    grid_search.fit(X_train, y_train)

    # log all the child runs
    for i in range(len(grid_search.cv_results_['params'])):

        with mlflow.start_run(nested=True) as child:
            mlflow.log_params(grid_search.cv_results_["params"][i])
            mlflow.log_metric("accuracy", grid_search.cv_results_["mean_test_score"][i])

    # Displaying the best parameters and the best score
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    # Log params
    mlflow.log_params(best_params)

    # Log metrics
    mlflow.log_metric("accuracy", best_score)

    # Log training data
    train_df = X_train.copy()
    train_df['target'] = y_train

    train_df = mlflow.data.from_pandas(train_df)
    mlflow.log_input(train_df, "training")

    # Log test data
    test_df = X_test.copy()
    test_df['target'] = y_test

    test_df = mlflow.data.from_pandas(test_df)
    mlflow.log_input(test_df, "testing")

    # Log source code
    mlflow.log_artifact(__file__)

    # Log the best model
    mlflow.sklearn.log_model(grid_search.best_estimator_, "random_forest")

    # Set tags
    mlflow.set_tag("author", "Vikash Das")

    print(best_params)
    print(best_score)