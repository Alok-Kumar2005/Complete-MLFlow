import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score , confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


mlflow.set_tracking_uri("http://127.0.0.1:5000")

# load the dataset 
wine = load_wine()
X = wine.data
y = wine.target

## train test split
X_train , X_test , y_train , y_test = train_test_split(X , y , test_size = 0.10 , random_state = 42)

# define the parameter for a RF model
max_depth = 5
n_estimators = 8

mlflow.set_experiment('mlflow-experiment1')   # way to create a experiment in mlflow

with mlflow.start_run():
    rf = RandomForestClassifier(max_depth = max_depth , n_estimators=n_estimators)
    rf.fit(X_train , y_train)

    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test , y_pred)

    mlflow.log_metric('accuracy' , accuracy)
    mlflow.log_param('max_depth' , max_depth)
    mlflow.log_param('n_estimators' , n_estimators)

    # creating a confusion_matrix plot
    cm = confusion_matrix(y_test , y_pred)
    plt.figure(figsize = (6,6))
    sns.heatmap(cm , annot=True , fmt = 'd' , cmap = 'Blues' , xticklabels=wine.target_names , yticklabels=wine.target_names)
    plt.ylabel('Actual')
    plt.xlabel('Prediction')
    plt.title('Confusion Matrix')

    # save plt
    plt.savefig('confusion_matrix.png')

    # log artifiacts using mlflow
    mlflow.log_artifact('confusion_matrix.png')
    mlflow.log_artifact(__file__)

    print(accuracy_score) 