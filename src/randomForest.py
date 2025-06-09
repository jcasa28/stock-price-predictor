import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor

def randomForest_train_predict(x_train, y_train, x_test):

    #Create the model
    model = RandomForestRegressor()

    #Define hyoperparameters to Tune

    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['auto', 'sqrt'],
        'bootstrap': [True, False]
    }

    GridSearch = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)

    #Fit the model
    GridSearch.fit(x_train, y_train)

    model=GridSearch.best_estimator_

    #Make predictions
    predictions = model.predict(x_test)

    return predictions