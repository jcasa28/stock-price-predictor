#Importing Libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os 
from tqdm import tqdm

#Importing the Trained Models for Predictions 
from src.lstm import lstm_train_predict
from src.randomForest import randomForest_train_predict
from src.svr import svr_train_predict

#1) IMPORTING DATA ----------------------------------------------------------------------

#CODE HERE:

data=pd.read_csv('data/stocks_data.csv')

df = pd.DataFrame(data)


#2) EVALUATION FUNCTION -----------------------------------------------------------------

#CODE HERE:
def evaluate_model(model, stock, Y_test, Y_predictions):
    print(f"Evaluating {model} for {stock}...")

    r2 = r2_score(Y_test, Y_predictions)
    mse = mean_squared_error(Y_test, Y_predictions)
    print(f"R2 Score: {r2:.4f}")
    print(f"Mean Squared Error: {mse:.4f}")
    return r2, mse



#3) PREDICTING STOCK --------------------------------------------------------------------

#CODE HERE:
# Selecting a single stock with at least 1256 rows
stock_counts = data["Name"].value_counts()
selected_stock = stock_counts[stock_counts >= 1256].index[0]
print(f"Selected stock with at least 1256 entries: {selected_stock}")

data = data[data["Name"] == selected_stock].iloc[:1256]
stocks = [selected_stock]
print(f"Processing stock: {selected_stock} with {len(data)} rows")

features = ['open', 'high', 'low', 'volume', 'return','rolling_mean','rolling_std'] #Features to be used for prediction

targe='close' #Target variable


results = {
    'lstm_r2': [], 'lstm_mse': [],
    'randomForest_r2': [], 'randomForest_mse': [],
    'svr_r2': [], 'svr_mse': []
}

for stock in tqdm(stocks, desc="Processing stocks"):  # Progress bar for stock processing

    # ---- Stock Data Preparation ----

    #CODE HERE:
    stock_data = data[data['Name'] == stock]

    X=stock_data[features]
    y=stock_data[targe]
    dates=stock_data['date']
    

    #splitting the data into train and test sets
    X_train, X_test, y_train, y_test, dates_train, dates_test = train_test_split(X, y, dates, test_size=0.2, random_state=42)

    #Sorting the test data by date
    X_test_sorted = X_test.sort_index()
    y_test_sorted = y_test.sort_index()
    dates_test_sorted = pd.to_datetime(dates_test.sort_values())
    
    # ---- LSTM Model ----

    #CODE HERE:
    Y_predictions = lstm_train_predict(X_train, y_train, X_test_sorted)
    mse, r2 = evaluate_model('LSTM', stock, y_test_sorted, Y_predictions)
    

    results['lstm_r2'].append(r2)
    results['lstm_mse'].append(mse)

    plt.figure(figsize=(10, 6))
    plt.plot(dates_test_sorted, y_test_sorted.values, label='True Values', color='green')
    plt.plot(dates_test_sorted, Y_predictions, label='LSTM Predictions', color='purple')
    plt.title(f'{stock} - LSTM')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.savefig(os.path.join('Workshop 4/result/lstm_plots', f'{stock}_lstm.png'))
    plt.close()

    # ---- Random Forest Model ----

    #CODE HERE:
    Y_predictions = randomForest_train_predict(X_train, y_train, X_test_sorted)
    mse, r2 = evaluate_model('Random Forest', stock, y_test_sorted, Y_predictions)


    results['randomForest_r2'].append(r2)
    results['randomForest_mse'].append(mse)

    plt.figure(figsize=(10, 6))
    plt.plot(dates_test_sorted, y_test_sorted.values, label='True Values', color='green')
    plt.plot(dates_test_sorted, Y_predictions, label='Random Forest Predictions', color='purple')
    plt.title(f'{stock} - Random Forest')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.savefig(os.path.join('Workshop 4/result/randomForest_plots', f'{stock}_randomForest.png'))
    plt.close()

    # ---- SVR (Support Vector Regressor) Model ----

    #CODE HERE:
    Y_predictions = svr_train_predict(X_train, y_train, X_test_sorted)
    mse, r2 = evaluate_model('SVR', stock, y_test_sorted, Y_predictions)


    results['svr_r2'].append(r2)
    results['svr_mse'].append(mse)

    plt.figure(figsize=(10, 6))
    plt.plot(dates_test_sorted, y_test_sorted.values, label='True Values', color='green')
    plt.plot(dates_test_sorted, Y_predictions, label='SVR Predictions', color='purple')
    plt.title(f'{stock} - SVR')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.savefig(os.path.join('Workshop 4/result/svr_plots', f'{stock}_svr.png'))
    plt.close()

#4) PRINTING RESULTS --------------------------------------------------------------------
print("Results:")

# LSTM Average Results 
lstm_r2 = np.mean(results['lstm_r2'])
lstm_mse = np.mean(results['lstm_mse'])
print(f"LSTM Average R2 Score: {lstm_r2:.4f}")
print(f"LSTM Average Mean Squared Error: {lstm_mse:.4f}")

# Random Forest Average Results
randomForest_r2 = np.mean(results['randomForest_r2'])
randomForest_mse = np.mean(results['randomForest_mse'])
print(f"Random Forest Average R2 Score: {randomForest_r2:.4f}")
print(f"Random Forest Average Mean Squared Error: {randomForest_mse:.4f}")

# SVR Average Results
svr_r2 = np.mean(results['svr_r2'])
svr_mse = np.mean(results['svr_mse'])
print(f"SVR Average R2 Score: {svr_r2:.4f}")
print(f"SVR Average Mean Squared Error: {svr_mse:.4f}")