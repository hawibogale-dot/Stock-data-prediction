# Standard library Import
import os
import pickle


# Data Handling & Processing
import numpy as np
import pandas as pd
import yfinance as yf

# Machine Learning & Statistics
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler


# Time Series & Forecasting
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_plotly

# Visualization
import matplotlib.pyplot as plt
import mplfinance as mpf
import seaborn as sns

# Technical Analysis
import ta.momentum
import ta.trend
import ta.volatility

# Holiday Calendars
from workalendar.usa import UnitedStates



# A function to make the data collection easier
def get_data(tickers, period="2y", interval="1h"):
    """
    Fetches stock data for the specified tickers.

    Parameters:
    ----------
    tickers : list or str
        A list of ticker symbols or a single ticker symbol.
    period : str
        A string of periods accepted by yahoo finance API.
    interval : str
        A string of intervals accepted by yahoo finance API.

    Returns:
    -------
    pandas.DataFrame
        A DataFrame containing the stock data for the tickers.
    """
    data = yf.download(
        tickers = tickers,
        period = period,
        interval= interval
    )
    return data

# A function that explores the data to help in data cleaning
def general_description(data):
    """
    Generates and prints general descriptive statistics and information for data cleaning purposes.

    This function provides essential details about the DataFrame, such as head of the DataFrame, missing values,
    summary statistics, number of duplicates (if any), number of columns with erroneous negative values (if any) and skewness measure, to aid in data cleaning and preprocessing steps.

    Parameters
    ----------
    data : pandas DataFrame
        DataFrame that will be analyzed for general description. 

    Returns
    -------
    None
        This function prints various statistics and information.
    """
    print("Head of the data:\n\t", data.head())
    
    print("\n\nInformation on data types:\n\t")
    print(data.info())

    print("\n\nData shape:\n\t", data.shape)

    # printing description of data
    print(f"\n\nDescription:\n\t", data.describe())
    
    # looking at the number of missing values
    print(f"\n\nMissing values count:\n\t", data.isna().sum())
    
    number_of_duplicates = data.duplicated().sum()
    print("\n\nNumber of duplicates:\n\t", number_of_duplicates)

    # calculating skewness and seeing how many rows are highly skewed
    number_skewed_columns = (np.abs(data.skew()) > 1).sum()
    print("\n\nNumber of skewed columns:\n\t", number_skewed_columns)

    # checking to see if there are any any unrealistic negative values
    number_of_negatives = ((data < 0).sum() != 0).sum()
    print("\n\nNumber of columns with negative values:\n\t", number_of_negatives)

# A function for line and boxplots
def get_plots(data):
    """
    Generates and displays visualizations for stock price data.

    This function:
    1. Handles MultiIndex columns by flattening them if necessary.
    2. Plots a line chart showing the trend of closing prices over time.
    3. Plots a boxplot to visualize the distribution of stock prices.

    Parameters:
    data (pandas.DataFrame): A DataFrame containing stock price data with a DateTime index.
                             It must include a 'Close' column for the line plot.

    Returns:
    None: The function displays the plots directly.
    """
    # checking if the dataframe has multiple indexes
    if isinstance(data.columns, pd.MultiIndex):
        data = data.droplevel(axis=1, level=1)
    sns.lineplot(data, x=data.index, y="Close")
    plt.show()
    sns.boxplot(data)
    plt.show()

# A function to help clean data
def clean_data(data, relevel_column=True, fill_na=True, winsorize_outliers=True, drop_duplicates=True):
    """
    Cleans stock market data by optionally releveling column indices, filling missing values, 
    and applying IQR-based Winsorization to handle outliers.

    Parameters:
    -----------
    data : pandas.DataFrame
        The input DataFrame containing stock data.
    relevel_column : bool, optional (default=True)
        If True, drops the second-level column index if present.
    fill_na : bool, optional (default=True)
        If True, forward-fills missing values in the dataset.
    winsorize_outliers : bool, optional (default=True)
        If True, caps extreme values in the 'Close' and 'Volume' columns using an adaptive IQR-based Winsorization.

    Returns:
    --------
    pandas.DataFrame
        The cleaned DataFrame with the specified transformations applied.
    """

    # Drop second-level column index only if it exists
    if relevel_column and isinstance(data.columns, pd.MultiIndex):
        data = data.droplevel(axis=1, level=1)
        print("Columns re-leveled.")

    # Forward-fill missing values
    if fill_na and data.isna().sum().sum() > 0:
        data = data.ffill()
        print("Missing values filled with forward fill.")

    # Apply adaptive Winsorization based on IQR
    if winsorize_outliers:
        for column in ["Close", "Volume"]:
            Q1 = data[column].quantile(0.25)
            Q3 = data[column].quantile(0.75)
            IQR = Q3 - Q1
            low_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # Capping values outside the bounds
            data[column] = np.where(data[column] < low_bound, low_bound, data[column])
            data[column] = np.where(data[column] > upper_bound, upper_bound, data[column])
        print("Outliers winsorized.")
    
    # Drop duplicates
    if drop_duplicates and (data.duplicated().sum() > 0):
        data = data.drop_duplicates()
        print("Dropped duplicates.")
    return data

# A function to summarize closing prices at different intervals
def summary(data, five_days=True, week=True, month=True, year=True):
    """
    Computes and prints summary statistics for the closing prices of a stock over different time periods.

    Parameters:
    data (pandas.DataFrame): A DataFrame containing stock data with a DateTime index. 
                             It must have a 'Close' column representing the stock's closing prices.
    five_days (bool, optional): If True, prints summary statistics for the last 5 days. Default is True.
    week (bool, optional): If True, prints summary statistics for the last 7 days. Default is True.
    month (bool, optional): If True, prints summary statistics for the last 1 month. Default is True.
    year (bool, optional): If True, prints summary statistics for the last 1 year. Default is True.

    Returns:
    None: The function prints the summary statistics directly.
    """
    data = pd.DataFrame(data)  # Ensure data is a DataFrame

    if five_days:
        index_start = data.index.max() - pd.Timedelta(days=5)
        print("Five days' summary:\n", data.loc[index_start:]["Close"].describe(), "\n")

    if week:
        index_start = data.index.max() - pd.Timedelta(days=7)
        print("Last week's summary:\n", data.loc[index_start:]["Close"].describe(), "\n")

    if month:
        index_start = data.index.max() - pd.DateOffset(months=1) 
        print("Last month's summary:\n", data.loc[index_start:]["Close"].describe(), "\n")

    if year:
        index_start = data.index.max() - pd.DateOffset(years=1)
        print("Last year's summary:\n", data.loc[index_start:]["Close"].describe())

# A function to generate candlestick plots
def plot_candlestick(data):
    """
    Plots candlestick charts of stock prices at different time intervals:
    yearly, monthly, weekly, daily, and hourly.

    Parameters:
    data (pandas.DataFrame): A DataFrame containing stock data with a DateTime index.
                             It must have 'Open', 'High', 'Low', and 'Close' columns.

    Returns:
    None: Displays the candlestick plots.
    """
    # Ensure data is sorted
    data = data.sort_index()

    # Resample data for different time intervals
    resampled_data = {
        "Yearly": data.resample('Y').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'}),
        "Monthly": data.resample('M').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'}),
        "Weekly": data.resample('W').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'}),
        "Daily": data.resample('D').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'}),
        "Hourly": data.resample('H').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'})
    }

    # Plot candlestick charts for each interval
    for period, df in resampled_data.items():
        if df.dropna().empty:  # Skip empty dataframes
            continue
        print(f"Plotting {period} Candlestick Chart")
        mpf.plot(df.dropna(), type='candle', style='charles', title=f"{period} Candlestick Chart", ylabel='Price', volume=False)

# A function that explores the relations between different stocks
def stocks_correlation(data_A, data_B, ticker_A="Stock A", ticker_B="Stock B"):
    """
    Calculates and visualizes the correlation between the closing prices of two stocks.

    Parameters:
    - data_A (pd.DataFrame): DataFrame containing stock A's historical data with a "Close" column.
    - data_B (pd.DataFrame): DataFrame containing stock B's historical data with a "Close" column.
    - ticker_A (str, optional): Ticker symbol or name of stock A (default: "Stock A").
    - ticker_B (str, optional): Ticker symbol or name of stock B (default: "Stock B").

    Prints:
    - The Pearson correlation coefficient between the two stocks' closing prices.

    Displays:
    - A scatter plot with stock A's closing price on the x-axis and stock B's closing price on the y-axis.

    Example Usage:
    >>> stocks_correlation(meta_df, apple_df, "META", "AAPL")
    """
    correlation = data_A["Close"].corr(data_B["Close"])
    print(f"Correlation between {ticker_A} and {ticker_B}: {correlation:.2f}")
    
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=data_A["Close"], y=data_B["Close"], s=5)
    
    plt.xlabel(f"{ticker_A} Closing Price")
    plt.ylabel(f"{ticker_B} Closing Price")
    plt.title(f"Scatter Plot: {ticker_A} vs {ticker_B}")
    
    plt.show()

# A function to get columns with outliers
def get_columns_outliers(data):
    """
    Identifies and returns the column names that contain outliers in the input DataFrame.

    This function detects columns that have outliers based on inter-quartile range (IQR). 
    It returns a list of columns where outliers are detected.

    Parameters
    ----------
    data : pandas DataFrame
        The input DataFrame containing the data to be analyzed for outliers.

    Returns
    -------
    outlier_columns : list
        A list of column names (or multi-level column combinations if the DataFrame has a multi-level column index)
        that contain outliers.

    Notes
    -----
    The function assumes that outliers are defined as values beyond a certain threshold (1.5*IQR).
    """

    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - (1.5 * IQR)
    upper = Q3 + (1.5 * IQR)

    # seeing if a column has outliers or not
    outlier_data = ((data < lower) | (data > upper)).sum() != 0
    
    # getting a list of columns with outliers in them
    outlier_columns = outlier_data[outlier_data==True].index
    
    return outlier_columns
    
# A function to help in feature engineering and saving the data
def feature_engineering_and_save_data(data, save_data=True, file_name=None, ema_span=20, bb_window=20, macd_fast=12, macd_slow=26, atr_window=14):
    """
    Perform feature engineering on a stock market DataFrame by adding common technical indicators.

    This function adds the following features to the input DataFrame:
        - Exponential Moving Average (EMA) based on the closing price.
        - Relative Strength Index (RSI) based on the closing price.
        - Bollinger Bands (Lower and Upper) based on the closing price.
        - Moving Average Convergence Divergence (MACD) based on the closing price.
        - Average True Range (ATR) based on the high, low, and close prices.
    
    It also saves the data to /datasets/ 

    Parameters:
    ----------
    data : pandas.DataFrame
        Input DataFrame containing stock market data with the following columns:
        - 'Open', 'High', 'Low', 'Close', and 'Volume'.
        The DataFrame index must be a pandas DateTimeIndex.
    save_data: book (default=True)
        True if the final engineered data should be saved as a csv file.
    file_name : str
        The name with which the data will be saved ot the /Datasets/ directory.

    ema_span : int, optional (default=20)
        The window size for calculating the Exponential Moving Average (EMA).

    bb_window : int, optional (default=20)
        The window size for calculating Bollinger Bands (Lower and Upper).

    macd_fast : int, optional (default=12)
        The fast window size for calculating the MACD line.

    macd_slow : int, optional (default=26)
        The slow window size for calculating the MACD line.

    atr_window : int, optional (default=14)
        The window size for calculating the Average True Range (ATR).

    Returns:
    -------
    pandas.DataFrame
        A modified DataFrame with the added technical indicators as new columns.
        The first `max(ema_span, rsi_window, bb_window, macd_slow, atr_window)` rows are dropped 
        to account for incomplete rolling window calculations.

    Notes:
    -----
    - The function modifies the input DataFrame in place but also returns it for flexibility.
    - The function also fills generated missing values using mean/median.

    Example:
    --------
    >>> import pandas as pd
    >>> import yfinance as yf
    >>> from ta import add_all_ta_features
    >>> df = yf.download("AAPL", start="2023-01-01", end="2023-12-31")
    >>> df = feature_engineering(df)
    >>> print(df.head())
    """
    data["EMA"] = data["Close"].ewm(span=ema_span).mean()

    data["BB_Low"] = ta.volatility.bollinger_lband(data["Close"], window=bb_window)
    data["BB_High"] = ta.volatility.bollinger_hband(data["Close"], window=bb_window)
    data["MACD"] = ta.trend.macd(data["Close"], window_fast=macd_fast, window_slow=macd_slow)
    data["ATR"] = ta.volatility.average_true_range(data["High"], data["Low"], data["Close"], window=atr_window)

  
    # New calculated feature will have some missing values and outliers
    # the following code deals with them
    means  = data.mean().to_dict() # getting mean for each column
    medians = data.median().to_dict() # getting median for each column
    outlier_columns = get_columns_outliers(data) # getting columns with outliers

    # filling missing values accordingly
    for column in data.columns:
        if column in outlier_columns:
            data[column] = data[column].fillna(medians[column])
        else:
            data[column] = data[column].fillna(means[column])

    # Define the path to save the file
    file_path = f"Datasets/{file_name}.csv"

    if save_data == True:
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        # Save the dataframe to CSV
        data.to_csv(file_path)
        print(f"Data saved as {file_path}")
    
    return data

# A function to run prophet model
def run_prophet(
    data, 
    train = True,
    predict=True,
    prophet=None,
    cross_validate=True, 
    periods=50, 
    seasonality_mode='multiplicative', 
    changepoint_prior_scale=0.1, 
    seasonality_prior_scale=5.0, 
    n_changepoints=50, 
    yearly_seasonality=False, 
    weekly_seasonality=True, 
    daily_seasonality=True,
    initial='365 days',
    period='30 days', 
    horizon='90 days'
):
    """
    Train a Facebook Prophet model on stock market data, perform cross-validation, 
    and make future predictions.

    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing stock market data with 'Datetime', 'Close', 'Volume', 
        'MACD', and 'ATR' columns.
    train : bool, optional
        Whether to perform training on the model. Default is True.
    predict : bool, optional
        Whether to make future predictions. Default is True.
    prophet : a prophet model
        A model that can be used to make predictions with. Default is None as model can be trained from scratch.
    cross_validate : bool, optional
        Whether to perform cross-validation on the model. Default is True.
    periods : int, optional
        Number of future periods to predict. Default is 50.
    seasonality_mode : str, optional
        Prophet's seasonality mode ('additive' or 'multiplicative'). Default is 'multiplicative'.
    changepoint_prior_scale : float, optional
        Flexibility of the trend change points. Default is 0.1.
    seasonality_prior_scale : float, optional
        Strength of seasonality prior. Default is 5.0.
    n_changepoints : int, optional
        Number of trend changepoints. Default is 50.
    yearly_seasonality : bool, optional
        Enable or disable yearly seasonality. Default is False.
    weekly_seasonality : bool, optional
        Enable or disable weekly seasonality. Default is True.
    daily_seasonality : bool, optional
        Enable or disable daily seasonality. Default is True.
    initial : str, optional
        Initial training period for cross-validation. Default is '365 days'.
    period : str, optional
        Period between cutoff dates for cross-validation. Default is '30 days'.
    horizon : str, optional
        Forecast horizon for cross-validation. Default is '90 days'.

    Returns:
    --------
    prophet : Prophet
        Trained Prophet model.
    transformed_predictions : pd.DataFrame
        DataFrame containing log-transformed future predictions.
    predictions : pd.Series
        Actual predictions with logarithm reversed.
    """
    
    # Prepare the dataset
    df = data[['Close', 'Volume', 'MACD', 'ATR']].reset_index()
    df.rename(columns={'Datetime': 'ds', 'Date' : 'ds', 'Close': 'y'}, inplace=True)
    df['ds'] = df['ds'].dt.tz_localize(None)  # Remove timezone information
    df['y'] = np.log(df['y'])  # Apply log transformation for better trend modeling

    # Create a holiday effect based on US Federal holidays
    calendar = UnitedStates()
    holidays = []
    for year in range(data.index.min().year, data.index.max().year + 2):
        holidays.extend(calendar.holidays(year))
        
    # Convert holiday names to holiday dates
    holiday_dates = [holiday[0] for holiday in holidays]  # Extract dates from holiday objects
    holidays_df = pd.DataFrame({
        'holiday': 'earnings_release',
        'ds': pd.to_datetime(holiday_dates),
        'lower_window': -5,  # Effect starts 5 days before
        'upper_window': 5,   # Effect ends 5 days after
    })

    if train:    
        # Initialize Prophet model with provided parameters
        prophet = Prophet(
            seasonality_mode=seasonality_mode, 
            changepoint_prior_scale=changepoint_prior_scale, 
            seasonality_prior_scale=seasonality_prior_scale, 
            holidays=holidays_df,
            n_changepoints=n_changepoints, 
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=weekly_seasonality,
            daily_seasonality=daily_seasonality
        )
        
        # Add additional regressors
        prophet.add_regressor('Volume')
        prophet.add_regressor('MACD')
        prophet.add_regressor('ATR')
        
        # Train the model
        prophet.fit(df)
        print("Prophet model fitted.\n")
    
    # Perform cross-validation
    if cross_validate:
        print("Starting cross-validation...\n")
        df_cv = cross_validation(prophet, initial=initial, period=period, horizon=horizon)
        df_p = performance_metrics(df_cv)
        print(df_p.head())
        print("Cross-validation finished.\n")
    
    # Prediction phase
    if predict:
        print("Starting prediction...\n")
        
        def predict_regressor(data, regressor, periods):
            """
            Predicts future values of a given regressor using Prophet.

            Parameters:
            - data (pd.DataFrame): Historical dataset containing 'ds' and the specified regressor.
            - regressor (str): Name of the regressor column.
            - periods (int): Number of periods to predict into the future.

            Returns:
            - pd.DataFrame: DataFrame with predicted values for the specified regressor.
            """
            
            model = Prophet(
                seasonality_mode='multiplicative', 
                changepoint_prior_scale=0.1, 
                seasonality_prior_scale=5.0, 
                holidays=holidays_df,
                n_changepoints=50, 
                yearly_seasonality=False, 
                weekly_seasonality=True, 
                daily_seasonality=True
            )
            
            df_reg = data[['ds', regressor]].rename(columns={regressor: 'y'})
            model.fit(df_reg)
            
            future = model.make_future_dataframe(periods=periods)
            future[regressor] = data[regressor]
            
            prediction = model.predict(future)
            return prediction
        
        # Create future dataframe for prediction
        future = prophet.make_future_dataframe(periods=periods)
        
        # Predict each regressor separately
        future['Volume'] = predict_regressor(df, 'Volume', periods)['yhat']
        future['MACD'] = predict_regressor(df, 'MACD', periods)['yhat']
        future['ATR'] = predict_regressor(df, 'ATR', periods)['yhat']

        future = future.ffill()
        
        # Generate final predictions
        transformed_predictions = prophet.predict(future)

        # inverse-transforming predictions
        predictions = transformed_predictions.copy()
        for column in ["yhat", "yhat_lower", "yhat_upper"]:
            predictions[column] = np.exp(predictions[column])
        print("Finished prediction.")
    
    return prophet, transformed_predictions, predictions


# A function to save the prophet model
def save_prophet_model(model, model_name):
    """
    Save a trained Prophet model to a specified directory.

    Parameters:
    model (Prophet): The trained Prophet model to be saved.
    model_name (str): The name to be used for the saved model file (without file extension).

    The model will be saved as a .pkl file in the 'models' directory.
    If the 'models' directory doesn't exist, it will be created.

    Example:
    save_prophet_model(model, 'stock_prediction_model')
    """
    # Ensure the models directory exists
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Define the path to save the model
    model_path = os.path.join('models', f'{model_name}.pkl')
    
    # Save the Prophet model to the specified path
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Model saved as {model_path}")

