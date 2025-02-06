# Stock Market Data Analysis, Visualization, and Predictive Modeling

## Overview
This project aims to analyze and predict the closing prices of six stocks over three different time ranges using Python. The selected stocks are:

- **Berkshire Hathaway (BRK-A & BRK-B)**
- **Oracle (ORCL)**
- **Meta (META)**
- **Tesla (TSLA)**
- **IBM (IBM)**
- **Coca-Cola (KO)**

Berkshire Hathaway has two stock classes (A and B), making it a total of six stocks analyzed in this project.

### **Time Ranges and Prediction Goals**
The predictions are made over three different time ranges:

1. **Short Range**: 1 month of data with a 2-minute interval. The model predicts stock prices for the next **30 minutes**.
2. **Medium Range**: 2 years of data with a 1-hour interval. The model predicts stock prices for the next **50 hours**.
3. **Long Range**: Maximum available data with a 1-day interval. The model predicts stock prices for the next **year**.

## **How to Run the Code**
The project consists of three main scripts:

- **`short_range_data_prediction.ipynb`**: Predicts stock prices for short-range data.
- **`medium_range_data_prediction.ipynb`**: Predicts stock prices for medium-range data.
- **`long_range_data_prediction.ipynb`**: Predicts stock prices for long-range data.

### **Steps to Run**
1. Ensure all required dependencies (listed below) are installed.
2. Place **`stocks_prediction_functions.py`** in the same directory as the main scripts.
3. Run any of the three scripts in any order:
   ```sh
   python short_range_data_prediction.ipynb
   python medium_range_data_prediction.ipynb
   python long_range_data_prediction.ipynb
   ```
4. The scripts will update and modify the datasets in **`/Datasets/`** and models in **`/Models/`** based on live data.

## **How to Use the Dashboard**
1. After making sure all the necessary libraries are installed, open **command prompt** or **powershell**.
2. Change the directory to **'../Stock Market Analysis/Dashboard'**.
3. Then type run **`streamlit run dashboard.py`**. This will lead you to a localhost webpage.
4. From the sidebar, choose the ticker of the stock you want. 
5. Choose the range of data you want to see from the side bar. This will display all hte plots and summary statistics in the dashboard.
6. Again from the sidebar, choose the range of prediction you want to see.
7. Click on the **Run Stock Prediction**.
8. Enjoy insights into the future!

### **Important Notes**
- Some models might take a long time to train and validate from scratch.
- **`long_range_data_prediction.py`** may take from **30 minutes to 3 hours** to run.
- If you are using anaconda environment, you have to activate your environment in **Anaconda Prompt**, navigate to **`dashboard.py`**'s directory, then type **`streamlit run dashboard.py`**.
- The dashboard may take up to 5 min when predicting long range data.

## **Dependencies**
To run this project, you need the following Python libraries:

```plaintext
numpy           # Numerical computations
pandas          # Data handling and processing
yfinance        # Fetching stock market data
scikit-learn    # Machine learning and preprocessing
prophet         # Time series forecasting
matplotlib      # Data visualization
mplfinance      # Candlestick chart visualization
seaborn         # Statistical data visualization
ta              # Technical analysis indicators
workalendar     # Handling market holidays
streamlit       # dashboard development
```
Install dependencies using:
```sh
pip install -r requirements.txt
```

## **Acknowledgments and Contributions**
We extend our heartfelt gratitude to Dr. Menore Tekeba of Addis Ababa University, Ethiopia, for providing the inspiration for this project, overseeing our progress, and for guiding us throughout the entire process with invaluable insights and mentorship.