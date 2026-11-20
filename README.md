# AI Expense Category Predictor

A beginner-friendly machine-learning project that classifies expense descriptions and estimates next-month spending.

## Features
- Predicts a category from a short transaction description using TF-IDF + Logistic Regression.
- Saves transactions locally in a CSV file.
- Displays category breakdown and monthly spending charts.
- Lets you change a saved transaction's category from the transaction history.
- Estimates next month's spending using a simple linear trend.
- Imports and exports CSV transactions.
- Analyzes uploaded transactions with spending totals, averages, category breakdowns, and monthly trends.
- Includes demo data so the app can be tested immediately.

## Technology
- Python
- Streamlit (web interface)
- Pandas and NumPy (data handling)
- Scikit-learn (machine learning)
- Plotly (charts)
- Joblib (save/load trained model)

## Run the project

1. Install Python 3.10 or newer.
2. Extract the ZIP and open a terminal in the project folder.
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Train the model:

   ```bash
   python train_model.py
   ```

5. Start the application:

   ```bash
   streamlit run app.py
   ```

6. Open the local URL shown in the terminal (usually http://localhost:8501).

## Try it
- Click **Load demo transactions** in the sidebar to populate the dashboard.
- Add an expense such as "pizza dinner", "bus ticket", "electricity bill", or "movie ticket".
- Expand **Change an expenditure category** below the transaction history to reclassify a saved transaction.
- Use the predictor box to test a description without saving it.
- Upload a CSV with columns `date`, `description`, `amount`. A `category` column is optional. The app analyzes valid rows and predicts missing categories; select **Import transactions** to add them to your saved history.

## How the AI works
1. The training file contains example descriptions with known expense categories.
2. TF-IDF converts text descriptions into numerical features based on word importance.
3. Logistic Regression learns patterns connecting words to categories.
4. When a new description is entered, the model predicts the most likely category.
5. For forecasting, the app totals expenses month by month and extends the recent linear trend to estimate the next month.

## Important limitations
- The included training data is a small educational sample, not a production-grade financial dataset. Predictions may be wrong for unusual or ambiguous descriptions.
- The monthly forecast is a simple baseline. It needs several months of representative data to be useful and should not be treated as financial advice.
- Data is stored locally in `data/expenses.csv`; it is not encrypted and is not uploaded to a cloud database. Do not enter sensitive financial account details.

## Project structure

```text
ai_expense_category_predictor/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── expense_category_model.joblib   # generated after training
└── data/
    ├── training_expenses.csv
    └── sample_expenses.csv
```
