import yfinance as yf
import pandas as pd
import os


def fetch_snp500_data(
    start_date="2016-01-01", end_date=None, output_folder="snp500_data"
):
    """
    Fetches historical data for each company in the S&P 500 index from Yahoo Finance
    and saves data to separate CSV files.

    Parameters:
    - start_date (str): Start date in the format 'YYYY-MM-DD'.
    - end_date (str): End date in the format 'YYYY-MM-DD'. Defaults to the current date.
    - output_folder (str): Output folder for CSV files. Defaults to 'snp500_data'.
    """
    # If end_date is not provided, set it to the current date
    if end_date is None:
        end_date = pd.to_datetime("today").strftime("%Y-%m-%d")

    # Get the list of S&P 500 companies with their corresponding tickers
    sp500_companies = pd.read_html(
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    )[0]

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    for index, row in sp500_companies.iterrows():
        ticker = row["Symbol"]
        company_name = row["Security"]

        try:
            # Fetch historical data using yfinance
            company_data = yf.download(ticker, start=start_date, end=end_date)

            # Extract date and closing price
            company_data = company_data[["Close"]].reset_index()

            # Save data to CSV
            output_csv = os.path.join(
                output_folder, f"{company_name}_{ticker}_data.csv"
            )
            company_data.to_csv(output_csv, index=False)

            print(f"Data for {company_name} ({ticker}) saved to {output_csv}")
        except Exception as e:
            print(f"Error fetching data for {company_name} ({ticker}): {str(e)}")


# Example usage
fetch_snp500_data()
