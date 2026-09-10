import selenium
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import time
from newspaper import Article
import os
import requests
import io
import PyPDF2
import pandas as pd
import csv
import sys
from datetime import datetime, timedelta

# Add path to timing utils
sys.path.append('/Users/arshiaarya/factchecking')
from timing_utils import timer, TimingBlock, TimingLogger

# Initialize logger with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
timer_logger = TimingLogger(f"scraping_times_{timestamp}.log")

# Setup WebDriver with Chrome - Modified for headless
with TimingBlock("Browser setup"):
    options = Options()
    options.add_argument("--incognito")
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    prefs = {
        "download.default_directory": os.path.join(os.getcwd(), "Downloads"),
        "plugins.always_open_pdf_externally": True
    }
    options.add_experimental_option("prefs", prefs)
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

@timer
def google_search(query, start_date, end_date):
    with TimingBlock(f"Google search for: {query[:50]}..."):
        driver.get("https://www.google.com")
        try:
            wait = WebDriverWait(driver, 10)
            search_box = wait.until(EC.presence_of_element_located((By.NAME, "q")))
            search_box.clear()
            date_filtered_query = f"{query} after:{start_date} before:{end_date}"
            search_box.send_keys(date_filtered_query)
            search_box.send_keys(Keys.RETURN)
            time.sleep(2)  # Added small delay
            wait.until(EC.presence_of_all_elements_located((By.ID, "search")))
            results = driver.find_elements(By.CSS_SELECTOR, "div.g")
            print(f"Number of results found: {len(results)}")
            search_results = []
            for result in results:
                try:
                    title_element = result.find_element(By.CSS_SELECTOR, "h3")
                    link_element = result.find_element(By.CSS_SELECTOR, "a")
                    title = title_element.text
                    link = link_element.get_attribute("href")
                    search_results.append({"title": title, "link": link})
                except Exception as e:
                    print(f"Error extracting result: {e}")
            return search_results
        except Exception as e:
            print(f"Error searching Google: {e}")
            return []

@timer
def extract_article_text(url):
    with TimingBlock(f"Article extraction from: {url[:50]}..."):
        article = Article(url)
        article.download()
        article.parse()
        return article.text

@timer
def process_and_write_results(claim, query, start_date, end_date, results, csv_writer):
    with TimingBlock(f"Processing results for claim: {claim[:50]}..."):
        for idx, result in enumerate(results):
            print(f"{idx + 1}. {result['title']}: {result['link']}")
            if 'link' in result.keys() and "http" in result['link']:
                if result['link'].endswith('.pdf'):
                    try:
                        with TimingBlock("PDF processing"):
                            response = requests.get(result['link'])
                            file = io.BytesIO(response.content)
                            reader = PyPDF2.PdfFileReader(file)
                            contents = reader.getPage(0).extract_text()
                            csv_writer.writerow([claim, query, start_date, end_date, result['title'], result['link'], contents])
                    except Exception as e:
                        print(f"Error extracting PDF text: {e}")
                else:
                    try:
                        article_text = extract_article_text(result['link'])
                        csv_writer.writerow([claim, query, start_date, end_date, result['title'], result['link'], article_text])
                    except Exception as e:
                        print(f"Error extracting article text: {e}")

# Main execution
if __name__ == '__main__':
    with TimingBlock("Total script execution"):
        # Read the CSV file
        with TimingBlock("Loading input data"):
            df = pd.read_csv("merged_claims_queries.csv")
            print(f"Total rows in DataFrame: {len(df)}")
            # Take only first 5 unique claims
            unique_claims = df['Claim'].unique()[:5]
            df = df[df['Claim'].isin(unique_claims)]
            print(f"Rows after filtering for first 5 claims: {len(df)}")

        # Directory setup
        with TimingBlock("Directory setup"):
            dir_name = "search_results"  # Changed directory name
            os.makedirs(dir_name, exist_ok=True)

            # Start from the last saved file
            file_num = len(os.listdir(dir_name))
            print(f"Starting from file number: {file_num}")

            # Open a new CSV file for writing
            current_file = os.path.join(dir_name, f"data_{file_num + 1}.csv")
            csv_file = open(current_file, 'w', newline='', encoding='utf-8')
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(['claim', 'query', 'start_date', 'end_date', 'title', 'link', 'article_text'])

        row_count = 0

        # Process claims and queries
        with TimingBlock("Processing all claims and queries"):
            for index, row in df.iterrows():
                with TimingBlock(f"Processing row {index}"):
                    query = row['Query'].replace('"', '')
                    claim = row['Claim']
                    start_date = row['claim_date']
                    end_date = row['end_date']
                    
                    if start_date and end_date:
                        try:
                            results = google_search(query, start_date, end_date)
                            print(f"Processing query: '{query}' for dates {start_date} to {end_date}")
                            
                            process_and_write_results(claim, query, start_date, end_date, results, csv_writer)
                            
                            row_count += len(results)
                            time.sleep(2)  # Added delay between queries
                            
                        except Exception as e:
                            print(f"Error processing row {index}: {e}")
                            continue
                    
                    if row_count >= 100:
                        csv_file.close()
                        file_num += 1
                        current_file = os.path.join(dir_name, f"data_{file_num + 1}.csv")
                        csv_file = open(current_file, 'w', newline='', encoding='utf-8')
                        csv_writer = csv.writer(csv_file)
                        csv_writer.writerow(['claim', 'query', 'start_date', 'end_date', 'title', 'link', 'article_text'])
                        row_count = 0

        # Cleanup
        with TimingBlock("Cleanup"):
            csv_file.close()
            driver.quit()

        print("Processing complete. Results saved in directory.")
