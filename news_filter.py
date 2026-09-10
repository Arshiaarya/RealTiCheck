import csv
from urllib.parse import urlparse
import pandas as pd

def load_news_domains(news_csv_path):
    """Load news domains from US sources."""
    news_domains = set()
    with open(news_csv_path, 'r') as infile:
        reader = csv.reader(infile)
        for row in reader:
            url = row[1]
            country = row[5]
            if country != "US":
                continue
            domain = urlparse(url).netloc
            news_domains.add(domain)
    return news_domains

def filter_links_by_domain(input_file, news_domains):
    """Filter links from the input file based on news domains."""
    # Read the input CSV file
    df = pd.read_csv(input_file)
    
    # Create a mask for filtering
    domain_mask = df['link'].apply(lambda x: urlparse(x).netloc in news_domains)
    
    # Filter the dataframe
    filtered_df = df[domain_mask]
    
    # Create output filename
    output_file = input_file.replace('.csv', '_filtered.csv')
    
    # Save filtered data
    filtered_df.to_csv(output_file, index=False)
    
    return df, filtered_df, output_file

def main():
    # File paths
    news_websites = "news.csv"
    input_file = "../evidence_retrieval/merged_claims_queries_train_remaining_processed.csv"
    
    # Load news domains
    news_domains = load_news_domains(news_websites)
    print(f"Loaded {len(news_domains)} US news domains")
    
    # Filter links
    df, filtered_df, output_file = filter_links_by_domain(input_file, news_domains)
    print(f"Filtered data saved to: {output_file}")
    print(f"Original rows: {len(df)}")
    print(f"Filtered rows: {len(filtered_df)}")

if __name__ == "__main__":
    main()
