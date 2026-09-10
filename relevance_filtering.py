import concurrent.futures
from functools import partial
import threading
from tqdm import tqdm
import numpy as np
import pandas as pd
import os
from openai import AzureOpenAI
import time
import json
import glob
import logging
import csv
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("entailment_verification.log"),
        logging.StreamHandler()
    ]
)

# Initialize OpenAI client
client = OpenAI(
    api_key=""
)

model_name = 'gpt-4o'

# Create a semaphore for concurrent requests
MAX_CONCURRENT_REQUESTS = 5
semaphore = threading.Semaphore(MAX_CONCURRENT_REQUESTS)
# Lock for thread-safe file writing
file_lock = threading.Lock()

def entailment_verification_with_retry(prompt, max_retries=5):
    for attempt in range(max_retries):
        try:
            with semaphore:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant. Provide your response in JSON format with keys 'answer' (Relevant or Irrelevant) and 'justification' (a string with explanation)."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0,
                    max_tokens=2048,
                    response_format={"type": "json_object"}
                )
                return json.loads(response.choices[0].message.content)
        except Exception as e:
            if attempt < max_retries - 1:
                sleep_time = (2 ** attempt) * 5
                logging.warning(f"Error occurred. Retrying in {sleep_time} seconds... Error: {str(e)}")
                time.sleep(sleep_time)
            else:
                logging.error(f"Error after {max_retries} attempts. Skipping this prompt. Error: {str(e)}")
                return {}

def process_row(row, prompt_template, text_column, writer, output_file):
    try:
        evidence = row[text_column]
        claim = row['claim']
        prompt = prompt_template.format(evidence=evidence, claim=claim)
        result = entailment_verification_with_retry(prompt)
        
        relevance = result.get('answer', 'Error')
        justification = result.get('justification', 'Error in processing')
        
        # Create output row by combining input row with new results
        output_row = {**row, 'Relevance': relevance, 'Justification': justification}
        
        # Thread-safe writing to CSV
        with file_lock:
            writer.writerow(output_row)
        
        return relevance, justification
    except Exception as e:
        logging.error(f"Error processing row: {str(e)}")
        # Write error row
        with file_lock:
            output_row = {**row, 'Relevance': 'Error', 'Justification': f'Error: {str(e)}'}
            writer.writerow(output_row)
        return 'Error', 'Error in processing'

def read_prompt(inp):
    try:
        with open(inp, "r") as f:
            return f.read()
    except Exception as e:
        logging.error(f"Error reading prompt file: {str(e)}")
        return ""

def process_file(input_file, text_column, output_suffix):
    logging.info(f"Processing {input_file}...")
    
    try:
        df = pd.read_csv(input_file)
        logging.info(f"Loaded {len(df)} rows from {input_file}")
    except Exception as e:
        logging.error(f"Error loading {input_file}: {str(e)}")
        return

    # Read prompt template
    prompt_template = read_prompt("promptrelevance.txt")
    if not prompt_template:
        logging.error("Prompt template is empty. Exiting.")
        return

    # Prepare output file
    output_file = f"{os.path.splitext(input_file)[0]}_{output_suffix}.csv"
    
    # Get all column names for the output CSV
    output_columns = list(df.columns) + ['Relevance', 'Justification']
    
    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=output_columns)
            writer.writeheader()
            
            # Convert DataFrame to list of dictionaries
            rows_list = df.to_dict('records')
            
            # Create partial function with writer
            process_prompt = partial(
                process_row, 
                prompt_template=prompt_template, 
                text_column=text_column,
                writer=writer,
                output_file=output_file
            )

            # Process rows with progress bar
            results = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENT_REQUESTS) as executor:
                for result in tqdm(
                    executor.map(process_prompt, rows_list), 
                    total=len(rows_list),
                    desc=f"Processing {os.path.basename(input_file)}"
                ):
                    results.append(result)

            # Log relevance distribution
            relevance_counts = pd.Series([r[0] for r in results]).value_counts()
            logging.info("\nRelevance distribution:")
            for label, count in relevance_counts.items():
                logging.info(f"{label}: {count} ({count/len(df)*100:.1f}%)")
            
    except Exception as e:
        logging.error(f"Error during processing: {str(e)}")

def main():
    # Define processing configurations for each format
    formats = [
        {
            'input_file': 'labeled_output_real_time_data.csv',
            'text_column': 'article_text',
            'output_suffix': 'relevance'
        },
    ]

    # Process each format
    for format_config in formats:
        if os.path.exists(format_config['input_file']):
            logging.info(f"\nProcessing format: {format_config['input_file']}")
            process_file(
                format_config['input_file'],
                format_config['text_column'],
                format_config['output_suffix']
            )
        else:
            logging.warning(f"File not found: {format_config['input_file']}")

    logging.info("All processing complete!")

if __name__ == "__main__":
    main()
