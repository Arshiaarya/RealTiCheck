import concurrent.futures
from functools import partial
import threading
from tqdm import tqdm
import numpy as np
import pandas as pd
import os
from openai import OpenAI
import time
import json
import glob
import logging
from dotenv import load_dotenv
import csv

# Load environment variables from a .env file
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

# Initialize the OpenAI client securely using environment variables
client = OpenAI(
    api_key=""
)

model_name = 'gpt-4o'

# Create a semaphore to limit the number of concurrent requests
MAX_CONCURRENT_REQUESTS = 5
semaphore = threading.Semaphore(MAX_CONCURRENT_REQUESTS)

def entailment_verification_with_retry(prompt, max_retries=5):
    for attempt in range(max_retries):
        try:
            with semaphore:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant. Provide your response in JSON format with keys 'answer' (True or False) and 'justification' (a string with cited sentence or sentences)."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0,
                    max_tokens=2048,
                    response_format={"type": "json_object"}
                )
                return json.loads(response.choices[0].message.content)
        except client.RateLimitError:
            if attempt < max_retries - 1:
                sleep_time = (2 ** attempt) * 5
                logging.warning(f"Rate limit exceeded. Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
            else:
                logging.error(f"Rate limit exceeded after {max_retries} attempts. Skipping this prompt.")
                return {}
        except Exception as e:
            logging.error(f"An error occurred: {str(e)}")
            return {}

def process_and_save_row(row, prompt_template, writer, original_columns):
    try:
        premise = row['sentence']
        hypothesis = row['claim']
        
        prompt = prompt_template.format(premise=premise, hypothesis=hypothesis)
        result = entailment_verification_with_retry(prompt)
        
        if 'answer' in result and 'justification' in result:
            entailment = result['answer']
            justification = result['justification']
        else:
            entailment = 'Error'
            justification = 'Error in processing'
            
        # Map the entailment value to new_label
        new_label_map = {
            True: "Supported",
            False: "Refuted",
            'True': "Supported",
            "False": "Refuted",
            'Error': 'Error',
            'not found': 'not found'
        }
        new_label = new_label_map.get(entailment, 'Error')
        
        # Create a new row with all original columns plus new ones
        output_row = {col: row[col] for col in original_columns}
        output_row.update({
            'Entailment': entailment,
            'Justification': justification,
            'new_label': new_label
        })
        
        # Write the row to CSV
        writer.writerow(output_row)
        return True
        
    except Exception as e:
        logging.error(f"Error processing row: {str(e)}")
        return False

def read_prompt(inp):
    try:
        with open(inp, "r") as f:
            return f.read()
    except FileNotFoundError:
        logging.error(f"Prompt file {inp} not found.")
        return ""
    except Exception as e:
        logging.error(f"Error reading prompt file: {str(e)}")
        return ""

def main():
    input_file = './merged_train_filtered/relevant_sentences.csv'
    output_file = './merged_train_filtered/relevant_sentences_2label.csv'
    
    try:
        # Read the CSV file and get the column names
        df = pd.read_csv(input_file)
        original_columns = df.columns.tolist()
        logging.info("CSV file loaded successfully.")
        
        # Create output file and write header
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            fieldnames = original_columns + ['Entailment', 'Justification', 'new_label']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            # Read the prompt template
            prompt_template = read_prompt("prompt-2label.txt")
            if not prompt_template:
                logging.error("Prompt template is empty. Exiting.")
                return
            
            # Process each row
            rows_processed = 0
            total_rows = len(df)
            
            with tqdm(total=total_rows) as pbar:
                for _, row in df.iterrows():
                    row_dict = row.to_dict()
                    success = process_and_save_row(row_dict, prompt_template, writer, original_columns)
                    if success:
                        rows_processed += 1
                    pbar.update(1)
            
            logging.info(f"Processing complete. {rows_processed}/{total_rows} rows processed successfully.")
            
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
