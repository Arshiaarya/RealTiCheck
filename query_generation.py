import numpy as np
import pandas as pd
import os.path as osp
import openai
from openai import AzureOpenAI
import time
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Initialize AzureOpenAI client
client = AzureOpenAI(
    api_key=" ",  
    api_version="2023-03-15-preview",
    azure_endpoint = "https://misinformation.openai.azure.com/"
)
deployment_name = 'gpt-4o'
MAX_CONCURRENT_REQUESTS = 5
semaphore = threading.Semaphore(MAX_CONCURRENT_REQUESTS)
MAX_CALLS_PER_MINUTE = 5

def inference(prompt, max_retries=5):
    for attempt in range(max_retries):
        try:
            with semaphore:
                response = client.chat.completions.create(
                    model=deployment_name,
                    messages=[
                        {"role": "system", "content": "You are a fact checker looking for evidence to fact check a particular claim. You need to generate 5 search queries to find relevant information to support or refute the claim. Provide these queries as a JSON object with keys 'query1' through 'query5'."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0,
                    max_tokens=2048,
                    response_format={"type": "json_object"}
                )
                return json.loads(response.choices[0].message.content)
        except openai.RateLimitError:
            if attempt < max_retries - 1:
                logging.warning(f"Rate limit exceeded. Retrying in {(2 ** attempt) * 5} seconds.")
                time.sleep((2 ** attempt) * 5)  # Exponential backoff
            else:
                logging.error(f"Rate limit exceeded after {max_retries} attempts. Skipping this prompt.")
                return {}
        except Exception as e:
            logging.error(f"An error occurred: {str(e)}")
            return {}



def generate_search_queries(prompt):
    max_retries = 5
    retry_delay = 10
    
   
   
    queries_json = inference(prompt, max_retries)

    # Ensure we have exactly 5 queries
    for i in range(1, 6):
        key = f"query{i}"
        if key not in queries_json:
            queries_json[key] = f"Additional query for {prompt}"
    query_list = [queries_json[f"query{i}"] for i in range(1, 6)]
    return query_list


def process_claim(claim):
    prompt = f"""Generate N different questions to criticize the following claim: {claim} outrageously.
    If the given claim is not informative enough to generate a query, you should answer "none". N=5. The questions generated should be diverse and cover distinct aspects of the claim.
    Make sure each query is completely and individually related to the claim to provide context for the claim. Share the reason behind each query and why it will be useful for factchecking."""
    queries = generate_search_queries(prompt)
    return claim, queries

def claim_to_queries(claims):
    claim_to_query = {}
    with ThreadPoolExecutor(max_workers=MAX_CALLS_PER_MINUTE) as executor:
        future_to_claim = {executor.submit(process_claim, claim): claim for claim in claims}
        for future in tqdm(as_completed(future_to_claim), total=len(claims)):
            claim, queries = future.result()
            claim_to_query[claim] = queries
            print(claim, queries)

    # Create a list of tuples where each tuple contains a claim and a query
    claim_query_tuples = [(claim, query) for claim, queries in claim_to_query.items() for query in queries]
    
    # Create a DataFrame from the list of tuples
    claims_and_queries = pd.DataFrame(claim_query_tuples, columns=['Claim', 'Query'])
    print(claims_and_queries.head())
    print(claims_and_queries.shape)
    return claims_and_queries

if __name__ == '__main__':
    claims = [
    "During an appearance at the Economic Club of Chicago, former President Trump claimed that no one died as a result of the January 6 attack except Trump supporter Ashli Babbitt.",
    "A Marist poll shows that 58 percent of Trump supporters are 'very concerned' about noncitizens voting in the 2024 election, and nearly 90 percent are at least 'concerned' about this issue.",
    "Donald Trump and J.D. Vance promoted a false claim that Haitian migrants were stealing and eating pets in Springfield, Ohio.",
    "On October 4, Elon Musk shared a claim from a SpaceX employee stating that FEMA was \"actively blocking shipments and seizing goods and services locally and locking them away to claim them as their own.\" Donald Trump subsequently reposted Musk's claim on his Truth Social platform.",
    "A representative for Donald Trump's campaign stated that Detroit's population has decreased by over 60 percent since 1960 and that Detroit has one of the highest homicide rates in the United States.",
    "Since June 2023, Barack Obama has helped raise more than $80 million for the presidential campaign, according to his aides.",
    "North Carolina surpassed previous years' totals for the first day of early voting, except for the 2020 presidential election when 348,599 people voted on the first day.",
    "Despite the widespread destruction caused by Hurricane Helene last month, North Carolina managed to open 76 polling places in the 25 counties declared federal disaster areas, only four fewer than the planned 80.",
    "Donald Trump's 2024 presidential campaign has refused to release his medical records.",
    "According to South Korea's spy agency, North Korea has sent 1,500 soldiers to Russia for training.",
    "Earlier this month, Vice President Kamala Harris held a meeting with Arab-American leaders in Michigan, during which the participants urged her to adopt a stance different from President Biden's approach to the Israel conflict.",
    "During a Univision town hall, Donald Trump insisted that January 6, 2021, was a \"day of love\" and that \"nothing was done wrong at all.\"",
    "Kamala Harris, who is aiming to become the first Black woman president, has been courting Black male voters who were recently criticized by former President Barack Obama for showing interest in Donald Trump.",
    "Several Democratic U.S. Senate nominees in swing states are distancing themselves from Kamala Harris and praising former President Donald Trump to protect their own re-election chances.",
    "Former President Donald Trump heavily courted auto workers despite the industry's union endorsing Vice President Kamala Harris, a play which appears to be paying off; in Pennsylvania, he is tied or slightly ahead of Harris in recent polling.",
    "Vice President Kamala Harris is facing backlash over allegations that she committed plagiarism in her 2009 book, \"Smart on Crime: A Career Prosecutor's Plan to Make Us Safer.\"",
    "Austrian professor and plagiarism expert Stefan Weber told Fox News that Vice President Kamala Harris and her co-author Joan O'C Hamilton committed plagiarism 27 times in their book \"Smart on Crime.\"",
    "A recent Marquette University poll found that 70 percent of Americans believe the country is on the wrong track.",
    "In July, President Joe Biden was reportedly forced by fellow Democrats to step aside in favor of Vice President Kamala Harris, who now leads the Democratic ticket, fueling further speculation about internal party conflicts.",
    "Within hours of taking office, the Biden administration offered Congress a bill aimed at fixing the U.S. immigration system.",
    "Donald Trump struggled to read scripted notes written by his handlers and repeatedly complained about not being able to use a teleprompter.",
    "Maxim's endorsement of Donald J. Trump for President in the 2024 election was made in a direct post on X (formerly known as Twitter) from the magazine's official account, stating, 'Maxim endorses Donald J. Trump for President.'",
    "In September, Politico reported that the Harris campaign amended one of its first ads, which originally stated that Harris took a job at McDonald's to 'pay her way' through college, indicating potential inconsistencies in her story.",
    "Melania Trump's memoir has reached the number one spot on the New York Times bestseller list shortly after its release this week.",
    "Virginia men's basketball coach Tony Bennett has announced his immediate retirement, an unexpected decision made just before the season opener."
]
    

    #df = df.head(10)
   

    df_queries = claim_to_queries(claims)
    df_queries.to_csv("oct18_queries.csv", index=False)
    print(df_queries.head())
    #df_queries.to_csv("../dataset/AVeriTeC/data/dev_queries_json.csv", index=False)
