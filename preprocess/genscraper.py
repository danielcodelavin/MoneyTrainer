from pygooglenews import GoogleNews
from datetime import datetime, timedelta
import csv
from typing import List, Dict
import time
import re
import csv
import os
from crawl4ai import AsyncWebCrawler
import asyncio
import logging
import json

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

############### PLAY FUNCTION; TOUCH ################
def play_scrape_articles(keywords: List[str], end_date: str, end_time: str, days_back: int = 3, max_articles_per_keyword: int = 10) -> List[Dict[str, str]]:
    end_datetime = datetime.strptime(f"{end_date} {end_time}", '%Y-%m-%d %H:%M:%S')
    start_datetime = end_datetime - timedelta(days=days_back)
    
    gn = GoogleNews(lang='en', country='US')
    all_articles = []
    
    for keyword in keywords:
        search = gn.search(keyword, from_=start_datetime.strftime('%Y-%m-%d'), to_=end_datetime.strftime('%Y-%m-%d'))
        
        article_count = 0
        for entry in search['entries']:
            pub_date = datetime.strptime(entry.published, '%a, %d %b %Y %H:%M:%S %Z')
            if start_datetime <= pub_date <= end_datetime:
                article = {
                    'title': entry.title,
                    #'date': pub_date.strftime('%Y-%m-%d %H:%M:%S'),
                    #'source': entry.source.title,
                    #'content': _clean_text(entry.summary),
                    #'link': entry.link
                }
                all_articles.append(article)
                article_count += 1
                if article_count >= max_articles_per_keyword:
                    break
            time.sleep(1)  # Be polite, wait a second between requests
    
    return all_articles








################## FUNCTION WORKS NEVER TOUCH ####################
def scrape_articles(keywords: List[str], end_date: str, end_time: str, days_back: int = 3) -> List[Dict[str, str]]:
    end_datetime = datetime.strptime(f"{end_date} {end_time}", '%Y-%m-%d %H:%M:%S')
    start_datetime = end_datetime - timedelta(days=days_back)
    
    gn = GoogleNews(lang='en', country='US')
    all_articles = []
    
    for keyword in keywords:
        search = gn.search(keyword, from_=start_datetime.strftime('%Y-%m-%d'), to_=end_datetime.strftime('%Y-%m-%d'))
        
        for entry in search['entries']:
            pub_date = datetime.strptime(entry.published, '%a, %d %b %Y %H:%M:%S %Z')
            if start_datetime <= pub_date <= end_datetime:
                article = {
                    'title': entry.title,
                    'date': pub_date.strftime('%Y-%m-%d %H:%M:%S'),
                    'source': entry.source.title,
                    'content': _clean_text(entry.summary),
                    'link': entry.link
                }
                all_articles.append(article)
            time.sleep(1)  # Be polite, wait a second between requests
    
    return all_articles

def save_to_csv(articles: List[Dict[str, str]], filename: str):
    with open(filename, 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=['title', 'date', 'source', 'content', 'link'])
        writer.writeheader()
        for article in articles:
            writer.writerow(article)

def play_save_to_csv(articles: List[Dict[str, str]], filename: str):
    with open(filename, 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=['title'])
        writer.writeheader()
        for article in articles:
            writer.writerow(article)

def _clean_text(text: str) -> str:
    # Remove HTML tags
    cleaned_text = re.sub(r'<[^>]+>', '', text)
    # Remove all symbols except full stops and commas
    cleaned_text = re.sub(r'[^\w\s.,]', '', cleaned_text)
    # Remove extra whitespace
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text

