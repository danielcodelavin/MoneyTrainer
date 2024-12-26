import numpy as np
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import logging
import torch
from typing import List, Tuple
import os
from sklearn.feature_extraction.text import CountVectorizer
from nltk.util import ngrams
import nltk
from nltk.tokenize import word_tokenize
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime

# Download required NLTK data
nltk.download('punkt')

class FinancialTopicModel:
    def __init__(
        self,
        n_gram_range: Tuple[int, int] = (1, 3),
        min_topic_size: int = 10,    
        nr_topics: int = 100        
    ):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Comprehensive sentiment vocabulary
        self.sentiment_vocab = {
            'bullish': [
                # Market-specific positive terms
                "rally", "surge", "jump", "soar", "climb", "breakout", "upgrade",
                "outperform", "beat", "exceed", "strong", "positive", "growth",
                "momentum", "confidence", "optimistic", "robust", "solid",
                
                # General positive terms
                "excellent", "outstanding", "phenomenal", "impressive", "remarkable",
                "exceptional", "fantastic", "extraordinary", "stellar", "superb",
                "favorable", "beneficial", "advantageous", "promising", "encouraging",
                "successful", "thriving", "booming", "flourishing", "prosperous",
                "lucrative", "profitable", "rewarding", "valuable", "worthy",
                "superior", "premium", "prime", "optimal", "ideal",
                "innovative", "revolutionary", "groundbreaking", "pioneering",
                "leading", "dominant", "commanding", "powerful", "influential",
                
                # Achievement terms
                "milestone", "breakthrough", "achievement", "accomplishment",
                "victory", "triumph", "success", "win", "advancement", "progress",
                "improvement", "enhancement", "upgrade", "boost", "strengthen",
                
                # Stability terms
                "stable", "secure", "reliable", "dependable", "trustworthy",
                "consistent", "steady", "sustainable", "durable", "resilient",
                
                # Growth terms
                "expand", "develop", "evolve", "mature", "accelerate",
                "expedite", "streamline", "optimize"
            ],
            
            'bearish': [
                # Market-specific negative terms
                "plunge", "crash", "sink", "tumble", "drop", "downgrade",
                "underperform", "miss", "weak", "negative", "decline", "slowdown",
                "concern", "risk", "pessimistic", "bearish", "downturn",
                
                # General negative terms
                "disappointing", "concerning", "troubling", "worrying", "alarming",
                "problematic", "challenging", "difficult", "tough", "hard",
                "unfavorable", "adverse", "detrimental", "harmful", "damaging",
                "poor", "inadequate", "insufficient", "subpar", "mediocre",
                "inferior", "deficient", "lacking", "limited", "restricted",
                "troubled", "distressed", "struggling", "failing", "deteriorating",
                "worsening", "degrading", "declining", "diminishing", "decreasing",
                
                # Risk terms
                "risky", "dangerous", "hazardous", "threatening", "vulnerable",
                "exposed", "susceptible", "volatile", "unstable", "uncertain",
                "unpredictable", "unreliable", "questionable", "doubtful",
                
                # Problem terms
                "problem", "issue", "challenge", "obstacle", "barrier",
                "hindrance", "impediment", "setback", "difficulty", "complication",
                "crisis", "emergency", "disaster", "catastrophe", "calamity",
                
                # Failure terms
                "fail", "failure", "collapse", "breakdown", "malfunction",
                "default", "bankruptcy", "insolvency", "loss", "deficit",
                "shortfall", "shortage", "scarcity", "constraint", "strike"
            ],
            
            'metrics': [
                # Financial metrics
                "earnings", "revenue", "profit", "margin", "sales", "guidance",
                "forecast", "outlook", "estimate", "target", "performance",
                "results", "returns", "yield", "dividend", "valuation",
                "price", "cost", "expense", "debt", "liability", "asset",
                "capital", "investment", "funding", "financing", "cash flow",
                "balance", "income", "growth rate", "ratio",
                
                # Market metrics
                "volume", "volatility", "momentum", "trend", "pattern",
                "average", "index", "benchmark", "indicator", "signal",
                "measure", "metric", "rating", "rank", "score",
                
                # Business metrics
                "market share", "customer base", "user growth", "adoption rate",
                "retention", "efficiency", "productivity", "output",
                "capacity", "utilization", "overhead", "profitability"
            ],
            
            'magnitude': [
                # Intensity
                "significantly", "substantially", "considerably", "notably",
                "markedly", "dramatically", "vastly", "hugely", "tremendously",
                "extraordinarily", "exceptionally", "extremely", "incredibly",
                
                # Moderate
                "moderately", "reasonably", "fairly", "relatively", "somewhat",
                "partially", "slightly", "marginally", "gradually", "steadily",
                
                # Comparative
                "more", "less", "higher", "lower", "better", "worse",
                "stronger", "weaker", "faster", "slower", "greater", "lesser"
            ]
        }
        
        # Stop words as a list
        self.custom_stop_words = [
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
            'for', 'of', 'with', 'by', 'from', 'up', 'down', 'that', 'this',
            'these', 'those', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'should', 'could', 'might', 'must', 'shall'
        ]
        
        # Initialize vectorizer
        self.vectorizer = CountVectorizer(
            ngram_range=n_gram_range,
            stop_words=self.custom_stop_words,
            vocabulary=self._create_sentiment_vocabulary(),
            min_df=3,
            max_df=0.95
        )
        
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        self.topic_model = BERTopic(
            embedding_model=self.sentence_model,
            vectorizer_model=self.vectorizer,
            n_gram_range=n_gram_range,
            min_topic_size=min_topic_size,
            nr_topics=nr_topics,
            verbose=True
        )
    
    def _create_sentiment_vocabulary(self) -> List[str]:
        """Create a comprehensive vocabulary focusing on sentiment patterns"""
        vocab = set()
        
        # Add base sentiment terms
        for category in self.sentiment_vocab.values():
            vocab.update(category)
        
        # Generate common financial patterns
        patterns = [
            f"{magnitude} {sentiment}" 
            for magnitude in self.sentiment_vocab['magnitude']
            for sentiment in (self.sentiment_vocab['bullish'] + self.sentiment_vocab['bearish'])
        ]
        
        # Generate metric-based patterns
        metric_patterns = [
            f"{metric} {sentiment}"
            for metric in self.sentiment_vocab['metrics']
            for sentiment in (self.sentiment_vocab['bullish'] + self.sentiment_vocab['bearish'])
        ]
        
        vocab.update(patterns)
        vocab.update(metric_patterns)
        
        return list(vocab)
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text to emphasize sentiment patterns"""
        if not isinstance(text, str):
            return ""
            
        try:
            tokens = word_tokenize(text.lower())
            
            # Generate n-grams that match our sentiment patterns
            bigrams = [' '.join(ng) for ng in ngrams(tokens, 2)]
            trigrams = [' '.join(ng) for ng in ngrams(tokens, 3)]
            
            # Prioritize sentiment-related phrases
            processed_text = text
            for phrase in bigrams + trigrams:
                if any(term in phrase for term in self._create_sentiment_vocabulary()):
                    processed_text = processed_text.replace(phrase, f"SENTIMENT_{phrase.replace(' ', '_')}")
            
            return processed_text
        except Exception as e:
            self.logger.warning(f"Error preprocessing text: {e}")
            return text

    def analyze_topics(self, headlines_file_path: str, output_dir: str) -> None:
        """Train model and perform detailed topic analysis"""
        self.logger.info("Starting sentiment-focused analysis process...")
        
        try:
            # Read and preprocess headlines
            with open(headlines_file_path, 'r', encoding='utf-8') as f:
                headlines_sets = [line.strip().split(';') for line in f]
                headlines = [h for headline_set in headlines_sets for h in headline_set]
            
            # Preprocess headlines
            processed_headlines = [self.preprocess_text(h) for h in headlines if h]
            
            self.logger.info(f"Training on {len(processed_headlines)} preprocessed headlines...")
            
            # Train the model
            topics, probs = self.topic_model.fit_transform(processed_headlines)
            
            # Save model first
            self.save_model(os.path.join(output_dir, "EX1_BERTOPIC_MODEL_100.pt"))
            
            # Prepare report content
            report_lines = []
            report_lines.append(f"Topic Analysis Report - Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            report_lines.append(f"Number of headlines processed: {len(processed_headlines)}\n")
            
            # Get topic info
            topic_info = self.topic_model.get_topic_info()
            
            # 1. Overall Statistics
            report_lines.append("=== OVERALL STATISTICS ===")
            report_lines.append(f"Total Documents: {len(headlines)}")
            report_lines.append(f"Total Topics: {len(topic_info)}")
            report_lines.append(f"Average Documents per Topic: {topic_info['Count'].mean():.2f}")
            report_lines.append(f"Most Common Topic Size: {topic_info['Count'].mode()[0]}")
            report_lines.append(f"Median Topic Size: {topic_info['Count'].median():.2f}")
            report_lines.append(f"Topic Size Range: {topic_info['Count'].min()} - {topic_info['Count'].max()}\n")
            
            # 2. Top Topics Analysis
            report_lines.append("=== TOP 40 LARGEST TOPICS ===")
            for idx, row in topic_info.head(40).iterrows():
                if idx != -1:  # Skip outlier topic
                    # Get more terms per topic
                    terms = self.topic_model.get_topic(idx)[:10]
                    term_str = ", ".join([f"{term[0]} ({term[1]:.3f})" for term in terms])
                    report_lines.append(f"\nTopic {idx} (Size: {row['Count']} documents)")
                    report_lines.append(f"Top Terms: {term_str}")
                    
                    # Get more example documents
                    docs = self.topic_model.get_representative_docs(idx)[:5]
                    report_lines.append("Example Headlines:")
                    for doc in docs:
                        report_lines.append(f"- {doc}")
                    
                    # Add topic coherence/quality metrics if available
                    try:
                        topic_coherence = self.topic_model.get_topic_coherence(idx)
                        report_lines.append(f"Topic Coherence: {topic_coherence:.3f}")
                    except:
                        pass
                    
                    # Add document diversity
                    unique_words = set()
                    for doc in docs:
                        unique_words.update(doc.lower().split())
                    report_lines.append(f"Unique Words in Examples: {len(unique_words)}")
                    report_lines.append("---")
            
            # 3. Sentiment Analysis per Topic
            report_lines.append("\n=== SENTIMENT DISTRIBUTION PER TOPIC ===")
            for idx in topic_info.head(40)['Topic']:
                if idx != -1:
                    docs = self.topic_model.get_representative_docs(idx)
                    bullish_count = sum(1 for doc in docs if any(term in doc.lower() for term in self.sentiment_vocab['bullish']))
                    bearish_count = sum(1 for doc in docs if any(term in doc.lower() for term in self.sentiment_vocab['bearish']))
                    
                    total = len(docs)
                    if total > 0:
                        report_lines.append(f"\nTopic {idx}:")
                        report_lines.append(f"Bullish: {bullish_count/total*100:.1f}%")
                        report_lines.append(f"Bearish: {bearish_count/total*100:.1f}%")
                        report_lines.append(f"Neutral: {(total-bullish_count-bearish_count)/total*100:.1f}%")
                        # Add sentiment strength
                        report_lines.append(f"Total Sentiment Documents: {bullish_count + bearish_count}")
                        if bullish_count + bearish_count > 0:
                            report_lines.append(f"Sentiment Strength: {abs(bullish_count-bearish_count)/(bullish_count+bearish_count):.2f}")
            
            # 4. Topic Distance Analysis
            report_lines.append("\n=== TOPIC DISTANCES AND RELATIONSHIPS ===")
            topic_embeddings = self.topic_model.topic_embeddings_
            if topic_embeddings is not None:
                similarities = cosine_similarity(topic_embeddings)
                top_topics = topic_info.head(40)['Topic'].tolist()
                
                # Find most similar topic pairs
                report_lines.append("\nMost Similar Topic Pairs:")
                similarity_pairs = []
                for i in top_topics:
                    for j in top_topics:
                        if i < j:
                            sim_score = similarities[i][j]
                            similarity_pairs.append((i, j, sim_score))
                
                # Sort by similarity and show top 10 most similar pairs
                similarity_pairs.sort(key=lambda x: x[2], reverse=True)
                for i, j, sim_score in similarity_pairs[:10]:
                    report_lines.append(f"Topics {i} and {j}: {sim_score:.3f}")
                
                # Find most distinct topics
                report_lines.append("\nMost Distinct Topics (Average Distance to Other Topics):")
                for i in top_topics:
                    avg_distance = np.mean([1 - similarities[i][j] for j in top_topics if i != j])
                    report_lines.append(f"Topic {i}: {avg_distance:.3f}")
            
            # Write report to file
            report_path = os.path.join(output_dir, "EX1_100_BERTOPIC_REPORT.txt")
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(report_lines))
            
            self.logger.info(f"Analysis report saved to {report_path}")
            
        except Exception as e:
            self.logger.error(f"Error in topic analysis: {e}")
            raise
        
    def save_model(self, model_path: str) -> None:
        """Save the trained model"""
        try:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            self.topic_model.save(model_path)
            self.logger.info(f"Model saved to {model_path}")
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
            raise

def main():
    # Define file paths
    headlines_file = '/Users/daniellavin/Desktop/proj/MoneyTrainer/x_preprocess/energy_headlines.txt'
    output_dir = '/Users/daniellavin/Desktop/proj/MoneyTrainer/x_preprocess/EX'
    
    # Initialize and train model
    model = FinancialTopicModel(
        n_gram_range=(1, 3),
        min_topic_size=10,
        nr_topics=100
    )
    
    # Analyze topics and generate report
    model.analyze_topics(headlines_file, output_dir)

if __name__ == "__main__":
    main()