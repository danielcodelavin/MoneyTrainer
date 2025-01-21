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
from hdbscan import HDBSCAN

# Set tokenizer parallelism before anything else
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Download required NLTK data
nltk.download('punkt')

class FinancialTopicModel:
    def __init__(
        self,
        n_gram_range: Tuple[int, int] = (1, 3),
        min_topic_size: int = 30,    
        nr_topics: int = 100        
    ):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Optimized sentiment vocabulary
        self.sentiment_vocab = {
            'bullish': [
                # Market-specific positive terms
                "rally", "surge", "soar", "breakout", "upgrade",
                "outperform", "beat", "exceed", "strong", "positive",
                "growth", "momentum", "confidence", "robust",
                
                # Key positive terms
                "excellent", "outstanding", "impressive", "remarkable",
                "exceptional", "stellar", "favorable", "promising",
                "successful", "thriving", "profitable", "valuable",
                
                # Achievement terms
                "milestone", "breakthrough", "achievement",
                "success", "advancement", "progress", "improvement",
                
                # Stability and growth
                "stable", "secure", "reliable", "sustainable",
                "expand", "accelerate", "optimize"
            ],
            
            'bearish': [
                # Market-specific negative terms
                "plunge", "crash", "tumble", "drop", "downgrade",
                "underperform", "miss", "weak", "negative", "decline",
                "risk", "bearish", "downturn",
                
                # Key negative terms
                "disappointing", "concerning", "troubling", "alarming",
                "challenging", "unfavorable", "adverse", "harmful",
                "poor", "insufficient", "restricted", "troubled",
                
                # Risk and problems
                "risky", "dangerous", "volatile", "unstable",
                "uncertain", "crisis", "emergency", "disaster",
                
                # Failure terms
                "fail", "failure", "collapse", "breakdown",
                "bankruptcy", "loss", "deficit", "shortage"
            ],
            
            'metrics': [
                # Core financial metrics
                "earnings", "revenue", "profit", "margin", "sales",
                "guidance", "forecast", "outlook", "target",
                "performance", "returns", "dividend", "price",
                "debt", "capital", "investment", "cash flow",
                
                # Key market metrics
                "volume", "volatility", "momentum", "trend",
                "index", "benchmark", "indicator",
                
                # Essential business metrics
                "market share", "growth rate", "efficiency",
                "productivity", "profitability"
            ],
            
            'magnitude': [
                # Strong intensity
                "significantly", "substantially", "considerably",
                "dramatically", "extremely",
                
                # Moderate
                "moderately", "relatively", "somewhat",
                "gradually",
                
                # Core comparatives
                "more", "less", "higher", "lower",
                "better", "worse", "stronger", "weaker"
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
        
        # Initialize HDBSCAN with optimal settings
        hdbscan_model = HDBSCAN(
            min_cluster_size=min_topic_size,
            metric='euclidean',
            cluster_selection_method='eom',
            prediction_data=True,
            min_samples=1,
            core_dist_n_jobs=1  # Prevent parallelism issues
        )
        
        # Initialize BERTopic with the HDBSCAN model
        self.topic_model = BERTopic(
            embedding_model=self.sentence_model,
            vectorizer_model=self.vectorizer,
            n_gram_range=n_gram_range,
            min_topic_size=min_topic_size,
            nr_topics=nr_topics,
            hdbscan_model=hdbscan_model,
            calculate_probabilities=True,
            verbose=True
        )
    
    def _create_sentiment_vocabulary(self) -> List[str]:
        """Create a comprehensive vocabulary focusing on sentiment patterns"""
        vocab = set()
        
        # Add base sentiment terms
        for category in self.sentiment_vocab.values():
            vocab.update(category)
        
        # Generate common financial patterns with limited combinations
        patterns = [
            f"{magnitude} {sentiment}" 
            for magnitude in self.sentiment_vocab['magnitude'][:5]  # Limit to top 5 magnitudes
            for sentiment in (self.sentiment_vocab['bullish'][:10] + self.sentiment_vocab['bearish'][:10])  # Limit to top terms
        ]
        
        # Generate metric-based patterns with key metrics only
        metric_patterns = [
            f"{metric} {sentiment}"
            for metric in self.sentiment_vocab['metrics'][:10]  # Limit to top 10 metrics
            for sentiment in (self.sentiment_vocab['bullish'][:5] + self.sentiment_vocab['bearish'][:5])  # Very selective sentiment terms
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
            
            # Move model to CPU for clustering if using MPS
            if hasattr(self.sentence_model, 'to'):
                self.sentence_model = self.sentence_model.to('cpu')
            
            # Train the model
            topics, probs = self.topic_model.fit_transform(processed_headlines)
            
            # Get topic info after training
            topic_info = self.topic_model.get_topic_info()
            
            # Save model first
            self.save_model(os.path.join(output_dir, "Y100_BERTOPIC.pt"))
            
            # Generate the analysis report
            self._generate_report(headlines, processed_headlines, topics, probs, topic_info, output_dir)
            
        except Exception as e:
            self.logger.error(f"Error in topic analysis: {e}")
            raise
    
    def _generate_report(self, headlines, processed_headlines, topics, probs, topic_info, output_dir):
        """Generate detailed analysis report"""
        report_lines = []
        report_lines.append(f"Topic Analysis Report - Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        report_lines.append(f"Number of headlines processed: {len(processed_headlines)}\n")
        
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
                terms = self.topic_model.get_topic(idx)[:10]
                term_str = ", ".join([f"{term[0]} ({term[1]:.3f})" for term in terms])
                report_lines.append(f"\nTopic {idx} (Size: {row['Count']} documents)")
                report_lines.append(f"Top Terms: {term_str}")
                
                docs = self.topic_model.get_representative_docs(idx)[:5]
                report_lines.append("Example Headlines:")
                for doc in docs:
                    report_lines.append(f"- {doc}")
                
                try:
                    topic_coherence = self.topic_model.get_topic_coherence(idx)
                    report_lines.append(f"Topic Coherence: {topic_coherence:.3f}")
                except:
                    pass
                
                unique_words = set()
                for doc in docs:
                    unique_words.update(doc.lower().split())
                report_lines.append(f"Unique Words in Examples: {len(unique_words)}")
                report_lines.append("---")
        
        # 3. Sentiment Analysis
        self._add_sentiment_analysis(topic_info, report_lines)
        
        # 4. Topic Distance Analysis
        self._add_distance_analysis(topic_info, report_lines)
        
        # Write report to file
        report_path = os.path.join(output_dir, "Y_100_Bertopic.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        self.logger.info(f"Analysis report saved to {report_path}")
    
    def _add_sentiment_analysis(self, topic_info, report_lines):
        """Add sentiment analysis section to report"""
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
                    report_lines.append(f"Total Sentiment Documents: {bullish_count + bearish_count}")
                    if bullish_count + bearish_count > 0:
                        report_lines.append(f"Sentiment Strength: {abs(bullish_count-bearish_count)/(bullish_count+bearish_count):.2f}")
    
    def _add_distance_analysis(self, topic_info, report_lines):
        """Add topic distance analysis section to report"""
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
    headlines_file = '/Users/daniellavin/Desktop/proj/MoneyTrainer/Y/RED_Bert_headlines.txt'
    output_dir = '/Users/daniellavin/Desktop/proj/MoneyTrainer/x_preprocess/Y'
    
    # Initialize and train model
    model = FinancialTopicModel(
        n_gram_range=(1, 3),
        min_topic_size=30,
        nr_topics=100
    )
    
    # Analyze topics and generate report
    model.analyze_topics(headlines_file, output_dir)

if __name__ == "__main__":
    main()