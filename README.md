Short-Term Stock Prediction with Historical Price & News Sentiment

“Markets are moved not only by numbers but by narratives.”  – Robert J. Shiller

⸻

1  Project Overview

This repository investigates whether very-recent price behaviour coupled with very-recent news sentiment can anticipate the next trading-day’s price movement.
The core idea is to represent the market state at t₀ as a single concatenated tensor:

[ ΔP-₉  ΔP-₈  …  ΔP-₀ ]  ⨁  [ Pr(topic₁) … Pr(topicᴷ) ]

where
ΔP-ᵢ = normalised price change i days ago, and
Pr(topicⱼ) = probability that the day’s news belongs to latent topic j uncovered by BERTopic.

A deep neural net ingests this tensor and outputs the expected direction (and optionally magnitude) of ΔP₊₁.

⸻

2  System Architecture

flowchart TD
    A[Historical Prices<br>(10-day window)] -->|normalise| C[Price Tensor]
    B[Recent News Articles<br>(T-3 days → T)] -->|BERTopic| D[Topic Probabilities]
    C --> E[Concatenate]
    D --> E
    E --> F[Neural Model]
    F --> G[Prediction
(Direction / Return)]

2.1  Data Acquisition
	•	Price stream – Queried every evening via a financial data API (e.g., FinHub, Alpha Vantage). Raw OHLCV vectors are converted to daily log-returns and z-score normalised per asset.
	•	News stream – Aggregated from public RSS feeds and paid APIs (Bloomberg, Reuters, Yahoo Finance) plus selective web-scraping of company press releases & SEC 8-K filings.  Scraping is asynchronous and rate-limited to respect TOS policies.

2.2  Text Processing & BERTopic

BERTopic blends modern language models with classic topic modelling:
	1.	Embedding – Sentences are converted to high-dimensional vectors using a Sentence-Transformer (e.g. all-mpnet-base-v2 fine-tuned on financial text).
	2.	Dimensionality Reduction – UMAP projects embeddings into a low-dimensional manifold that preserves local structure.
	3.	Clustering – HDBSCAN discovers dense regions (= topics) without pre-specifying k and assigns each article a soft membership score.
	4.	Class-based TF-IDF – c-TF-IDF extracts interpretable keywords per topic.

For every trading day the pipeline aggregates article-level probabilities into a topic probability vector representing the dominant sentiments that day.  Typical K ≈ 25–40 topics.

2.3  Feature Engineering

Component	Shape	Description
Price tensor	(10,)	Normalised returns t-9 … t-0
Sentiment tensor	(K,)	Topic probability distribution
Concatenated feature	(10 + K,)	Final input to model

2.4  Predictive Model

A lightweight feed-forward network (3–4 dense layers, GELU activations, dropout & batch-norm) maps the feature vector to either:
	•	Classification – bullish / bearish / flat (softmax), or
	•	Regression – expected ΔP₊₁ (linear head).

Loss is weighted to address class imbalance (bull moves are rarer than flat moves). Training employs AdamW, cosine LR decay, and early-stopping on validation L1 distance.

⸻

3  Reading the Training Logs

Training pipelines emit CSV logs like:

Epoch,Train_Loss,Val_Loss,Precision,Recall,Accuracy,L1_Distance
99,0.674884,0.430498,0.695652,0.615385,0.913265,0.031350

	•	Train_Loss vs Val_Loss – Divergence may signal over-fitting.
	•	Precision ≈ 0.70 – Out of all bullish predictions ≈ 70 % were correct.
	•	Recall ≈ 0.62 – The model captured ~62 % of all real bullish days.
	•	Accuracy > 0.90 – High because the “flat” class dominates; always inspect precision & recall.
	•	L1_Distance ≈ 0.03 – Mean absolute prediction error ≈ 3 % (in the chosen return units).

Tip: Focus on the precision-recall curve when the decision threshold is tunable – it gives more insight than accuracy in imbalanced settings.

⸻

4  Performance Break-downs (Illustrative)

Averaged across six experimental runs:
	•	By Market-Cap – Large-caps outperform small-caps by ~3 pp in precision.
	•	By Trading Volume – Low-volume equities show the lowest L1 error, possibly because news moves them disproportionately.
	•	By Sector – Basic Materials led (precision ≈ 77 %), Finance lagged due to complex derivative-driven moves.

These findings highlight that one universal model does not fit all buckets – further work could involve training sector-specific heads.

⸻

5  Deep-Dive: What Exactly is BERTopic?

BERTopic = Bidirectional Encoder Representations + Topic clustering

Step	Tool / Alg.	Purpose
1  Embeddings	Transformer encoder (BERT, RoBERTa)	Captures semantic context beyond bag-of-words
2  UMAP	Non-linear dimensionality reduction	Retains neighbours while compressing vectors
3  HDBSCAN	Density-based clustering	Finds topic groups without specifying k
4  c-TF-IDF	Re-weighed term importance	Generates human-readable topic keywords

Advantages over classic LDA:
	•	Works with sentence embeddings → better handles synonyms / context.
	•	Soft memberships → each document gets a probability per topic, perfect for tensor input.
	•	Dynamic topic count → adapts when the news cycle intensifies.

⸻

6  Data Pipeline in Practice
	1.	Scheduler triggers scraping at 22:00 UTC after market close.
	2.	ETL Jobs write raw JSON to object storage; metadata tracked in a SQLite ledger.
	3.	Feature Builder merges price & sentiment tensors and persists them as parquet.
	4.	Trainer consumes latest parquet files, logs metrics to TensorBoard & CSV.
	5.	Inference Service (optional) publishes next-day predictions via REST.

⸻

7  Limitations & Ethical Notes
	•	Look-Ahead Bias – All pipelines enforce strict date-based joins; verify before live trading.
	•	News Coverage Bias – Small-cap firms may lack news, skewing predictions to “flat.”
	•	Regulatory Risk – Web-scraping terms evolve; ensure compliance with each source.

⸻

8  Future Work
	•	Incorporate intraday sentiment (Twitter, StockTwits) for higher granularity.
	•	Replace static transformer with continual-learning FinBERT fine-tuned monthly.
	•	Test Graph Neural Networks treating tickers as nodes linked by supply-chain edges.

⸻

9  Contributing

We welcome PRs on new data connectors, model architectures, and evaluation dashboards.
Please include unit tests and abide by the MIT License when submitting code.

⸻

10  Citation

If you use this work, please cite:

@misc{shortterm-stock-news-tensor,
  title  = {Short-Term Stock Prediction with Price & News Tensors},
  year   = {2025},
  author = {Contributors of this repository},
  url    = {https://github.com/<org>/<repo>}
}


⸻

Licensed under the MIT License.