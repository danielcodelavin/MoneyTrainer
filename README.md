Short-Term Stock Prediction with Price + News Tensors

A practical README

⸻

1 Concept in a Nutshell

The project asks a single question:

Given (i) the last few days of intraday price action and (ii) everything the news said yesterday, can we guess what the stock will do tomorrow?

To answer it, every training sample is turned into one 181-dimensional tensor

Component	Length	Purpose
Movement Score	1	Quality of the next-day price move (–1 … +1)
Price Tensor	80	80 hourly closes leading up to the prediction time, normalised
News Topics	100	Probability distribution over 100 latent news topics


⸻

2 How the Tensor Is Built

2.1 Price window
	•	Source – yfinance hourly candles (pre- and post-market included).
	•	Span – Up to 10 calendar days are scanned until ≥ 80 closes are collected.
	•	Clean-up –
	•	Checks for NaN/Inf/zero or negative prices.
	•	z-score normalisation:  (x-\mu)/\sigma per symbol.

2.2 Movement score (ground-truth label)

For the next trading day the script computes:

overall trend   (open ➜ close)        … 50 %
max positive     intraday high        … 25 %
max negative     intraday low         … 25 %

Each component is capped at ±10 %, rescaled to –1 … +1 and blended into a single number.

2.3 News headlines
	•	Where they come from – A lightweight scraper calls a financial-news API for the given ticker and date.
	•	Cleaning – duplicates dropped, leading/trailing whitespace stripped.

2.4 Topic probabilities with BERTopic
	1.	Load a pre-trained BERTopic model (100 topics, frozen).
	2.	Transform the headlines → matrix p ∈ ℝⁿˣ¹⁰⁰.
	3.	Mean-pool across all headlines for that day.
	4.	Normalise so probabilities sum to 1 (guarding against rounding drift).

2.5 Putting it together

integrated = torch.cat([movement_score,
                        price_tensor,      # 80 × 1
                        topic_probs])      # 100 × 1
torch.save(integrated, "<symbol>_<yyyymmdd>_<hhmmss>.pt")

No CSV, JSON, Parquet, or database layers—just raw .pt files ready for the model.

⸻

3 Example Training Metrics (illustrative)

Epoch	Train Loss	Val Loss	Precision	Recall	Accuracy	L1 Dist
99	0.675	0.431	0.696	0.615	0.913	0.031

	•	Precision – of all “up-moves” predicted, ≈ 70 % were correct.
	•	Recall – model caught ≈ 62 % of actual up-moves.
	•	Accuracy – dominated by the “flat” class, so always read with Precision/Recall.
	•	L1 distance – average absolute error on the movement score.

⸻

4 Quick Tour of the Codebase

Area	What it does
Validation helpers	Detect bad or insufficient raw price data.
Price feature builder	Collects 80 recent closes, normalises tensor.
News fetcher	Pulls yesterday’s headlines for the symbol & date.
BERTopic wrapper	Converts headline list → 100-d probability vector.
Tensor assembler	Concatenates (score, price, topics) and saves .pt.
Driver script	Random-date generation, looping over shuffled symbol list, progress bar, retry logic, error logging.


⸻

5 Model (one working example)

A compact feed-forward net:

Input 181 ▸ Dense 256 ▸ GELU ▸ Dropout
        ▸ Dense 128 ▸ GELU ▸ BatchNorm
        ▸ Dense  64 ▸ GELU
        ▸ Head   1  (regression)   or
                    3 (bull / bear / flat)

	•	Optimiser AdamW   •  Scheduler Cosine decay  •  Early-stop on Val L1

⸻

6 Extending the Project
	•	More intraday context – increase min_points or add volume tensor.
	•	Sector-specific heads – fine-tune final layers per GICS sector.
	•	Alternative sentiment – feed Reddit/Twitter embeddings alongside the BERTopic vector.

Pull requests are welcome—please include unit tests.

⸻

License

MIT – do what you want, just keep the notices.
