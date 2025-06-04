# MoneyTrainer: Predicting Short-Term Data Movements with News & Time Series Analysis

**MoneyTrainer** is an advanced research project dedicated to predicting short-term movements in dynamic datasets, with a primary focus on financial markets. It achieves this by synergizing high-dimensional semantic encoding of recent news articles with normalized historical time series data, feeding information into sophisticated machine learning models.

The aim is to incorporate the sentiment and thematic undercurrents of textual news data, thereby capturing a more holistic view of the factors influencing short-term fluctuations.

## Core Pipeline: From Raw Data to Integrated Tensors

The data processing pipeline is a critical component, designed to fetch, clean, transform, and integrate diverse data sources into a unified tensor format suitable for advanced model training. This pipeline is consistent across all modeling experiments.

### 1. Data Sources
The pipeline leverages:
* **Historical Time Series Data:** Stock prices (hourly closing prices) sourced using the `yfinance` library.
* **News Headlines:** Fetched for specific stock symbols and relevant dates using the `Finnhub API` (via a custom `finhub_scrape` module) and potentially supplemented by `GoogleNews`-library.

### 2. Time Series Data Processing
For each stock and a given historical `start_datetime`:
* **Fetching:** Hourly 'Close' prices are retrieved for a trailing window (e.g., 10 days, aiming for a minimum of 80 data points using `prepare_single_stock_data`).
* **Validation (`validate_raw_data`):** Ensures data integrity (non-empty, no NaN/infinity values, positive prices).
* **Normalization:** The raw price sequence is normalized using Z-score normalization: $$(X - \mu) / \sigma$$. This standardizes the data, making it more amenable for neural network processing.
* **Output:** A 1D PyTorch tensor of 80 normalized historical price points.

### 3. News Data Processing & BERTopic Encoding
Shorter than time series data, usually just past 2 days, news older than that hardly make an impact:
* **Headline Collection:** Relevant news headlines are collected for the stock.
* **BERTopic Transformation (`process_headlines_with_bertopic`):**
    * Utilizes a **pre-trained BERTopic model** (e.g., `Y100_BERTOPIC.pt`, a semantic model trained by the project author - fine tuned on finance vocab (e.g. "merger" or "expectation")).
    * BERTopic performs several steps:
        1.  Embeds headlines using sentence transformers.
        2.  Clusters these embeddings to discover latent topics.
        3.  Generates topic representations.
    * The output for a set of headlines is a **mean-pooled probability distribution across a fixed number of topics** (e.g., 100 topics). This vector is normalized to sum to 1.
* **Output:** A 1D PyTorch tensor representing the news sentiment and thematic focus for the given stock over the recent period (e.g., 100 topic probabilities). *Note: The dimensionality of this topic vector can be configured (e.g., 30 or 100 topics depending on the specific BERTopic model used for an experiment).*

### 4. Ground Truth Generation: The Movement Score
To train supervised models, a custom target variable, the **Movement Score**, is calculated for the day *following* the period of the input data:
* **Data Fetching (`returngroundtruthstock`):** Hourly 'Close' prices for the target prediction day are fetched.
* **Custom Calculation (`calculate_movement_score`):** This score, ranging from -1 to 1, quantifies the "quality" of the price movement on the target day. It's a composite metric:
    * **Overall Trend (50% weight):** (Close price - Open price) / Open price.
    * **Maximum Positive Deviation (25% weight):** (Highest price - Open price) / Open price.
    * **Maximum Negative Deviation (25% weight):** (Lowest price - Open price) / Open price.
    * All percentage changes are **capped at ±10%** before scoring to focus on common, short-term movements and reduce outlier impact.

### 5. Data Integration & Tensor Creation (`create_integrated_tensor`)
The processed data streams are combined into a single, flat PyTorch tensor:
* **Structure:**
    1.  `movement_score` (1 scalar value - the target).
    2.  Normalized historical stock data tensor (e.g., 80 values).
    3.  BERTopic news embedding tensor (e.g., 100 values).
* **Saving:** Each integrated tensor is saved as a separate `.pt` file, forming the dataset for training and evaluation. Example tensor length: $1 + 80 + 100 = 181$ elements.

### 6. Dataset Generation & Time-Disjunct Regimes
* **Automated Pipeline (`main` function in preprocessing script):**
    * Reads stock symbols and associated metadata (name, industry, sector) from a CSV file.
    * Iteratively processes stocks, selecting random dates within predefined ranges to generate a diverse dataset.
    * **Crucially, training and validation datasets are generated from strictly non-overlapping time periods** to ensure a robust evaluation of the model's ability to generalize to unseen future data, simulating real-world performance.
        * **Training Data Period Example:** News/stock data from `2024-01-15` to `2024-11-01` (ground truth from the subsequent day).
        * **Validation Data Period Example:** News/stock data from `2024-11-02` to `2025-01-16` (ground truth from the subsequent day).

## Modeling Architectures

This project explores several advanced modeling techniques, all leveraging the integrated time series and news tensors.

### 1. Multi-Layer Perceptrons (MLPs)
* Initial explorations included various MLP architectures.
* The `PrecisionFocusedMLP` (used within the Tandem Model) is a sophisticated MLP variant:
    * Separate processing pathways for stock sequence data and topic vectors.
    * Features are fused and then fed into two heads: one for predicting the `movement_score` and another for predicting the model's `confidence` in its prediction.
    * Employs `ResidualBlock`s with `LeakyReLU` and `LayerNorm` for stable training of deeper networks.

### 2. Hybrid CNN-LSTM (Model "Y")
This architecture is designed to capture complex patterns from both modalities:
* **Stock Sequence Pathway:**
    * A 1D Convolutional Neural Network (CNN) extracts local features from the normalized price sequence.
    * The output sequence from the CNN is fed into a Long Short-Term Memory (LSTM) network to model temporal dependencies.
    * An Attention mechanism is applied to the LSTM outputs, allowing the model to weigh the importance of different time steps.
* **News Topic Pathway:**
    * An MLP processes the BERTopic news vector.
* **Fusion:** Representations from both pathways are concatenated and passed through a final MLP to predict the `movement_score`.

### 3. Tandem Model: A Dynamic Weighted Ensemble (Potentially Novel)
This is one of the most innovative aspects of the project, featuring an ensemble of randomly initialized MLPs (`PrecisionFocusedMLP`) that dynamically adjust their influence.
* **Ensemble Structure:** Consists of multiple `PrecisionFocusedMLP` instances (e.g., 7 models).
* **Dynamic Voting Power:**
    * Each sub-model maintains a `PredictionHistory` (a sliding window of its recent predictions, ground truths, and confidences).
    * During inference, `calculate_model_weights` dynamically computes voting weights for each sub-model. These weights are based on a combination of each model's recent **precision** and **L1 score (inverse of L1 distance)**.
    * The final ensemble prediction is a weighted average of the sub-models' outputs.
* **Confidence Estimation:** The ensemble's confidence can be derived, for example, from the maximum confidence score among the sub-models.
* **Specialized Training:**
    * Sub-models are trained jointly.
    * A custom `PrecisionFocusedLoss` is employed:
        * Combines a primary regression loss (e.g., MSE on the movement score) with a Binary Cross-Entropy loss for the confidence head (penalizing high confidence on incorrect direction predictions).
        * Includes an explicit penalty for false positives in the movement score prediction, guiding the model towards higher precision.

## Training & Evaluation

* **Framework:** `PyTorch` is used for all model implementations and training.
* **Data Loading:** Custom `Dataset` and `DataLoader` classes efficiently load the preprocessed `.pt` tensors.
* **Optimization:** Standard optimizers like Adam are used, often with learning rate schedulers (e.g., `ReduceLROnPlateau`). Gradient clipping is also applied.
* **Metrics:**
    * Common metrics: Mean Squared Error (MSE), Mean Absolute Error (MAE).
    * Directional Accuracy: Percentage of predictions where the sign of the predicted movement score matches the sign of the true movement score.
    * For the Tandem Model, a richer set of metrics is tracked, including **Precision, Recall, L1 Distance, and Prediction Rate** (percentage of non-zero predictions).
* **Reporting:** Detailed training progress, including loss and validation metrics (and model weights for the Tandem model), are logged to report files for analysis (see example `1TX_report.txt`).
* **Time-Disjunct Validation:** As mentioned, the use of separate, chronologically ordered training and validation sets is a cornerstone of the evaluation strategy, providing a more realistic assessment of generalization capabilities.

## Key Technologies & Libraries

* **Deep Learning:** `PyTorch`
* **NLP & Topic Modeling:** `BERTopic`, `sentence-transformers`
* **Data Acquisition:** `yfinance`, `Finnhub API` (via `finhub_scrape`), `pygooglenews`
* **Data Manipulation & Scientific Computing:** `Pandas`, `NumPy`
* **Utilities:** `tqdm`

---

This project represents a rigorous and innovative approach to financial (or general data) forecasting, pushing the boundaries by integrating semantic news analysis with quantitative methods and exploring novel ensemble techniques.
