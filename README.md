# Reddit Post Classification: Identifying Novice vs. Experienced Investors

## Overview

This project focuses on developing a text classification model to differentiate between two types of Reddit users based on their posts:

- **Experienced investors (professionals)** posting in r/investing
- **Novice investors (amateurs)** posting in r/personalfinance

By leveraging natural language processing (NLP) and machine learning, the model aims to provide actionable insights for fintech companies looking to target potential new investors effectively.

## Problem Statement

Fintech companies often struggle to identify and engage with novice investors, who represent an untapped market for educational tools and beginner-friendly investment products. By classifying Reddit posts based on investing experience, companies can:

- Optimize marketing strategies by targeting the right audience
- Develop personalized financial content for different user segments
- Improve conversion rates by guiding novices toward suitable investment options

This project investigates various machine learning models to determine which approach best classifies posts between r/investing and r/personalfinance.

## Approach

### Dataset & Features

- **Data Source**: Reddit posts from r/investing and r/personalfinance
- **Preprocessing**: Tokenization, stopword removal, lemmatization
- **Feature Engineering**: CountVectorizer & TF-IDF

### Models Tested:

- Logistic Regression (baseline)
- Naïve Bayes
- Custom Vocabulary Implementations

### Evaluation Metrics

- **Precision**: Measures how many predicted r/personalfinance posts were correct
- **Recall**: Measures how many actual r/personalfinance posts were correctly identified
- **F1-Score**: Balance between precision and recall
- **Overall Accuracy**: Proportion of correctly classified posts

## Model Performance

### 1. Logistic Regression + CountVectorizer (Baseline)
- **Accuracy**: 84%
- **Precision**: 86%
- **Recall**: 89%
- **F1-Score**: 87%
- **Findings**: Strong baseline model with balanced precision and recall.

### 2. Logistic Regression + TF-IDF (Best Model)
- **Accuracy**: 85%
- **Precision**: 86%
- **Recall**: 91%
- **F1-Score**: 88%
- **Findings**: TF-IDF captured more relevant features, improving recall and F1-score.

### 3. Naïve Bayes + CountVectorizer
- **Accuracy**: 85%
- **Precision**: 87%
- **Recall**: 88%
- **F1-Score**: 88%
- **Findings**: Performs on par with Logistic Regression but benefits from word frequency-based features.

### 4. Naïve Bayes + TF-IDF
- **Accuracy**: 84%
- **Precision**: 83%
- **Recall**: 93%
- **F1-Score**: 88%
- **Findings**: Higher recall but slightly lower precision than CountVectorizer.

### 5. Logistic Regression + Custom Vocabulary (CountVectorizer & TF-IDF)
- **Accuracy Drop**: 85% → 73%
- **Recall for r/investing**: 40% (major drop)
- **Findings**: Custom vocabulary added bias, reducing generalization.

## Key Takeaways

- TF-IDF improves recall and F1-score, making it the best performing feature extraction method.
- Naïve Bayes performs well with CountVectorizer, but its performance decreases with TF-IDF.
- Custom Vocabulary is ineffective, as manually selecting words introduces bias and limits model flexibility.
- Logistic Regression + TF-IDF is the optimal model, balancing precision, recall, and generalizability.

## Business Impact

For fintech companies, leveraging this classification model can:
- Optimize customer segmentation by identifying novice investors
- Improve engagement through targeted educational content
- Enhance conversion rates by guiding users toward suitable investment tools

Future improvements could include deep learning models (e.g., transformers) to capture richer contextual information in Reddit discussions.

## Next Steps

- Expand dataset by incorporating more Reddit subreddits related to investing.
- Implement deep learning models (e.g., BERT) to improve classification.
- Deploy a live model API for real-time subreddit classification.
