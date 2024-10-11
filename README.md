## Problem Statement 

The aim of this project is to develop a classification model that is able to accurately distinguish between two types of users based on their Reddit posts: users who are experienced investors (professionals) and users who are interested in finance but are not heavily investing yet (amateurs). 

This model will analyze posts from r/investing, where more advanced discussions around investing occurs and r/personalfinance which entails a wider array of topics, often focused on personal financial management and introductory-level investment advice. 

**How does this apply to the real world?**

For fintech companies, being able to effectively target novice investors can lead to more efficient marketing strategies. Companies often face the challenge of identifying and targeting novice investors, who represent an untapped customer base, without wasting resources on experienced investors who may already have established investment patterns. By identifying and focusing on users who are new to investing but still interested in expanding their finances, companies can tailor educational content, products, and services specifically to meet the needs of this group. For example, they could offer simplified investment platforms, beginner investment plans, or educational materials to convert finance enthusiasts into active investors.

## Model Training and Evaluation

### Logistic Regression Model, CountVectorizer

> This is my baseline model. Logistic Regression is a simple, interpretable model that acts as a benchmark before diving into more complex models. In this case, the Logistic Regression model performed fairly well with CountVectorizer. This model acheived high accuracy, precision and recall making it a great starting point to compare the rest of our models to.

#### Findings 
* **Precision**: 0.86
> * This means that 86% of the times the model predicted r/personalfinance, it was correct. Because precision is high, we can assume that we have low false positives. This means that the model does not incorrectly label too many r/investing posts as r/personalfinance.
* **Recall**: 0.89
> * This means that the model identified 89% of the actual r/personalfinance posts correctly. False negatives are low because recall is high. The model is correctly identifying most of the r/personalfinance posts.
* **F-1 Score**: 0.87
> * This is the balance between precision and recall.
* **Overall Accuracy**: 0.84

### Logistic Regression Model, TfidfVectorizer

> TF-IDF Vectorizer puts more emphasis on less frequent, unique terms in documents. This allows the model to better capture the distingushing features of each subreddit.

#### Findings 
* **Precision**: 0.86
> * This means that 86% of the times the model predicted r/personalfinance, it was correct. Because precision is high, we can assume that we have low false positives. This means that the model does not incorrectly label too many r/investing posts as r/personalfinance.
* **Recall**: 0.91 (Higher than CountVectorizer)
> * This means that the model identified 91% of the actual r/personalfinance posts correctly. False negatives are low because recall is high. The model is correctly identifying most of the r/personalfinance posts. One thing to note, the recall for r/investing decreased to 77%. 
* **F-1 Score**: 0.88
> * This is the balance between precision and recall.
* **Overall Accuracy**: 0.85
> * Compared to the baseline model, TfidfVectorizer slightly improves the accuracy from 0.84 to 0.85. This suggests that TF-IDF is capturing more relevant features than the word counts from CountVectorizer. 

### Logistic Regression Model, CountVectorizer with Custom Vocabulary Parameter

> While preprocessing my text data, I noticed that there were several words that were unique to each subreddit. I thought that maybe my model would improve if the vocabulary paramter was hand-picked based on this. I went ahead and fit my Logistic Regression model using a CountVectorizer and my own vocabulary parameter. The custom vocabulary ultimately **did not** improve the model's performance over the baseline model. From this, we can assume that the manually defined vocabulary may not capture meaningful features compared to vocabulary automatically chosen by the vectorizer. There can be several reason for these drawbacks. By limiting the vocabulary, the model may be missing out on other relevant words that may provide nuance. Another drawback to this model may be that it is easier to inject personal bias. This results in essential terms being ignored. Custom vocabulary also poses flexibilty issues. The model may not generalize well across different or updated datasets leading to poor future performance. By hand selecting the vocabulary, the model risks becoming an oversimplified model that may not be able to capture patterns in the data or becoming too specific and overfit. Finally, creating this model manually helped me understand how tedious the process of parsing through data to find key words to add to the vocabulary would be. Despite these drawbacks, the model surprisingly did not perform worse. However, the model's performance did not improve either. This suggests that the manually defined vocabulary parameter may not be able to capture enough nuance to improve the model's performance.

#### Findings 
* **Precision**: 0.86
> * This means that 86% of the times the model predicted r/personalfinance, it was correct. Because precision is high, we can assume that we have low false positives. This means that the model does not incorrectly label too many r/investing posts as r/personalfinance.
* **Recall**: 0.89
> * This means that the model identified 89% of the actual r/personalfinance posts correctly. False negatives are low because recall is high. The model is correctly identifying most of the r/personalfinance posts.
* **F-1 Score**: 0.87
> * This is the balanced score between precision and recall.
* **Overall Accuracy**: 0.84

### Logistic Regression Model, TfidfVectorizer with Custom Vocabulary Parameter

> As mentioned above, using custom vocabulary has multiple drawbacks. When they are not well-researched or there is not enough data to capture relevant terms effectively, they can harm a model's performance. 

#### Findings 
* **Precision**: 0.70
> * There is a significant drop in accuracy from 85% in the baseline model to 70%. 
* **Recall**: 0.95
> * While the recall for r/personalfinance was 95%, the recall for r/investing was a mere 40%. This suggests that many r/investing posts are misclassified as r/personalfinance. This suggests a high number of False Negatives for r/investing which significantly decreased performance. 
* **F-1 Score**: 0.81
> * The imbalance between r/investing scores and r/personalfinance scores suggests that the model is overfitting to r/personalfinance and failing to identify r/investing correctly. 
* **Overall Accuracy**: 0.73

### Naive Bayes Model, CountVectorizer

> Naive Bayes generally works well with text data. This is seen especially when words are independent of one another because the model automatically assumes that features are not dependent. CountVectorizer captures word frequencies well for this model which contributes to the improved results. 

#### Findings 
* **Precision**: 0.87
> * Naive Bayes seems to be performing on par with Logistic Regression + TfidfVectorizer. The precision for r/personalfinance is slighty higher. 
* **Recall**: 0.88
> * This means that the model identified 88% of the actual r/personalfinance posts correctly. False negatives are low because recall is high. Recall for r/investing is also higher with 81% which suggests that this model is performing better than the baseline in accurately identifying r/investing posts. Fewer false negatives are observed for r/investing as the model has improved from the baseline. 
* **F-1 Score**: 0.88
> * This is the balance between precision and recall.
* **Overall Accuracy**: 0.85

### Naive Bayes Model, TfidfVectorizer

> TF-IDF may not be able to capture word relevance as effectively for Naive Bayes in this dataset as it did for Logistic Regression. This may be due to the fact that Naive Bayes benefits more from raw frequency counts seen in CountVectorizer.

#### Findings 
* **Precision**: 0.83
> * This means that 83% of the times the model predicted r/personalfinance, it was correct. Because precision is high, we can assume that we have low false positives. This means that the model does not incorrectly label too many r/investing posts as r/personalfinance.
* **Recall**: 0.93
> * While recall for r/personalfinance is 93% we see that recall for r/investing dropped to 72%. This means that the model faile to classify many r/investing posts. This results in a higher number of false negatives for r/investing. 
* **F-1 Score**: 0.88
> * This is the balance between precision and recall.
* **Overall Accuracy**: 0.84

## Conclusion 

**Key Metrics**
> * Precision: This measured how many of the posts predicted to be in a subreddit actually belonged to that subreddit. This was an important metric to note because false positives, in this case when r/investing posts are classified as r/personalfinance, are costly. This would mean the company would be marketing to professional investors when their target audience is novices.
> * Recall: This measured how many of the actual posts from a subreddit were correctly identified. Having false negatives and missing r/personalfinance posts are detrimental because the company could be missing out on potential users.
> * F1- Score: This gives an overall picture of the model's performance.

**Logistic Regression with TF-IDF**
> * Best Score: 0.848
> * Precision: 0.86
> * Recall: 0.91
> * F1-Score: 0.88
> * This model has very high recall, meaning it is able to correctly identify 91% of the posts from r/personalfinance. It also has a strong precison of 86% meaning that a high percentage of posts classified as r/personalfinance are correct.

**Naive Bayes with CountVectorizer**
> * Best Score: 0.851
> * Precision: 0.87
> * Recall: 0.88
> * F1-Score: 0.88
> * This model has a slightly lower recall for r/personalfinance (from 0.91 to 0.88). The other metrics are fairly on par with the Logistic Regression model. This means that the model is still very good but it misses a few more r/personalfinance posts compared to the logistic model.

**FP vs FN**
> * False Positives (FP): This occurs when the model is predicting a post is from r/personalfinance when it's actually from r/investing. Too many false positives could result in a company targeting users who may already be investors and would not find any benefit from the company's product.
> * False Negatives (FN): This occurs when the model is predicting a post is from r/investing when it is actually from r/personalfinance. False negatives could result in a company missing potential users who are novices and may be interested in the company's product. This may be worse for the company in the long run.

**Best Model**

#### Logistic Regression with TF-IDF!! 
This model has the highest recall for identifying posts from r/personalfinance. This means that a company that uses this model to scout for novices who are interested in investing will reach their target audience. It is the best choice because it prioritizes capturing as many r/personalfinance posts as possible (these are potential users). 
