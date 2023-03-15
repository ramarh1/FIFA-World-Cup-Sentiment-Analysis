# FIFA-World-Cup-Sentiment-Analysis
Giang Nguyen
Ramar Huntley
Xinmeng Wang
Adeniyi Samson


Agenda

Descriptive Analysis
  - In this step, we’re going to introduce our dataset and break down the data by extracted information.
Data Preprocessing
  - In this step, we’re going to discuss briefly which preprocessing steps are taken to prepare for the machine learning models.
Model Performance & Selection
  - In this step, we’re going to discuss all the conducted models and their performances. Lastly, we’re discussing our choice of model and the insights from the analysis.


Data Overview
 - The dataset was created using the Snscrape and the cardiffnlp/twitter-roberta-base-sentiment-latest model in Hugging Face Hub.The dataset includes tweets in English containing the hashtag #WorldCup2022. - Source: Kaggle

Data Visualization
 - some highlights from the 30,000 Tweets collected

* Mosts of the Tweets about FIFA have positive sentiments
Positive - 8489
Neurtal - 8251
Negative - 5784

* More Tweets came from mobile devices than from website
From Android -  31.78%
From Iphone - 44.31%
From Ipad - 1.12%
from Webapp - 20.99%

Quick views at hot  topics in the tweets
 -  BTS - the opening ceremony performancer & Elon Musk are among the top most mentioned usernames in the tweets.
 -  #WordCup2022 is way more popular than #FIFAWorldCup2022

Word cloud of all the hashtags
Challenge: We’re not able to increase the resolution of the word cloud pictures

Data Preprocessing 
- A breakdown of what were done to the dataset to prepare for machine learning models

Preprocessing Steps 
1 - Extracted urls, usernames, and hashtags
2 - Remove all extracted information and stop words to reduce word repetition and reduce document sizes
3 - Create TFIDF
4 - Conduct our own sentiment labels, using Vader


                    Logistic  Regression    Naive Bayes     Random Forest   Transformer Base

Model Accuracy            79%                 71%              68%               81%
---------------------------------------------------------------------------------------------------
Precision 
on negative, 
neutral,
positive             77%, 79%, 80%        63%, 72%, 72%    71%, 64%, 71%     83%, 72%, 90%
---------------------------------------------------------------------------------------------------
Recall 
on negative, 
neutral, 
positive             59%, 84%, 84%        78%, 54%, 79%    64%, 66%, 73%     85%, 83%, 75%
---------------------------------------------------------------------------------------------------


Takeaways
1. The models have the most disagreements in labeling neutral sentiments
2. Transformer Base model seems to have the highest accuracy scores
3. Random forest seems to have the hardest time to predict accurate sentiments
4. Logistic regression using the vader module sentiment labels has a better performance
5. There are a lot of spams and bot generated content that got mixed into the tweets - diluting the content























