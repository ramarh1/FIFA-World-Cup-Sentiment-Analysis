# FIFA-World-Cup-Sentiment-Analysis

<img width="544" alt="Screenshot 2023-03-15 at 6 22 09 PM" src="https://user-images.githubusercontent.com/88624677/225456720-ab209392-f633-469d-b00c-185d10ddae0a.png">


Contributors:
Giang Nguyen,
Ramar Huntley,
Xinmeng Wang,
Adeniyi Samson

### Data Overview and Purpose
 - The dataset was created using the Snscrape and the cardiffnlp/twitter-roberta-base-sentiment-latest model in Hugging Face Hub.The dataset includes tweets in English containing the hashtag #WorldCup2022. - Source: Kaggle
 - The purpose of this research is to study which machine learning model would perform better at predicting sentiment analysis of tweets related to the FIFA World Cup on Twitter.

### Pipeline Overview
1. Descriptive Analysis: In this step, we’re going to introduce our dataset and break down the data by extracted information.
2. Data Preprocessing: In this step, we’re going to discuss briefly which preprocessing steps are taken to prepare for the machine learning models.
3. Model Performance & Selection: In this step, we’re going to discuss all the conducted models and their performances. Lastly, we’re discussing our choice of model and the insights from the analysis.

### Data Visualization
 Some highlights from the 30,000 Tweets collected

* Mosts of the Tweets about FIFA have positive sentiments
![image](https://user-images.githubusercontent.com/88624677/225453783-df87558d-346d-4551-82b0-7f202b6bd3ee.png)


* More Tweets came from mobile devices than from website
![image](https://user-images.githubusercontent.com/88624677/225453827-76962df9-f8b9-4da2-ab2a-e8c6b6266258.png)

#### Hot Topics
 -  BTS - the opening ceremony performancer & Elon Musk are among the top most mentioned usernames in the tweets.
 -  #WordCup2022 is way more popular than #FIFAWorldCup2022
 ![image](https://user-images.githubusercontent.com/88624677/225453885-e72d3b3f-07a0-4c30-b30c-cb2a8c0169fd.png)


Word cloud of all the hashtags[^1]
![image](https://user-images.githubusercontent.com/88624677/225453929-31751e87-19cc-4c7e-90db-831c1e3e14bf.png)

Word cloud of all tweets[^1]
![image](https://user-images.githubusercontent.com/88624677/225456060-627260ef-4b17-4042-9304-da040e05f9ca.png)

### Data Preprocessing 
- A breakdown of what was done to the dataset to prepare the machine learning models

#### Preprocessing Steps 
1. Extracted urls, usernames, and hashtags
2. Remove all extracted information and stop words to reduce word repetition and reduce document sizes
3. Create TFIDF
4. Conduct our own sentiment labels, using Vader

```python
# Combine all above to one function to fix texts
def fix_Text(text):
	letters = re.sub("[^a-zA-Z]https?:\/\/\S*", " ", str(text)
	                 )  # remove all non-letters and urls
	letters_1 = re.sub("#[A-Za-z0-9_]+", "", str(letters))  # remove all hashtags
	letters_2 = re.sub("@[A-Za-z0-9_]+", "", str(letters_1)
	                   )  # remove all mentions
	letters_3 = re.sub(r'[^\x00-\x7F]+', ' ', str(letters_2))

	words = letters_3.lower().split()  # make all letters lowercase
	meaningful = [snow.stem(word)
               for word in words if word not in all_stop_words]  # convert to stemmed words
	return (" ".join(meaningful))
```


### Model Performance
 
 | Metrics        | Logistic Regression| Naive Bayes| Random Forest | Transformer Base
| ------------- |:---------------------:| :-----:|:-------------------:|:----------------:
| Model Accuracy     | 79%         | 71% | 68% | 81%
| Precision (neg,neu,pos)      | 77%, 79%, 80%              |   63%, 72%, 72% |71%, 64%, 71% | 83%, 72%, 90%
| Recall (neg,neu,pos) | 59%, 84%, 84%             |    78%, 54%, 79% | 64%, 66%, 73% | 85%, 83%, 75%


### Insights/Limitations
1. The models have the most disagreements in labeling neutral sentiments
2. Transformer Base model seems to have the highest accuracy scores
3. Random forest seems to have the hardest time to predict accurate sentiments
4. Logistic regression using the vader module sentiment labels has a better performance
5. There are a lot of spams and bot generated content that got mixed into the tweets - diluting the content
6. There were unbalanced classes, which could have produced bias when performing the machine learning testing and training.



[^1]: We’re not able to increase the resolution of the word cloud pictures
