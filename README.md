# An Over-The-Top Analysis of Elon Musk’s Tweeting Habits
## Adam VanWyk, Beckett Hartig

## Introduction
Our project centered around predicting the frequency of Elon Musk’s tweets for use in trading that market on the prediction market platform, Polymarket. A prediction market acts similarly to any derivatives market. That is, you can freely buy and sell contracts whose price is derived based on something in the real world. A contract on Polymarket can be worth anywhere from 0 to 1 U.S. Dollar. Specifically, you can purchase a YES or NO contract for binary events around questions such as (seen in Figure One) “Will Elon Musk tweet between 380 and 399 times from December 2nd to December 9th, 2025?” If the answer to the question turns out to be YES, the value of your contract becomes $1. Contract prices are mechanically linked, i.e. if a YES contract is worth $0.60, a NO contract in the same market would be worth $0.40. That is because the value represents the percentage chance of the event occurring. For example, a YES contract worth $0.60 represents a 60% of the event happening because you should be indifferent whether you lose or win, as defined by the equation below:
60%40100-40%60100=0


Figure One: Elon Musk Tweet Count Market

We were interested in determining if we could predict how often Elon Musk would tweet based on historical data because, if so, it could theoretically be used to trade the market shown in Figure One. We theorized that the frequency of Elon Musk’s tweets was not completely random because people often operate in phases with how often they use particular apps, such as X. We also thought that since Elon Musk has a big political and corporate presence, big news events like elections or events within his companies may spark periods of his elevated burst use of the X platform. As such, we sought to build a neural network-based model to predict the number of tweets Elon Musk would post the next day, given his recent posting behavior.

## Methodology & Data
The dataset we used is publicly available on the website xtracker.io. The available CSV file contains every tweet Elon Musk has posted on X since April 18, 2024, including a tweet ID, the text content of the post, and importantly, second-granular timestamp. While the dataset was robust, we realized that just having the timestamp and text wasn’t enough. As such, we immediately began to determine what features could be derived from the text and timestamp content of the dataset. We used a mixture of the Python CSV library and the Pandas library to create a new CSV that summarized Musk’s activity for each day with the following features and definitions:
Text Derived:
Retweets Total
Number of tweets that were retweets that day
Easily determined by the presence of “RT” at the beginning of the tweet string
Average Tweet Length
Determined by counting the characters of each tweet string
Time Derived
Total Tweets
The number of times Musk tweeted that day
Tweet Ratio
The percentage of tweets that were retweets
Average Tweets Per Hour
Total Tweets / 24
Active Hours Count
How many hours of the day (e.g., between 8:00 and 8:59) did Musk post a tweet?
Mean Intertweet Gap
How much time, on average, was Musk waiting before positing another tweet
Standard Deviation of Intertweet Gap
Minimum Intertweet Gap
Maximum Intertweet Gap
Burst Count
How many times did Elon Musk wait less than 10 minutes before posting another tweet?
Max Burst Length
What is the maximum number of tweets Musk posted in a burst as defined above?
Burstiness
(std intertweet gap - mean intertweet gap)                                                     /   (std intertweet gap + mean intertweet gap)
Tweet Entropy
Shannon Entropy 
-(P(xi)ln(P(xi))
Where P(xi) is the probability of a tweet occurring in hour i of the day
7 Day Average of Total Tweets
Calculated using a rolling 7-day window
7 Day Variance of Total Tweets

With this data in hand, calculated for every day from April 18, 2024, to the present, we were able to move on to the modeling portion of the task. Of course, our goal was to predict the total number of tweets for the next day given the past daily summaries in our data. As we theorized that Elon Musk’s behavior over a period of days was not random and could be used to predict the next day’s total, our original idea was to use an LSTM model over some number of day summaries to capture temporal patterns. This makes sense. As seen in Figure Two below, while there is a significant amount of noise in the data on the number of tweets each day, there are clear regimes of posting behavior that can be traced. We also tested a simple dense neural network, trained on the data from one day to predict the next day as a baseline, and a few other more complex networks, such as a CNN for time series.

Figure Two: Total Number of Tweets Every Day in Data

Since we were trying to predict a continuous number, we used a few specific metrics to analyze the performance of our models. For one, we analyzed training and validation loss to see the evolution of model performance throughout training. When testing the model’s predictive capacity, we used mean absolute error. The choice to use mean absolute error was notable because it has a real-world interpretation: If mean absolute error was 10, that meant that the model was off by 10 tweets on an average day. We also used a Seaborn heatmap to analyze the correlation between features, and we used permutation feature importance to learn if the model was emphasizing one feature strongly and not learning from multiple data points.

## Results
### LSTM Model
For the LSTM model, the architecture we landed on is shown below in Figure Three. The model includes two sequential LSTM layers, the first with a hidden state of 32 dimensions, and the second with a hidden state of 64 dimensions. We also applied mild L2 regularization to each of those layers to prevent overfitting. The output layer is simply a dense layer of dimension 1.

Figure Three: Keras LSTM Architecture
We used a 60/20/20 training/validation/test split to train the model. Having a large validation and test split was important for us to ensure the model wasn’t overfitting as the dataset only contained data from just over 500 days. Because of this, we also applied early stopping when compiling the model. Figure Four below shows validation and training loss in the final model. Interestingly, validation loss worsens after just 5 epochs, and the early stopping condition stops the training of the model.

Figure Four: Convergence of Training and Validation Loss

Figure Five below shows some examples of predictions the LSTM model made. Because the return_sequence parameter is left on in the model layers, we can see the model’s prediction at every step. Based on this, we can clearly see how the model behaves. Interestingly, it appears to act generally like a smoothing function. There is a fair amount of noise in the input sequence, and it is generally slow to react to big movements in the inputs in its predictions. 
Ultimately, the model performed with a validation mean absolute error of 0.476, and a mean test error of 12.459. This means that, in test, the LSTM was off by an average of roughly 12 tweets. Likely, the complexity of the model and the limited size of the data set can explain the discrepancy in validation and test error. In other words, the complexity of the model was causing it to overfit.

Figure Five: Example LSTM Predictions

### Dense Model
The architecture for our dense model can be seen in Figure Six. This is a relatively shallow neural network with only two dense layers with 32 dimensions each. Similarly to our LSTM model we applied a mild L2 regularization to both layers to prevent overfitting due to our relatively small dataset. The output layer is once again a dense layer of dimension 1 to produce our desired predictory total tweet output. 

Figure Six: Keras Sequential Architecture
In this model, we originally used a random train/test split of 80/20, but pivoted to a chronological split (also 80/20 train/test) to account for our intended goal of time series forecasting.   Figure Seven below displays the model's prediction of total tweets for the next day over all of our data, as well as the seven-day average of total tweets in one day. The blue-highlighted section is the ending 20% used as our test set. 

Figure Seven: Sequential Predictions and Actual Tweet Totals
For every run of our sequential model, we examine the Permutation Feature Importance of every feature to determine which features are most important in any given instance of the model. The PFI for the instance corresponding to Figure Seven can be seen below in Figure Eight, below. This Instance of the PFI calculation is a fair representation of the features that are most often considered important. 
Feature
PFI Calculation
tweets_total
5.858348
mean_intertweet_gap
2.949440
time_of_last_tweet_cos    


0.757110
active_hours_count        


0.734427
std_intertweet_gap        
0.709939
night_tweets              
0.616544
retweets_total            
0.562260
burst_count               
0.560508
afternoon_tweets          
0.531879
morning_tweets            
0.482988

Figure Eight: Example Permutation Feature Importance 


### Model Comparison
Our LSTM model was designed to capture the temporal structure of Elon Musk’s tweet posting behavior, and our simpler dense model was designed to learn the overarching pattern based on our daily-feature vectors. 
Our dense neural network reached a test MAE (Mean Average Error) of ~15, meaning that on average, its prediction varied from the actual total tweets by 15 tweets. Our LSTM model reached a test MAE of approximately 10-12 tweets. Considering the volatility of Elon Musk’s tweeting frequency and the limited size of our training data, this prediction error is quite impressive.

## Discussion
Data collection and feature engineering from the CSV file available on xtracker.io was by far the most challenging and tedious aspect of our project. While this data was easily accessible and free, it was not without its flaws; a large percentage of the data was scrambled into the incorrect column or row, making data wrangling difficult. However, once the data was arranged correctly, feature engineering was a streamlined process. As discussed earlier, most of our features were calculated with regard to the time/frequency of the tweets, with a minority of the features based on the actual text’s characteristics. 
We were successful in algebraically crafting features based on the time and content of Elon Musk’s tweets that allowed our models to, with varying accuracy, learn the patterns required to predict the next day’s total tweets. Although with access to more data, more complex neural networks may be able to derive the algebraic patterns we outlined in our features. Due to our limited data, we saw it best to give our networks these patterns pre-calculated so they could spend their limited computational resources on observing these patterns. 

## Conclusion
In this project, we set out to determine whether Elon Musk’s tweeting behavior could be predicted based on his historic patterns in his activity on X. After engineering a set of temporal and text-derived features from his raw timestamped tweeting history, we built two distinct neural networks to forecast the total number of tweets he would post the next day. Considering our best model’s accuracy of a MAE of 10-12 tweets, the LSTM model is very close to being profitably applicable to Polymarket’s Elon Musk tweet frequency market. However, a slightly more accurate model (MAE < 8) would be able to predict greatly safer predictions/bets, considering the bucket size of 20 tweets per bet, as a MAE of 10 may leave more than two buckets as highly probable. With a further regard to application, designing a model to predict the next week’s total instead of the next day would provide another level of possible application benefit. On a weekly scale, being anywhere close to 20-30 tweets within the actual value would be definitely applicable. 

## Sources
https://www.xtracker.io/
https://docs.polymarket.com/polymarket-learn/get-started/what-is-polymarket
https://seaborn.pydata.org/generated/seaborn.heatmap.html
https://keras.io/api/layers/recurrent_layers/lstm/
https://keras.io/api/layers/convolution_layers/
https://keras.io/api/models/sequential/



