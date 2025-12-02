# NeuralNetFinalProj

## data
elonmusk.csv is the raw data. **daily_features.csv** contains the "summaries" for each day- i.e., the features for a given day. **tweets_with_partial_features.csv** contains the "partial day summaries" we talked about during the most recent class. **tweettimes6.csv** contains some data for target values, that is, target values if we want to target the end-of-week total number of tweets, not just the following day. 

## models
**lstm_tf.py** contains the lstm that Adam has been working on to predict the number of tweets for the following day of the last. **setupdata.py** contains (at the bottom) the lstm that Beckett was working using the partial days, and predicting the number of tweets at the end of the week.

## questions
Since we determined its a more reasonable goal to predict the number of tweets in the following day, how can we improve Adam's lstm to have more accuracy there? Also do you think it would be possible to predict the final number of tweets at the end of the day have the last value in the lstm sequence before prediction be a partial day? What do you think is flawed about using the idea of a partial day? Please also take a look at the code files mentioned above and give some general feedback on the models- that would be very helpful especially because we havent done anything besides code yet. We're pretty confident with the way we generated our data / features but if you want to take a look at how we did that as well, there are some files like **FIRST.py** and **SECOND.py** that help with that. Thanks for the feedback!
