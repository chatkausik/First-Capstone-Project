# First-Capstone-Project
First Capstone Project -- Movie IMDB Score Recommendation
Predict IMDB movie rating by Kausik Chattapadhyay (chat.kausik@gmail.com) 

Background:
How can we tell the greatness of a movie before it is released in cinema? 
This question puzzled me for a long time since there is no universal way to claim the goodness of movies. Many people rely on critics to gauge the quality of a film, while others use their instincts. But it takes the time to obtain a reasonable amount of critics review after a movie is released. And human instinct sometimes is unreliable.

Question:
Given that thousands of movies were produced each year, is there a better way for us to tell the greatness of movie without relying on critics or our own instincts?
Will the number of human faces in movie poster correlate with the movie rating?

STEP-1:
Dataset source: https://www.kaggle.com/deepmatrix/imdb-5000-movie-dataset
movie_metadata.csv

# Face detection from movie posters

Will the number of human faces in movie poster correlate with the movie rating?

Movie poster is an important way to make public aware of the movie before its release. It is quite common to see faces in movie posters. It should be pointed out that, most movies have more than one posters. Some may argue it is unreliable to detect faces only from one poster. Well, it is indeed true. However, just like a great book usually having a single cover, I believe a great movie needs to have a "main" poster, the one that the director likes most, or long-remembered by viewers. I have no way to tell which posters are the "main" posters. I assume the poster that I webscraped from IMDB main page of a movie is the "main" poster.

# Correlation analysis
Choosing 15 continuous variables, I plotted the correlation matrix below. Note that "imdb_score" in the matrix denote the IMDB rating score of a movie. The matrix reveals that:

The "cast_total_facebook_likes" has a strong positive correlation with the "actor_1_facebook_likes", and has smaller positive correlation with both "actor_2_facebook_likes" and "actor_3_facebook_likes".

The "movie_facebook_likes" has strong correlation with "num_critic_for_reviews", meaning that the popularity of a movie in social network can be largely affected by the critics.

The "movie_facebook_likes" has relatively large correlation with the "num_voted_users".

The movie "gross" has strong positive correlation with the "num_voted_users".

Surprisingly, there are some pairwise correlations that are perhaps counter-intuitive:

The "imdb_score" has very small but positive correlation with the "director_facebook_likes", meaning a popular director does not necessarily mean his directed movie is great.

The "imdb_score" has very small but positive correlation with the "actor_1_facebook_likes", meaning that an actor is popular in social network does not mean that a movie is high rating if he is the leading actor. So do supporting actors.

The "imdb_score" has small but positive correlation with "duration". Long movies tend to have high rating.

The "imdb_score" has small but negative correlation with "facenumber_in_poster". It is perhaps not a good idea to have many faces in movie poster if a movie wants to be great.

The "imdb_score" has almost no correlation with "budget". Throwing money at a movie will not necessarily make it great.

# Insights:
Regression model can predict the actual imdb_score with less than 50% accuracy based on certain predictors say 'director_facebook_likes','duration', 'actor_1_facebook_likes','actor_2_facebook_likes','actor_3_facebook_likes', 'facenumber_in_poster','title_year', 'budget'.

Since the fitted Random Forest/Bayesian model explains more variability than that of multiple linear regression, I will use the results from Random Forest/Bayesian to explain the insights found so far:

The most important factor that affects movie rating is the duration. The longer the movie is, the higher the rating will be.

Budget is important, although there is no strong correlation between budget and movie rating.

The facebook popularity of director is an important factor to affect a movie rating.

The facebook popularity of the top 3 actors/actresses is important.

The number of faces in movie poster has a non-neglectable effect to the movie rating.

After discretizing the imdb_score to two categories Bad (0 to 7.5) and good(7.6 to 10) and fit to different classification models say Decision Tree, RandomForest and Logistic Regression, i am getting 82% of average model prediction accuracy with 18% error prediction rate for the test. That's good actually.

Whereas with original IMDB_score, different regression models can predict only with 47% accuracy.
