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

