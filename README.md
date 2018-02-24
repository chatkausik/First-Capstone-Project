## Project: IMDB Movie score recommender & Movie review sentiment analysis.
## by Kausik Chattapadhyay (chat.kausik@gmail.com) 

## Stages in the Data Science Life Cycle:

## STEP-1 Purpose, motivation and description:

1. What can we say about the success of a movie before it is released? Are there
certain companies (Pixar?) that have found a consistent formula? Given that major
films costing over $100 million to produce can still flop, this question is more
important than ever to the industry.

2. This question puzzled almost everybody for a long time since there is no universal
way to claim the success of movies. Many people rely on critics to gauge the quality
of a film, while others use their instincts. But it takes the time to obtain a reasonable
amount of critic’s review after a movie is released. And human instinct sometimes is
unreliable.

3. Predicting IMDB Score of a movie before it released in cinemas is my primary goal
for this project without relying on critic’s review data and human instincts data along
with movie review sentiment analysis and character level language modelling.

4. This will benefit all cinema lovers like me or film producers/directors who can get a
high of overview of the probable score of the new movie they are releasing like a
pre-poll forecast.

## Question:
## How to recommend IMDB Movie Score(Score ranges from 0-Low to 10-Excellent) for the new movies?
## How to identify bad(0) or good(1) sentiment from Movie reviews?

5. We have 28 in-dependent variables that I scraped from IMDB site. Roughly
speaking, half of the variables are directly related to movies themselves, such as
title, year, duration, country, color, aspect ratio, language, budget, gross, movie
IMDB Link, human faces in primary poster etc. Another half is related to the people
who involved in the production of the movies, eg, director names, director Facebook
popularity, movie rating from critics, lead and supporting actor names and their
facebook popularity, movie facebook likes, user review comments for already
released movies etc. Used review text and sentiment fields to RNN sentiment
analysis.

6. After discretizing the imdb_score to two categories Bad (0 to 5) and good (6
to 10) and fit to different classification models say Decision Tree, Random Forest, XGBoost
and Logistic Regression, i am getting 82% of average model prediction accuracy
with 18% error prediction rate for the test.

## STEP-2 Data acquisition:

Received the 50000 IMDB movie dataset from kaggle in a csv format.
I was able to obtain all needed 28 variables for 5043 movies and 4906 posters
(998MB), spanning across 100 years in 66 countries. There are 2399 unique director
names, and thousands of actors/actresses. 

## STEP-3 Data Cleaning:
1. In almost all columns, I have missing values except the output variable “IMDB
SCORE” that I am going to predict as part of this problem. That’s good news.

2. After careful analysis, I have found that I have only 2% of data loss if I
remove 80% of the missing value observations. So, I go for it. For some
fields, say actor facebook likes, director facebook likes and movie facebook
likes, I have interpolated the missing value with mean.

3. With pandas info () method, I have analyzed the data types for each features
and corrected some of the data types from object to categorical to save space
in memory and faster processing. Such variables are imdb score (good or
bad) etc.

4. Used pandas describe () method to get descriptive statistics for all numerical
features to identify any outliers if any, max, min, mean, std, median, 25 and
755 percentiles. Luckily no outliers are found.

5. Created a new output variable after discretizing the imdb_score to two
categories Bad (0 to 5) and good (6 to 10).

6. Many machine learning algorithms require input features on the same scale
for optimal performance. Thus, we standardize some columns say gross,
budget before we can feed them to a model.

7. Movie reviews are sequence of words, therefore we want to build an rnn
model to process the words in each sequence, and at the end, classify the
entire sequence to 0 or 1 classes.

8. To prepare the data for input to neural network, we need to encode it into
numeric values. Find the unique words in the entire dataset.

9. So far, we&#39;ve converted sequence of words into sequence of integers. But sequences
currently have different length. In order to generate input data that is compatible with
our RNN architecture, we will need to make sure that all the sequences have the
same length. Create matrix of zeroes where each row corresponds to a sequence of
size 200-- fill the index of words in each sequence from the right hand side of the
matrix.

10. After we preprocess the dataset, we can proceed with splitting the data into separete
training and test sets.

11. Finally, we define one helper function that breaks a given dataset into chunks and
returns a generator to iterate through these chunks(mini-batches).

12. This is very useful techniques for handling memory limitations. This is the
recommended approach for splitting the dataset into mini-batches for training a
neural network, rather than creating all the data splits upfront and keeping them in
memory during training.

## STEP-4 Exploratory data Analysis(EDA):

1. After plotting histogram on the output variable imdb_score, i have found that
distribution is normal like bell curve having mean of 6.44 and most scores are
between 5.8(25%) and 7.2(75%) with std of 1.13.

2. Plotted scatterplots between imdb_score and all other independent variables
to find out the relation between them say strongly correlated or weakly
correlated.

3. Before we begin, it is necessary to investigate the correlation of those variables.
These strongly correlation independent variables can be used easily for my model.

## Face detection from movie posters vs IMDB Score:
1. It should be pointed out that, it is unfair to rate movie solely based on the number of human
faces in poster, because there are great movies whose posters have many faces. For
example, the poster of the movie &quot;(500) Days of summer&quot; has 43 faces, all from the same
actress.
2. But remember that having large face number (&gt; 10) in poster and simultaneously being a
great movie is uncommon based on my findings. 
3. Overall, nearly 95% of all the 4096 posters have less than 5 faces. Besides, Great movies
tend to have fewer faces in posters.
4. If a poster has one or no human faces, we cannot tell if the movie is great simply from
poster.
5. If a poster has more than 5 faces, the likelihood of the movie being great is low.

## IMDB score VS country:
1. USA and UK are the two countries that produced the most number of movies in the past
century, including a large amount of bad movies.
2. The median IMDB scores for both USA and UK are, however, not the highest among all
countries.
3. Some developing countries, such as Libya, Iran, Brazil, and Afghanistan, produced a small
number of movies with high median IMDB scores.

## IMDB score VS movie facebook popularity:
1. From the scatter plot below, we can find that overall, the movies that have very high
facebook likes tend to be the ones that have IMDB scores around 8.0. As we know, IMDB
scores of higher than 8.0 are considered as the greatest movies in the IMDB top 250 list. It is
interesting to see that those greatest movies do not have the highest facebook popularity.

2. The movie &quot;Mad Max&quot; and &quot;Batman vs Superman&quot; both have very high facebook likes, but
their IMDB scores are slightly above 8.0. The movie &quot;The Godfather&quot; is deemed as one of
the greatest movies, but its facebook popularity is hugely dwarfed by that of the &quot;Interstellar&quot;.

## IMDB score VS director facebook popularity:
1. From the scatter plot, it can be seen that the directors who directed movies of rating higher
than 6.0 tend to have more facebook popularity than the ones who directed movies of rating
lower than 6.0.
2. And I listed the top four directors who have the most number of facebook popularity
(Christopher Nolan, David Fincher, Martin Scorsese, and Quentin Tarantino), along with their
four representative movies.

# Correlation analysis
Choosing 15 continuous variables, I plotted the correlation matrix below. Note that "imdb_score" in the matrix denote the IMDB rating score of a movie. The matrix reveals that:

The "cast_total_facebook_likes" has a strong positive correlation with the "actor_1_facebook_likes", and has smaller positive correlation with both "actor_2_facebook_likes" and "actor_3_facebook_likes".

The "movie_facebook_likes" has strong correlation with "num_critic_for_reviews", meaning that the popularity of a movie in social network can be largely affected by the critics.

The "movie_facebook_likes" has relatively large correlation with the "num_voted_users".

The movie "gross" has strong positive correlation with the "num_voted_users".

# Surprisingly, there are some pairwise correlations that are perhaps counter-intuitive:

The "imdb_score" has very small but positive correlation with the "director_facebook_likes", meaning a popular director does not necessarily mean his directed movie is great.

The "imdb_score" has very small but positive correlation with the "actor_1_facebook_likes", meaning that an actor is popular in social network does not mean that a movie is high rating if he is the leading actor. So do supporting actors.

The "imdb_score" has small but positive correlation with "duration". Long movies tend to have high rating.

The "imdb_score" has small but negative correlation with "facenumber_in_poster". It is perhaps not a good idea to have many faces in movie poster if a movie wants to be great.

The "imdb_score" has almost no correlation with "budget". Throwing money at a movie will not necessarily make it great.

## STEP-5 Feature Selection:

1. Although initially I used 28 variables from IMDB website, many variables are not
applicable to predict movie rating. I will therefore only select several critical variables.

2. Correlation matrix shows that multicollinearity exists in the 15 continuous variables.
We need to further remove some variables to reduce multicollinearity.

3. Therefore, I remove the following variables: &quot;gross&quot;, &quot;cast_total_facebook_likes&quot;,
&quot;num_critic_for_reviews&quot;, &quot;num_voted_users&quot;, and &quot;movie_facebook_likes&quot;.
 Some variables are not applicable for prediction, such as &quot;num_voted_users&quot; and
&quot;movie_facebook_likes&quot;, because these numbers will be unavailable before a movie
is released.

4. Used F-regression/chi2 for feature selection in machine learning pipeline from
sklearn Selectkbest.

## STEP-6 Modeling:
1. So far, we cleaned the data and did some exploratory analysis and did some
testing to get the useful features.

2. Many machine learning algorithms require input features on the same scale
for optimal performance. Thus, we standardize some columns say gross,
budget before we can feed them to a model.

3. Before we construct our first model pipeline, we divide the dataset into a
separate training dataset (80 percent of the data) and a separate test dataset
(20 percent of the data).

4. After discretizing the imdb_score to two categories Bad (0 to 7.5) and good (7.6 to
10) and fit to different classification models say Decision Tree, Random Forest and
Logistic Regression, i am getting 82% of average model prediction accuracy with
18% error prediction rate for the test.


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
