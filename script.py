#Project_Description : This work is a Sentiment anaylsis program that analyses the tweets fetched from Twitter using Python.

# Importing the Libraries
import tweepy
from textblob import TextBlob
from wordcloud import WordCloud
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')


# loading the API Credentials Data
from google.colab import files
uploaded = files.upload()


# Reading the CSV File
import pandas as pd
log = pd.read_csv('login.csv')


# Getting the Twitter API Credentials
consumerKey = log['key'][0]
consumerSecret = log['key'][1]
accessToken = log['key'][2]
accessTokenSecret = log['key'][3]


# Creating the Authentication Object.
authenticate = tweepy.OAuthHandler(consumerKey, consumerSecret)

# Seting the Access Token and Access Token Secret.
authenticate.set_access_token(accessToken, accessTokenSecret)

# Creating the API object while passing in the Auth Information.
api = tweepy.API(authenticate, wait_on_rate_limit = True)


# Extracting 100 tweets from a twitter user
# In this project, I have choosen the twitter of Founder. KIIT & KISS 
#DR. ACHYUTA SAMANATA for the purpose of Sentiment Analysis.
posts = api.user_timeline(screen_name = "achyuta_samanta", count = 100, 
                          lang = "en", tweet_mode = "extended")


# Printing the last 10 tweets from the account 'achyuta_samanata'.
print("Show the 10 recent tweets: \n") # printing the 10 latest tweets till Sunday, 
# 2 PM(25th April).
i = 1
for tweet in posts[0:10]:
  print(str(i) + '> ' + tweet.full_text + '\n')
  i = i + 1
  
  
  # Creating a dataframe with a column called tweets.
df = pd.DataFrame( [tweet.full_text for tweet in posts], columns = ['Tweets'])

# Showing the first 10 rows/tweets of data.
df.head(10)


# Cleaning the Data, by creating a function to clear special characters in the tweets.
def cleanTxt(text):
  text = re.sub(r'@[A-Za-z0-9+]', '',text) #Removing @mentions
  text = re.sub(r'#', '',text) #Removing the '#' symbol
  text = re.sub(r'RT[\s]+', '',text) #Removing the RT
  text = re.sub(r"http\S+", "", text) #Removing the hyperlinks
  text = re.sub(r'"', '',text)

  return text

#Cleaning the text
df['Tweets'] = df['Tweets'].apply(cleanTxt)

#Showing the cleaned text.
df



# Creating a function to get the Subjectivity
def getSubjectivity(text):
  return TextBlob(text).sentiment.subjectivity

# Creating a function to get the Polarity
def getPolarity(text):
  return TextBlob(text).sentiment.polarity

# Creating two columns for Subjectivity and Polarity respectively
df['Subjectivity'] = df['Tweets'].apply(getSubjectivity)

df['Polarity'] = df['Tweets'].apply(getPolarity)

# Showing the results
df


# Plotting the Word Cloud.
frequent_Words = ' '.join(twts for twts in df['Tweets'])
WordCloud = WordCloud(width = 500, height=400,random_state = 21,
                      max_font_size = 120).generate(frequent_Words)
   
plt.imshow(WordCloud, interpolation="bilinear")
plt.axis('off')
plt.show()


# Creating a function to compute Postive, Neutral and Negative Analysis of Polarity of the Tweets. 

def getAnalysis(score):
  if score > 0:
    return 'Postive'
  elif score == 0:
    return 'Neutral'
  else:
    return 'Negative'

df['Analysis'] = df['Polarity'].apply(getAnalysis)

# Showing the Dataframes
df


 Printing all of the Postive Polarity Tweets
k = 1 
sortedDF = df.sort_values(by=['Polarity'])
for i in range(0, sortedDF.shape[0]):
  if(sortedDF['Analysis'][i]=='Postive'):
    print(str(k) + '> ' +sortedDF['Tweets'][i])
    print()
    k = k+1


# Printing all of the Negative Polarity Tweets
k = 1 
sortedDF = df.sort_values(by=['Polarity'], ascending='False')
for i in range(0, sortedDF.shape[0]):
  if(sortedDF['Analysis'][i]=='Negative'):
    print(str(k) + '> ' +sortedDF['Tweets'][i])
    print()
    k = k+1
    
# Printing all of the Neutral Polarity Tweets
neutral_tweets = df[df.Analysis == 'Neutral']
neutral_tweets = neutral_tweets['Tweets']
neutral_tweets


# Ploting the Subjectivity vs Polarity
plt.figure(figsize=(8,8))
for i in range(0, df.shape[0]):
  plt.scatter(df['Polarity'][i], df['Subjectivity'][i], color='Red')

plt.title('Sentiment Analysis')
plt.xlabel('Polarity')
plt.ylabel('Subjectivity')
plt.show()


# Getting the percentage of the Postive Tweets. 
postive_tweets = df[df.Analysis == 'Postive']
postive_tweets = postive_tweets['Tweets']

round((postive_tweets.shape[0]/df.shape[0])*100, 1)


# Getting the percentage of the Negative Tweets. 
negative_tweets = df[df.Analysis == 'Negative']
negative_tweets = negative_tweets['Tweets']

round((negative_tweets.shape[0]/df.shape[0])*100, 1)


# Getting the percentage Neutral Polarity Tweets
neutral_tweets = df[df.Analysis == 'Neutral']
neutral_tweets = neutral_tweets['Tweets']


round((neutral_tweets.shape[0]/df.shape[0])*100, 1)



# Showing the value count of tweets of different polarities.
 df['Analysis'].value_counts()
 # Plotting and visualizing the counts
 plt.figure(figsize=(6,8))
 plt.title('Sentiment Analysis')
 plt.xlabel('Sentiment')
 plt.ylabel('Counts')
 df['Analysis'].value_counts().plot(kind='bar')
 plt.show



## declare the variables for the pie chart, using the Counter variables for “sizes”
labels = 'Positive', 'Negative', 'Neutral'
sizes = [63.0, 8.0, 29.0]
colors = ['blue', 'yellow', 'black']

## use matplotlib to plot the chart
plt.figure(figsize=(10,10))
plt.pie(sizes, labels = labels, colors = colors, shadow = True, startangle = 90)
plt.title("pie-chart presentation as per polarity(+ve,-ve,Neutral)")
plt.show()



#_________________________________THE END__________________________________
