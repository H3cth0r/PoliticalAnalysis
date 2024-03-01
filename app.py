from tweety import Twitter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from credentials import username_t, password_t
import seaborn as sns

from flask import Flask, request

from TwitterScrapper import TwitterScrapper
from functionalities import SentimentAnalizer, plotSentimentAnalysis

@app.route("/")
def mainEntry():
    return "Howdy Y'all"

@app.route("/evaluate/candidate/<target>")
def evaluateCandidateTwitter(target):
    args = request.args
    topics = args.getlist("topic")

    candidate_scraper = TwitterScrapper(username_t, password_t, target)
    candidate_scraper.downloadTargetTweets()
    for topic in topics:candidate_scraper.downloadTopicTweets(topic,pages=)

    numeric_user_info = ["author.followers_count", "author.normal_followers_count", "author.friends_count", "author.favourites_count", "author.statuses_count", "author.media_count", "author.listed_count"]
    numeric_cols_per_tweet = ["views", "retweet_counts", "quote_counts", "bookmark_count", "reply_counts", "likes"]

    df_target = candidate_scraper.df
    for col in numeric_user_info + numeric_cols_per_tweet:
        df_target[col] = pd.to_numeric(df_target[col], errors="coerce")

    target_vals = []
    for col in numeric_cols_per_tweet:
        target_vals.append(df_target[(df_target["author.username"] == target)][col].sum())

    barWidth = 0.25
    br1 = np.arange(len(target_vals))
    br2 = [x + barWidth for x in br1]

    return "hello"

@app.route("/evaluate/compare/<target_one>/<target_two>", methods=["POST"])
def evaluateCompareCandidates(target_one, target_two):
    if request.method != "POST":
        return "Ivalid request"

    args = request.args
    json_data = request.get_json()

    # Target One Scraping
    target_one_scraper = TwitterScrapper(username_t, password_t, target_one)
    target_one_scraper.downloadTargetTweets()
    for topic in json_data["target_one_topics"]:
        target_one_scraper.downloadTopicTweets(topic, pages=2)

    # Target Two Scraping
    target_two_scraper = TwitterScrapper(username_t, password_t, target_two)
    target_two_scraper.downloadTargetTweets()
    for topic in json_data["target_two_topics"]:
        target_two_scraper.downloadTopicTweets(topic, pages=2)

    # Dataframe Numeric Columns
    numeric_user_info = ["author.followers_count", "author.normal_followers_count", "author.friends_count", "author.favourites_count", "author.statuses_count", "author.media_count", "author.listed_count"]
    numeric_cols_per_tweet = ["views", "retweet_counts", "quote_counts", "bookmark_count", "reply_counts", "likes"]

    # Generate DFs and process numeric columns
    df_target_one = target_one_scraper.df
    df_target_two = target_two_scraper.df
    for col in numeric_user_info + numeric_cols_per_tweet:
        df_target_one[col] = pd.to_numeric(df_target_one[col], errors="coerce")
        df_target_two[col] = pd.to_numeric(df_target_two[col], errors="coerce")

    # calculate values from numeric columns
    target_one_vals = []
    target_two_vals = []
    for col in numeric_cols_per_tweet:
        target_one_vals.append(df_target_one[(df_target_one["author.username"] == target_one)][cols].sum())
        target_two_vals.append(df_target_two[(df_target_two["author.username"] == target_two)][cols].sum())

    # Plot tweets comparisons - Pie charts
    labels = [target_one, target_two]
    counter = 0
    for i in range(len(target_one_vals)):
        fig, axs = plt.subplots(1, 2, figsize=(15, 5))
        values = [target_one_vals[i], target_two_vals[i]]
        axs[counter].pie(values, labels=labels, outopct="%1.1f%%", startangle=90)
        axs[counter].set_title(f'{numeric_user_info[i]}')
        if counter < 2: counter += 1
        else: counter = 0
        plt.show()

    # Plot tweets data
    target_one_bio = df_target_one[(df_target_one["author.username"] == target_one)].iloc[[0]][numeric_user_info].to_numpy()[0]
    target_two_bio = df_target_two[(df_target_two["author.username"] == target_two)].iloc[[0]][numeric_user_info].to_numpy()[0]
    labels = [target_one, target_two]
    counter = 0
    for i in range(len(target_one_bio)):
        fig, axs = plt.subplots(1, 2, figsize=(15, 5))
        values = [target_one_bio[i], target_two_bio[i]]
        axs[counter].pie(values, labels=labels, outopct="%1.1f%%", startangle=90)
        axs[counter].set_title(f'{numeric_user_info[i]}')
        if counter < 2: counter += 1
        else: counter = 0
        plt.show()
    

    # Sentiment Analyzer
    target_one_SA = SentimentAnalizer()
    target_two_SA = SentimentAnalizer()
    df_target_one[labels] = df_target_one["text"].apply(lambda x: pd.Series(target_one_SA.applyPipeline(x)))
    df_target_two[lables] = df_target_two["text"].apply(lambda x: pd.Series(target_two_SA.applyPipeline(x)))

    df_target_one_mean = df_target_one[df_target_one["author.username"] != target_one][labels].mean()
    df_target_two_mean = df_target_two[df_target_two["author.username"] != target_two][labels].mean()

    # Ploting sentiment analysis
    plotSentimentAnalysis(df_target_one, target_one, df_target_two, target_two)



    return "hello"

if __name__ == "__main__":
    pass
