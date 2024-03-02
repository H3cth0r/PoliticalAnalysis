from TwitterScrapper import TwitterScraper
import pandas as pd
from pysentimiento.preprocessing import preprocess_tweet
from pysentimiento import create_analyzer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import os
from wordcloud import WordCloud
import nltk
from nltk.corpus import PlaintextCorpusReader
from nltk import FreqDist
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')

class SentimentAnalyzer:
  def __init__(self):
    self.analyser = create_analyzer(task="sentiment", lang="es")
  def applyPreprocess(self, text_t): return preprocess_tweet(text_t)
  def applyAnalysis(self, text_t): return self.analyser.predict(text_t)
  def applyPipeline(self, text_t):
    text_t = str(text_t)
    result = self.analyser.predict(self.applyPreprocess(text_t))
    #return tuple(self.analyser.predict(preprocess_tweet(text_t)).probas.values()) # NEG, NEU, POS
    return tuple(result.probas.values())
"""
=======================================================================================
=======================================================================================
=======================================================================================
=======================================================================================
"""
def scatter_plot_with_title(data_frame, x_column, y_column, title):
    plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
    plt.scatter(data_frame[x_column], data_frame[y_column])
    plt.title(title)
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    # plt.show()
    plt.savefig(f'./plots/{title}scatter.png')

def get_frequency_corpus(my_corpus):
  frequent_words = {}
  for word in my_corpus.words():
    if word in frequent_words:
      frequent_words[word]+=1
    else:
      frequent_words[word]=1
  keys = list(frequent_words.keys())
  values = list(frequent_words.values())
  sorted_value_index = np.flip(np.argsort(values))
  sorted_dict = [(keys[i], values[i]) for i in sorted_value_index]
  return sorted_dict
def remove_stopwords(frequency_list):
    stop_words = set(stopwords.words("spanish"))
    filtered_tokens = [(token, count) for token,count in frequency_list if token.lower() not in stop_words]
    return filtered_tokens
def clean_alphanumeric(frequency_list):
   return  [(token, count) for token,count in frequency_list if token.isalnum()]
def lowercase_tokens(frequency_list):
  return [(token.lower(), count) for token, count in frequency_list]
def clean_unnecessary_tokens(frequency_list, list_tokens):
  return [(token, count) for token,count in frequency_list if token not in list_tokens]
def normalize_document(doc):
    doc = re.sub(r'[^a-zA-Z\s]', '', doc, re.I|re.A)
    doc = doc.lower()
    doc = doc.strip()
    tokens = wpt.tokenize(doc)
    filtered_tokens = [token for token in tokens if token not in stop_words]
    doc = ' '.join(filtered_tokens)
    return doc

"""
=======================================================================================
=======================================================================================
=======================================================================================
=======================================================================================
"""

def plotSentimentAnalysisCompare(df_target_one, target_one, df_target_two, target_two):
    labels = ["NEG", "NEU", "POS"]
    fig, ax = plt.subplots()
    width = 0.35
    x = np.arange(len(labels))
    users = [target_one, target_two]
    
    bar1 = ax.bar(x - width/2, df_target_one, width, label=target_one)
    bar2 = ax.bar(x + width/2, df_target_two, width, label=target_two)

    ax.set_ylabel('Mean Sentiment Score')
    ax.set_title('Sentiment Analysis Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    # Add text annotations
    for bar, user in zip([bar1, bar2], users):
        for rect in bar:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2, height, f'{height:.2f}', ha='center', va='bottom', color='black', fontsize=8)
    # plt.show()
    plt.savefig(f'./plots/{target_one}_vs_{target_two}_sentiments.png')

def plotPieComparison(target_one_vals, target_one, target_two_vals, target_two):
    numeric_user_info = ["author.followers_count", "author.normal_followers_count", "author.friends_count", "author.favourites_count", "author.statuses_count", "author.media_count", "author.listed_count"]
    numeric_cols_per_tweet = ["views", "retweet_counts", "quote_counts", "bookmark_count", "reply_counts", "likes"]

    labels = [target_one, target_two]
    counter = 0
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    img_counter = 0
    for i in range(len(target_one_vals)):
        values = [target_one_vals[i], target_two_vals[i]]
        axs[counter].pie(values, labels=labels, autopct="%1.1f%%", startangle=90)
        axs[counter].set_title(f'{numeric_user_info[i]}')
        if counter < 1: counter += 1
        else:
            plt.savefig(f'./plots/{target_one}_vs_{target_two}_pie_{img_counter}.png')
            fig, axs = plt.subplots(1, 2, figsize=(15, 5))
            counter = 0
            img_counter += 1
        # plt.show()

"""
=======================================================================================
=======================================================================================
=======================================================================================
=======================================================================================
"""

def analyzeCandidateTwitter(username_t, password_t, target_t, target_sa, topics=[]):
    target_twitter_scraper = TwitterScraper(username_t, password_t, target_t)
    target_twitter_scraper.downloadTargetTweets()

    for topic in topics: target_twitter_scraper.downloadTopicTweets(topic, pages=2)

    numeric_user_info = ["author.followers_count", "author.normal_followers_count", "author.friends_count", "author.favourites_count", "author.statuses_count", "author.media_count", "author.listed_count"]
    numeric_cols_per_tweet = ["views", "retweet_counts", "quote_counts", "bookmark_count", "reply_counts", "likes"]

    df_target = target_twitter_scraper.df
    for col in numeric_cols_per_tweet + numeric_user_info:
        df_target[col] =pd.to_numeric(df_target[col], errors="coerce")

    target_vals = []
    for col in numeric_cols_per_tweet:
        target_vals.append(df_target[(df_target["author.username"] == target_t)][col].sum())

    target_bio = df_target[(df_target["author.username"] == target_t)].iloc[[0]][numeric_user_info].to_numpy()[0]

    labels = ["NEG", "NEU", "POS"]
    df_target[labels] = df_target["text"].apply(lambda x: pd.Series(target_sa.applyPipeline(x)))

    df_target_mean = df_target[df_target["author.username"] != target_t][labels].mean()

    df_target_subset = df_target[df_target["author.username"] != target_t][["NEG", "NEU", "POS"]]

    return df_target, target_vals, target_bio, df_target_mean, df_target_subset

def analyzeCandidateCorpus(df_target, target_t):
    corpus_root = "./corpus_save"
    os.makedirs(corpus_root, exist_ok=True)

    for i, row in df_target[df_target["author.username"] != target_t].iterrows():
        filename = f"{row['author.username']}_{i}.txt"
        with open(f"{corpus_root}/{filename}", "w", encoding="utf-8") as file:
            file.write(row['text'])
    
    corpus = PlaintextCorpusReader(corpus_root, '.*\.txt')
    frequency = get_frequency_corpus(corpus)

    df              = pd.DataFrame(frequency, columns=["token", "count"])
    df["position"] = df.apply((lambda x : x.name), axis=1)

    df["zipf"]= df.apply((lambda x: x.name*x["count"]), axis=1)

    new_frequency = remove_stopwords(frequency)

    new_frequency = clean_alphanumeric(new_frequency)

    tokens_to_remove  = ["co", "t", "https"]
    new_frequency     = clean_unnecessary_tokens(new_frequency, tokens_to_remove)

    new_frequency     = lowercase_tokens(new_frequency)

    df = pd.DataFrame(new_frequency, columns =['token', 'count'])
    df["position"]= df.apply((lambda x: x.name), axis=1)

    df["zipf"]= df.apply((lambda x: x.name*x["count"]), axis=1)

    words_from_dataframe = ' '.join(df.values.flatten().astype(str))
    oc_cloud= WordCloud(background_color='white', max_words=300, max_font_size=40,random_state=1).generate(words_from_dataframe)
    oc_cloud.to_file(f'./plots/{target_t}_wordcloud.png')
