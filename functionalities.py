from TwitterScrapper import TwitterScraper
import pandas as pd
from pysentimiento.preprocessing import preprocess_tweet
from pysentimiento import create_analyzer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import requests

from googleapiclient.discovery import build
from google.oauth2 import service_account

import os
from wordcloud import WordCloud
import nltk
from nltk.corpus import PlaintextCorpusReader
from nltk import FreqDist
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')

"""
=================================================================================
=================================================================================
=================================================================================
=================================================================================
"""
def remove_all_content(directory_path="./plots"):
    # Check if the directory exists
    if os.path.exists(directory_path):
        # Iterate over all files and subdirectories in the directory
        for item in os.listdir(directory_path):
            item_path = os.path.join(directory_path, item)

            # Remove files
            if os.path.isfile(item_path):
                os.remove(item_path)

            # Recursively remove subdirectories
            elif os.path.isdir(item_path):
                remove_all_content(item_path)
                os.rmdir(item_path)

def authenticate(SERVICE_ACCOUNT_FILE, SCOPES):
    creds = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    return creds

def upload_file(name_t, PARENT_FOLDER_ID_t, file_path, creds):
    # creds = authenticate()
    service = build("drive", "v3", credentials=creds)

    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%Y_%m_%d_%H_%M_%S")
    name_t = formatted_datetime + name_t 

    file_metadata = {
        "name": name_t,
        "parents": [PARENT_FOLDER_ID_t],
    }

    # media = MediaFileUpload(file_path, resumable=True)

    file = service.files().create(
        body=file_metadata,
        media_body=file_path
    ).execute()

    return name_t
def upload_all_images(directory_path, PARENT_FOLDER_ID, SERVICE_ACCOUNT_FILE, SCOPES):
    creds = authenticate(SERVICE_ACCOUNT_FILE, SCOPES)
    service = build("drive", "v3", credentials=creds)

    file_names = []

    # Loop through all files in the specified directory
    for file_name in os.listdir(directory_path):
        if file_name.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp")):
            file_path = os.path.join(directory_path, file_name)

            # Call the upload_file function for each image file
            new_name = upload_file(file_name, PARENT_FOLDER_ID, file_path, creds)
            file_names.append(new_name)
    return file_names

def saveDataOnSpreadSheet(SPREADSHEET_ID, SERVICE_ACCOUNT_FILE, SCOPES):
    creds = authenticate(SERVICE_ACCOUNT_FILE, SCOPES)
    service = build("sheets", "v4", credentials=creds)
    sheet = service.spreadsheets()
    values = [["Prueba!", "another"]]
    result = sheet.values().append(spreadsheetId=SPREADSHEET_ID,
                                   range="plots!A1",
                                   valueInputOption = "USER_ENTERED",
                                   body={"values":values}
    ).execute()
    return f"{result.get('updates').get('updatedCells')}"

def upload_plots_reference(email_t, images_t, SPREADSHEET_ID, SERVICE_ACCOUNT_FILE, SCOPES):
    creds = authenticate(SERVICE_ACCOUNT_FILE, SCOPES)
    service = build("sheets", "v4", credentials=creds)
    sheet = service.spreadsheets()
    for image in images_t:
        values = [["PlotsImages/"+image, email_t]]
        result = sheet.values().append(spreadsheetId=SPREADSHEET_ID,
                                       range="plots!A1",
                                       valueInputOption = "USER_ENTERED",
                                       body={"values":values}
        ).execute()
    return "done"
def upload_scores(email_t, scores_t, SPREADSHEET_ID, SERVICE_ACCOUNT_FILE, SCOPES):
    creds = authenticate(SERVICE_ACCOUNT_FILE, SCOPES)
    service = build("sheets", "v4", credentials=creds)
    sheet = service.spreadsheets()
    scores_t = scores_t["calculated_scores"]
    keys_scores = scores_t.keys()
    keys_calculated_scores = ["final_score", "current_charge_score" , "desired_charge_score" ,
            "time_in_politics_score" , "positive_reputation" , "not_negative_reputation" ,
            "tweeter_followers_score" , "political_party_score" , "tweeter_retweets_score" ,
            "tweeter_views_score" , "positivity_score" , "requiered_service_score" 
    ]
    values_keys = []
    for key_score in keys_calculated_scores:
        values_keys.append(scores_t[key_score])
    values_keys.append(email_t)
    values = [values_keys]
    result = sheet.values().append(spreadsheetId=SPREADSHEET_ID,
                                   range="scores!A1",
                                   valueInputOption = "USER_ENTERED",
                                   body={"values":values}
    ).execute()
    return "done"

     
"""
=================================================================================
=================================================================================
=================================================================================
=================================================================================
"""
def order_probabilities(output):
    labels = ["NEG", "NEU", "POS"]
    ordered_probas = []
    for label in labels:
      ordered_probas.append(output.probas[label])
    return ordered_probas

class SentimentAnalyzer:
  def __init__(self):
    self.analyser = create_analyzer(task="sentiment", lang="es")
  def applyPreprocess(self, text_t): return preprocess_tweet(text_t)
  def applyAnalysis(self, text_t): return self.analyser.predict(text_t)
  def applyPipeline(self, text_t):
    text_t = str(text_t)
    result = tuple(order_probabilities(self.analyser.predict(self.applyPreprocess(text_t))))
    return result
"""
=======================================================================================
=======================================================================================
=======================================================================================
=======================================================================================
"""
def scatter_plot_with_title(data_frame, x_column, y_column, title):
    plt.figure(figsize=(10, 10))  # Adjust the figure size as needed
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

# def upload_file(name_t, creds_t, PARENT_FOLDER_ID_t, file_path):
def plotPieComparison(target_one_vals, target_one, target_two_vals, target_two, plot_name="bio"):
    numeric_user_info = ["author.followers_count", "author.normal_followers_count", "author.friends_count", "author.favourites_count", "author.statuses_count", "author.media_count", "author.listed_count"]
    numeric_cols_per_tweet = ["views", "retweet_counts", "quote_counts", "bookmark_count", "reply_counts", "likes"]

    labels = [target_one, target_two]
    counter = 0
    fig, axs = plt.subplots(1, 2, figsize=(15, 12))
    img_counter = 0
    pie_plots = []
    for i in range(len(target_one_vals)):
        values = [target_one_vals[i], target_two_vals[i]]
        axs[counter].pie(values, labels=labels, autopct="%1.1f%%", startangle=90)
        axs[counter].set_title(f'{numeric_user_info[i]}')
        pie_plots.append({
            "plot_name" : plot_name,
            "title" : numeric_user_info[i],
            "values" : [int(val) for val in values],
            "labels" : labels
        })
        if counter < 1: counter += 1
        else:
            plt.savefig(f'./plots/{target_one}_vs_{target_two}_pie_{img_counter}_{plot_name}.png')
            fig, axs = plt.subplots(1, 2, figsize=(15, 12))
            counter = 0
            img_counter += 1
        # plt.show()
    return pie_plots 

"""
=======================================================================================
=======================================================================================
=======================================================================================
=======================================================================================
"""

def analyzeCandidateTwitter(username_t, password_t, target_t, target_sa, topics=[]):
    target_twitter_scraper = TwitterScraper(username_t, password_t, target_t)
    try:
        target_twitter_scraper.downloadTargetTweets()
    except:
        print("No data user")

    for topic in topics: target_twitter_scraper.downloadTopicTweets(topic, pages=1)

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


def getInstagramBio(target_t, headers_t):
    """
    https://www.instagram.com/api/v1/users/web_profile_info/?username=claudia_shein
    """
    path = f"https://www.instagram.com/api/v1/users/web_profile_info/?username={target_t}"

    response = requests.get(path, headers=headers_t)
    if response.status_code == 200:
        bio = {"followers": 0, "following": 0} 
        try:
            bio = {
                    "followers" : response.json()["data"]["user"]["edge_followed_by"]["count"],
                    "following" : response.json()["data"]["user"]["edge_follow"]["count"],
                    }
        except:
            bio = {"followers": 0, "following": 0} 
            print("Instagram error")
        return bio
    else:return {"followers": 0, "following": 0} 
    


def calculate_viability_score(budget, num_services, time_before_election):
    service_price = 10000
    optimal_days_before_election = 365
    required_budget = num_services * service_price * 0.3
    viability_score = (budget * service_price * num_services) / ((time_before_election + optimal_days_before_election) * required_budget)
    return min(1, max(0, viability_score))  # Ensure the score is between 0 and 1
def days_until_date(date_str):
    try:
        # Parse the input date string
        date_object = datetime.strptime(date_str, "%d/%m/%Y")

        # Get the current date
        current_date = datetime.now()

        # Calculate the difference in days
        days_until = (date_object - current_date).days

        return max(0, days_until)  # Ensure a non-negative result
    except ValueError:
        print("Invalid date format. Please use DD/MM/YYYY.")
        return None
def calculateScore(json_data, target_t, df_target_one, df_target_two, df_target_one_mean, df_target_two_mean, target_one_bio, target_two_bio, target_one_vals, target_two_vals):
    """
    Political Experience 20%
    - Actual  charge 5%
    - Time in politics 10%
    - Desired charge 5%

    Online Reputation 20%
    - Sentiment Analysis tweets(people) positive 10%
    - Sentiment Analysis tweets(people) negative 10%
    
    Social Network analysis 25%
    - number of followers 10%
        - Twitter
        - Instagram
    - interaction 10%
        - Retweets
        - Views
    - Sentiment analysis (candidate tweets) 5%
    
    Requiered Service 20%
    - Budged 10%
    - Number of services required by candidate 10%
    - time before election 15%

    """

    # percentage
    score_percentage = {
            "current_charge" : 0.05,
            "desired_charge" : 0.05,
            "time_in_politics" : 0.1,
            "positive_reputation" : 0.1,
            "not_negative_reputation" : 0.1,
            "tweeter_followers_score" : 0.1,
            "retweets_score" : 0.1,
            "views_score" : 0.1,
            "positivity_score" : 0.1,
            "requiered_service_score" : 0.1,
            "political_party_score" : 0.1,
    }

    # current Charge 5%
    political_charges = ["Presidente Municipal", "Magistrado", 
                         "Regidor", "Alcalde", "Gobernador", 
                         "Ministro", "Diputado", "Senador", 
                         "Secretario de Estado", 
                         "Presidente"]
    percentage_lambda = lambda charge: (political_charges.index(charge) + 1) * score_percentage["current_charge"] / len(political_charges)
    current_charge_score = 0 if json_data["current_political_charge"] not in political_charges else percentage_lambda(json_data["current_political_charge"])

    # Desired charge
    desired_charge_score = 0 if json_data["desired_political_charge"] not in political_charges else score_percentage["desired_charge"] - percentage_lambda(json_data["desired_political_charge"])

    # Time in politics
    calculate_percentage = lambda value: min(value, 9) * score_percentage["time_in_politics"] / 9 if value < 10 else score_percentage["time_in_politics"]
    time_in_politics_score = calculate_percentage(json_data["time_in_politics"])

    # Online reputation
    labels = ["NEG", "NEU", "POS"]
    positive_reputation = (df_target_one_mean[labels[-1]] * score_percentage["positive_reputation"]) / df_target_two_mean[labels[-1]]
    positive_reputation = score_percentage["positive_reputation"] if positive_reputation > score_percentage["positive_reputation"] else positive_reputation

    not_negative_reputation = (df_target_one_mean[labels[0]] * score_percentage["not_negative_reputation"]) / df_target_two_mean[labels[0]]
    not_negative_reputation = score_percentage["not_negative_reputation"] if not_negative_reputation > score_percentage["not_negative_reputation"] else not_negative_reputation
    # not_negative_reputation = score_percentage["not_negative_reputation"] - not_negative_reputation

    # Twitter comparison
    tweeter_followers_score = (target_one_bio[0] * score_percentage["tweeter_followers_score"])/target_two_bio[0]
    tweeter_followers_score = score_percentage["tweeter_followers_score"] if tweeter_followers_score > score_percentage["tweeter_followers_score"] else tweeter_followers_score 

    # Retweets
    tweeter_retweets_score = (target_one_vals[1] * score_percentage["retweets_score"]) / target_two_vals[1]
    tweeter_retweets_score = score_percentage["retweets_score"] if tweeter_retweets_score > score_percentage["retweets_score"] else tweeter_retweets_score

    # Views
    tweeter_views_score = (target_one_vals[0] * score_percentage["views_score"]) / target_two_vals[1]
    tweeter_views_score = score_percentage["views_score"] if tweeter_views_score > score_percentage["views_score"] else tweeter_views_score

    # Sentiment Analysis positivity candidate
    positivity_score = df_target_one[df_target_one["author.username"] == target_t]["POS"].mean()
    positivity_score = (positivity_score * score_percentage["positivity_score"])
    positivity_score = score_percentage["positivity_score"] if positivity_score > score_percentage["positivity_score"] else positivity_score 

    # Required Service
    requiered_service_score = calculate_viability_score(json_data["budget"], len(json_data["services"]), days_until_date(json_data["end_date"]))
    requiered_service_score = requiered_service_score * score_percentage["requiered_service_score"]

    # political party
    political_parties = {
            "Partido Acción Nacional" : 0.01,
            "Morena" : 0.01,
            "Movimiento Ciudadano" : 0.01,
            "Partido Revolucional Institucional" : 0.01,
            "Partido de la Revolución Democrática" : 0.01,
            "Partido Verde Ecologista" : 0.01,
            "Partido  del Trabajo" : 0.01,
            "Otro":0.01,
    }
    political_party_score = political_parties[json_data["political_party"]]

    # score = current_charge_score + desired_charge_score + time_in_politics_score + positive_reputation + not_negative_reputation + tweeter_followers_score + instagram_followers_score + tweeter_retweets_score + tweeter_views_score + positivity_score + requiered_service_score
    score = current_charge_score + desired_charge_score + political_party_score + time_in_politics_score + positive_reputation + not_negative_reputation + tweeter_followers_score + tweeter_retweets_score + tweeter_views_score + positivity_score + requiered_service_score
    calculated_scores = {
            "final_score" : score,
            "current_charge_score" : current_charge_score,
            "desired_charge_score" : desired_charge_score,
            "time_in_politics_score" : time_in_politics_score,
            "positive_reputation" : positive_reputation,
            "not_negative_reputation" : not_negative_reputation,
            "tweeter_followers_score" : tweeter_followers_score,
            "tweeter_retweets_score" : tweeter_retweets_score,
            "tweeter_views_score" : tweeter_views_score,
            "positivity_score" : positivity_score,
            "requiered_service_score" : requiered_service_score,
            "political_party_score" : political_party_score
    }
    result_scores = {
            "calculated_scores" : calculated_scores,
            "ponderation_scores" :  score_percentage,
    }

    return result_scores

def calculateScoreNoTwitter(json_data):
    # percentage
    score_percentage = {
            "current_charge" : 0.15,
            "desired_charge" : 0.15,
            "time_in_politics" : 0.2,
            "positive_reputation" : 0,
            "not_negative_reputation" : 0,
            "tweeter_followers_score" : 0,
            "retweets_score" : 0,
            "views_score" : 0,
            "positivity_score" : 0,
            "requiered_service_score" : 0.2,
            "political_party_score" : 0.2,
    }

    # current Charge 5%
    political_charges = ["Presidente Municipal", "Magistrado", 
                         "Regidor", "Alcalde", "Gobernador", 
                         "Ministro", "Diputado", "Senador", 
                         "Secretario de Estado", 
                         "Presidente"]
    percentage_lambda = lambda charge: (political_charges.index(charge) + 1) * score_percentage["current_charge"] / len(political_charges)
    current_charge_score = 0 if json_data["current_political_charge"] not in political_charges else percentage_lambda(json_data["current_political_charge"])

    # Desired charge
    desired_charge_score = 0 if json_data["desired_political_charge"] not in political_charges else score_percentage["desired_charge"] - percentage_lambda(json_data["desired_political_charge"])

    # Time in politics
    calculate_percentage = lambda value: min(value, 9) * score_percentage["time_in_politics"] / 9 if value < 10 else score_percentage["time_in_politics"]
    time_in_politics_score = calculate_percentage(json_data["time_in_politics"])

    # Online reputation
    positive_reputation = score_percentage["positive_reputation"]
    not_negative_reputation = score_percentage["not_negative_reputation"]
    # Twitter comparison
    tweeter_followers_score = score_percentage["tweeter_followers_score"]
    # Retweets
    tweeter_retweets_score = score_percentage["retweets_score"]
    # Views
    tweeter_views_score = score_percentage["views_score"]
    # Sentiment Analysis positivity candidate
    positivity_score = score_percentage["positivity_score"]

    # Required Service
    requiered_service_score = calculate_viability_score(json_data["budget"], len(json_data["services"]), days_until_date(json_data["end_date"]))
    requiered_service_score = requiered_service_score * score_percentage["requiered_service_score"]

    # political party
    political_parties = {
            "Partido Acción Nacional" : 0.11,
            "Morena" : 0.14,
            "Movimiento Ciudadano" : 0.15,
            "Partido Revolucional Institucional" : 0.08,
            "Partido de la Revolución Democrática" : 0.11,
            "Partido Verde Ecologista" : 0.05,
            "Partido  del Trabajo" : 0.16,
            "Otro":0.05,
    }
    political_party_score = political_parties[json_data["political_party"]]

    # score = current_charge_score + desired_charge_score + time_in_politics_score + positive_reputation + not_negative_reputation + tweeter_followers_score + instagram_followers_score + tweeter_retweets_score + tweeter_views_score + positivity_score + requiered_service_score
    score = current_charge_score + desired_charge_score + political_party_score + time_in_politics_score + positive_reputation + not_negative_reputation + tweeter_followers_score + tweeter_retweets_score + tweeter_views_score + positivity_score + requiered_service_score
    calculated_scores = {
            "final_score" : score,
            "current_charge_score" : current_charge_score,
            "desired_charge_score" : desired_charge_score,
            "time_in_politics_score" : time_in_politics_score,
            "positive_reputation" : positive_reputation,
            "not_negative_reputation" : not_negative_reputation,
            "tweeter_followers_score" : tweeter_followers_score,
            "tweeter_retweets_score" : tweeter_retweets_score,
            "tweeter_views_score" : tweeter_views_score,
            "positivity_score" : positivity_score,
            "requiered_service_score" : requiered_service_score,
            "political_party_score" : political_party_score
    }
    result_scores = {
            "calculated_scores" : calculated_scores,
            "ponderation_scores" :  score_percentage,
    }

    return result_scores

