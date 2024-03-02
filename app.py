from tweety import Twitter
import pandas as pd
from credentials import username_t, password_t
# import seaborn as sns

from flask import Flask, request

from functionalities import (
        SentimentAnalyzer,
        analyzeCandidateTwitter,
        plotSentimentAnalysisCompare,
        plotPieComparison,
        analyzeCandidateCorpus,
)

app = Flask(__name__)
sentiment_analyzer = SentimentAnalyzer()

@app.route("/")
def mainEntry():
    return "Howdy Y'all"

@app.route("/evaluate/candidate/<target>")
def evaluateCandidateTwitter(target):
    return "Hi!"

@app.route("/evaluate/compare/<target_one>/<target_two>", methods=["POST"])
def evaluateCompareCandidates(target_one, target_two):
    if request.method != "POST":
        return "Ivalid request"

    args = request.args
    json_data = request.get_json()

    df_targe_one, target_one_vals, target_one_bio, df_target_one_mean, df_targe_one_subset = analyzeCandidateTwitter(username_t, password_t, target_one, sentiment_analyzer, json_data["target_one_topics"])
    df_targe_two, target_two_vals, target_two_bio, df_target_two_mean, df_targe_two_subset = analyzeCandidateTwitter(username_t, password_t, target_two, sentiment_analyzer, json_data["target_two_topics"])

    plotPieComparison(target_one_vals, target_one, target_two_vals, target_two)
    plotPieComparison(target_one_bio, target_one, target_two_bio, target_two)

    plotSentimentAnalysisCompare(df_target_one_mean, target_one, df_target_two_mean, target_two)

    analyzeCandidateCorpus(df_targe_one, target_one)

    return "hello"

if __name__ == "__main__":
    app.run(debug =True)
