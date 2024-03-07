from tweety import Twitter
import pandas as pd
from credentials import username_t, password_t, headers, PARENT_FOLDER_ID, SPREAD_SHEET_ID
# import seaborn as sns
import os

from flask import Flask, request, jsonify

from functionalities import (
        SentimentAnalyzer,
        analyzeCandidateTwitter,
        plotSentimentAnalysisCompare,
        plotPieComparison,
        analyzeCandidateCorpus,
        getInstagramBio,
        calculateScore,
        upload_all_images,
        remove_all_content,
        saveDataOnSpreadSheet,
        upload_plots_reference,
        upload_scores,
)

SCOPES = ["https://www.googleapis.com/auth/drive"]
SERVICE_ACCOUNT_FILE = "./service_account.json"


app = Flask(__name__)
sentiment_analyzer = SentimentAnalyzer()

os.makedirs("./plots", exist_ok=True)


@app.route("/")
def mainEntry():
    return "Howdy Y'all"

@app.route("/test", methods=["POST"])
def evaluateCandidateTwitter():
    saveDataOnSpreadSheet(SPREAD_SHEET_ID, SERVICE_ACCOUNT_FILE, SCOPES)
    return "done"

@app.route("/evaluate/compare/<target_one>/<target_two>", methods=["POST"])
def evaluateCompareCandidates(target_one, target_two):
    if request.method != "POST":
        return "Ivalid request"

    args = request.args
    json_data = request.get_json()

    df_targe_one, target_one_vals, target_one_bio, df_target_one_mean, df_targe_one_subset = analyzeCandidateTwitter(username_t, password_t, target_one, sentiment_analyzer, json_data["target_one_topics"])
    df_targe_two, target_two_vals, target_two_bio, df_target_two_mean, df_targe_two_subset = analyzeCandidateTwitter(username_t, password_t, target_two, sentiment_analyzer, json_data["target_two_topics"])

    plots_vals = plotPieComparison(target_one_vals, target_one, target_two_vals, target_two, plot_name="vals")
    plots_bios = plotPieComparison(target_one_bio, target_one, target_two_bio, target_two, plot_name="bio")

    plotSentimentAnalysisCompare(df_target_one_mean, target_one, df_target_two_mean, target_two)

    analyzeCandidateCorpus(df_targe_one, target_one)

    # target_one_instagram_bio = getInstagramBio(json_data["target_one_instagram"], headers) 
    # target_two_instagram_bio = getInstagramBio(json_data["target_two_instagram"], headers) 

    # result_score = calculateScore(json_data, target_one, df_targe_one, df_targe_two, df_target_one_mean, df_target_two_mean, target_one_bio, target_two_bio, target_one_vals, target_two_vals, target_one_instagram_bio, target_two_instagram_bio)
    result_score = calculateScore(json_data, target_one, df_targe_one, df_targe_two, df_target_one_mean, df_target_two_mean, target_one_bio, target_two_bio, target_one_vals, target_two_vals)

    plot_names = upload_all_images("./plots", PARENT_FOLDER_ID, SERVICE_ACCOUNT_FILE, SCOPES)

    upload_plots_reference(json_data["email"], plot_names, SPREAD_SHEET_ID, SERVICE_ACCOUNT_FILE, SCOPES)
    upload_scores(json_data["email"], result_score, SPREAD_SHEET_ID, SERVICE_ACCOUNT_FILE, SCOPES)
    return_dict = {
            "score" : result_score,
            "plots_imgs" : plot_names,
            "plots_pie" : plots_vals + plots_bios
    }
    remove_all_content("./plots")
    return jsonify(return_dict)

if __name__ == "__main__":
    # app.run(debug =True)
    # CMD [ "python", "-m" , "flask", "run", "--host=0.0.0.0"]
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
