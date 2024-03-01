class SentimentAnalyzer:
  def __init__(self):
    self.analyser = create_analyzer(task="sentiment", lang="es")
  def applyPreprocess(self, text_t): return preprocess_tweet(text_t)
  def applyAnalysis(self, text_t): return self.analyser.predict(text_t)
  def applyPipeline(self, text_t): return tuple(self.analyser.predict(preprocess_tweet(text_t)).probas.values()) # NEG, NEU, POS

def plotSentimentAnalysis(df_target_one, target_one, df_target_two, target_two):
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
    plt.show()


