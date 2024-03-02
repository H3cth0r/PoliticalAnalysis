from tweety import Twitter
import pandas as pd

class TwitterScraper:
  def __init__(self, username: str, password: str, target: str):
    self.target     = target

    # init session
    self.app        = Twitter("session")
    self.app.start(username, password)
    self.app.sign_in(username, password)

    self.keys = []
    self.df = pd.DataFrame()

  def get_all_keys(self, data):
    keys = set()
    def extract_keys(obj, prefix=""):
      for key, value in obj.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if not isinstance(value, dict):
          keys.add(full_key)
        if isinstance(value, dict):
          extract_keys(value, prefix=full_key)
    for item in data:
      if isinstance(item, dict):
        extract_keys(item)
    self.keys = list(keys)
    return
  def createDataFrame(self, data, keys):
    result_dict = {key: [] for key in keys}
    for item_ in data:
      for key in keys:
        splitted = key.split(".")
        current_value = item_
        try:
          for part in splitted:
            if current_value is not None and part in current_value:
              current_value = current_value[part]
            else:
              current_value = None
              break
          result_dict[key].append(current_value)
        except TypeError:
          result_dict[key].append(None)
    return result_dict

  def processTweets(self, tweets_arr):
    if not self.keys:
      self.get_all_keys(tweets_arr)
    data_df = self.createDataFrame(tweets_arr, self.keys)
    return pd.DataFrame(data_df)

  def joinToDataFrame(self, df_t):
    if self.df.empty:
      self.df = df_t
    else:
      self.df = pd.concat([self.df, df_t], axis=0, ignore_index=True)

  def downloadTargetTweets(self, pages=1, join: bool = True, returnIt:bool=False):
    user = self.app.get_user_info(self.target)
    user_tweets = self.app.get_tweets(user, pages=pages)
    processed = self.processTweets(user_tweets)
    self.joinToDataFrame(processed)
    if returnIt:
      return processed
    return

  def downloadTopicTweets(self, topic: str, pages: int = 1, join:bool=True, returnIt:bool=False):
    lookout = self.app.search(topic, pages=pages)
    processed = self.processTweets(lookout)
    self.joinToDataFrame(processed)
    if returnIt:
      return processed
    return
