# yelp_reviews sentiments

Sentiment analysis on Yelp reviews. 
The reviews are extracted from review.json in Yelp Dataset [https://www.yelp.com/dataset] by using json_to_csv_converter.py. 
For each review, if stars>=4, we say its sentiment is positive, if stars<=2, we say its sentiment is negative. We extracted 130 K reviews with sentiments, split them into train (100 K), validation (10 K) and test (20 K).
The model used for sentiment analysis are LSTM (and bidirectional LSTM) trained from scratch, and fine-tuned DistilBERT from transformers.
