# Yelp Reviews Sentiments

## Sentiment analysis on Yelp reviews. 

The reviews are extracted from *review.json* in [Yelp Dataset](https://www.yelp.com/dataset) by using `json_to_csv_converter.py`. 
*130 K* reviews are processed and stored into (`yelp_review_first_130K_with_sentiment.csv`) with `review_dataset_extraction.ipynb`. For each review, if *stars >= 4*, we call its sentiment positive; if *stars <= 2*, we call its sentiment negative. The data are split into train (*100 K*), validation (*10 K*), and test (*20 K*).
Models used for sentiment analysis are LSTM (and Bidirectional LSTM) trained from scratch, and fine-tuned DistilBERT from [HuggingFace Transformers](https://huggingface.co/docs/transformers).

Run the LSTM model `rnn.ipynb` in Jupyter Notebook; 
Run the DistilBERT model with: `python bert.py`.
