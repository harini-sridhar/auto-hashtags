
import nltk
from nltk.corpus import stopwords
import string
from gensim.models import KeyedVectors
import pickle

def remove_stopwords(caption):
    # Remove stopwords from caption
    stopWords = set(stopwords.words('english'))
    # Add punctuation to stopWords
    for p in string.punctuation:
        stopWords.add(p)
    # Filter words in stopWords, words less than 2 chars and verbs
    caption_tags = [word for word in caption if word not in stopWords]
    caption_tags = [word for word in caption_tags if len(word) > 2]
    caption_tags = [word for word in caption_tags if 'ing' not in word]

    return caption_tags


# Find caption tag similarities with categories
def similarity_scores(caption, hashtag):

    categories = list(hashtag.keys())
    print(categories)

    # Set a similarity threshold
    sim_threshold = 0.7

    # Use the glove model
    model = KeyedVectors.load_word2vec_format(
        './models/glove/glove.6B/glove.6B.100d.txt.word2vec',
        binary=False, limit=100000)

    # Find the best category match for each word in the caption
    best_match = []
    for token in caption:
        best_category = model.most_similar_to_given(token, categories)
        # Only choose category with similarity greater than the threshold
        if model.similarity(token, best_category) > sim_threshold:
            best_match.append((token,best_category))

    return best_match


# Get the relevant popular hashtags
def get_hashtags (caption, hashtag):
    best_category_matches = similarity_scores(caption, hashtag)
    insta_tags=[]
    for categ in best_category_matches:
        insta_tags.extend(hashtag[categ[1]])
    return insta_tags
