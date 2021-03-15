
"""
Created on Sun Mar 14 10:55:58 2021

@author: Gokul A.
"""
import pandas as pd  # use pandas==1.0.5
#pd.__version__
#!pip install pandas==1.0.5

import numpy as np  # use numpy==1.20.1
#np.__version__
import matplotlib.pyplot as plt
import re   # use re==2.2.1
#re.__version__
import pickle

"""
downloaded json file from 

http://deepyeti.ucsd.edu/jianmo/amazon/categoryFiles/Clothing_Shoes_and_Jewelry.json.gz

"""

def data_coll():
    df_reader = pd.read_json(r'Clothing_Shoes_and_Jewelry.json', lines = True, chunksize = 100000 )

    counter = 1
    for chunk in df_reader:
        new_df = pd.DataFrame(chunk[['overall','reviewText','summary']])
        new_df1 = new_df[new_df['overall'] == 5].sample(4000)
        new_df2 = new_df[new_df['overall'] == 4].sample(4000)
        new_df3 = new_df[new_df['overall'] == 3].sample(8000)
        new_df4 = new_df[new_df['overall'] == 2].sample(4000)
        new_df5 = new_df[new_df['overall'] == 1].sample(4000)
    
        new_df6 = pd.concat([new_df1,new_df2,new_df3,new_df4,new_df5], axis = 0, ignore_index = True)
    
        new_df6.to_csv(str(counter)+".csv", index = False)
    
        new_df = None
        counter = counter + 1
    
    from glob import glob

    filenames = glob('*.csv')

    #['1.csv','2.csv'.......]


    dataframes = [pd.read_csv(f) for f in filenames]

    frame = pd.concat(dataframes, axis = 0, ignore_index = True)

    frame.to_csv('balanced_review.csv', index = False)
    

def data_clean_wordcloud():
    global df
    df = pd.read_csv('balanced_review.csv')

    df.columns.tolist()

    df.head()
    df['overall'].value_counts()

    df.isnull().any(axis = 0)

    df.isnull().any(axis = 1)

    df[df.isnull().any(axis = 1)]

    df.dropna(inplace = True)

    df['overall'] != 3

    df = df[df['overall'] != 3]

    df['Positivity'] = np.where(df['overall'] > 3, 1, 0 )
    
    df['Positivity'].value_counts()
    
    Review_text=list(df["reviewText"])
    # Joinining all the reviews into single paragraph 
    text_rev_string = " ".join(Review_text)
  
    # Removing unwanted symbols incase if exists
    text_rev_string = re.sub("[^A-Za-z" "]+"," ",text_rev_string).lower()
    text_rev_string = re.sub("[0-9" "]+"," ",text_rev_string)

    text_reviews_words = text_rev_string.split(" ")

    with open(r"stop.txt","r") as sw:
        stopwords = sw.read()
        stopwords = stopwords.split("\n")

    text_reviews_words = [w for w in text_reviews_words if not w in stopwords]

    # Joinining all the reviews into single paragraph 
    text_rev_string = " ".join(text_reviews_words)
    
    from wordcloud import WordCloud
    wordcloud_ip = WordCloud(
                          background_color='black',
                          width=1800,
                          height=1400
                          ).generate(text_rev_string)

    plt.imshow(wordcloud_ip)

    # positive words # Choose the path for +ve words stored in system
    with open(r"positive-words.txt","r") as pos:
        poswords = pos.read().split("\n")
  

    # negative words  Choose path for -ve words stored in system
    with open(r"negative-words.txt","r") as neg:
        negwords = neg.read().split("\n")


    # negative word cloud
    # Choosing the only words which are present in negwords
    text_neg_in_neg = " ".join ([w for w in text_reviews_words if w in negwords])

    wordcloud_neg_in_neg = WordCloud(
                          background_color='black',
                          width=1800,
                          height=1400
                          ).generate(text_neg_in_neg)

    plt.imshow(wordcloud_neg_in_neg)

    # Positive word cloud
    # Choosing the only words which are present in positive words
    text_pos_in_pos = " ".join ([w for w in text_reviews_words if w in poswords])
    wordcloud_pos_in_pos = WordCloud(
                          background_color='black',
                          width=1800,
                          height=1400
                          ).generate(text_pos_in_pos)

    plt.imshow(wordcloud_pos_in_pos)
    
def make_model():
    from sklearn.model_selection import train_test_split


    features_train, features_test, labels_train, labels_test = train_test_split(df['reviewText'], df['Positivity'], random_state = 42 ) 
    """
    features_train reviewText 791004
    labels_train positivity 791004
    features_test 263668
    labels_test 263668
    """
    global vect
    from sklearn.feature_extraction.text import CountVectorizer


    vect = CountVectorizer().fit(features_train)    
    
    features_train_vectorized = vect.transform(features_train)
    #version 01
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression()
    model.fit(features_train_vectorized, labels_train)


    predictions = model.predict(vect.transform(features_test))


    from sklearn.metrics import roc_auc_score
    roc_auc_score(labels_test, predictions)    
    
    #TF-IDF - term frequency inverse document frequency
    #version 02

    features_train, features_test, labels_train, labels_test = train_test_split(df['reviewText'], df['Positivity'], random_state = 42 ) 


    from sklearn.feature_extraction.text import TfidfVectorizer

    vect = TfidfVectorizer(min_df = 5).fit(features_train)

    len(vect.get_feature_names())
    #26705
    #imp attribute
    vect.vocabulary_


    features_train_vectorized = vect.transform(features_train)

    from sklearn.linear_model import LogisticRegression
    global model
    model = LogisticRegression()
    model.fit(features_train_vectorized, labels_train)


    predictions = model.predict(vect.transform(features_test))


    from sklearn.metrics import roc_auc_score
    roc_auc_score(labels_test, predictions)
    #90.49 accuracy

def pickle_model():
    pkl_filename = "pickle_model.pkl"

    with open(pkl_filename, 'wb') as file:
        pickle.dump(model, file)
    
    pickle.dump(vect.vocabulary_,open('feature.pkl','wb'))
    
def main():
    global model
    global df
    global vect
    data_coll()
    data_clean_wordcloud()
    make_model()
    pickle_model()
    vect = None
    model = None
    df = None
if __name__ == '__main__':
    main()    
    
    