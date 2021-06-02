from flask import Flask, render_template, url_for, request
from numpy.lib.function_base import append
import pandas as pd
from scipy.sparse import csr_matrix
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn import metrics
import re
import string
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')

class Predict():

    def Recommend_Books(self,book):
        books_data = pd.read_csv('./Datasets/GoodBooks/books.csv',error_bad_lines = False)
        tags_data = pd.read_csv('./Datasets/GoodBooks/book_tags.csv')
        ratings_data = pd.read_csv('./Datasets/GoodBooks/ratings.csv')
        book_tags = pd.read_csv('./Datasets/GoodBooks/tags.csv')
        # stop_words=set(STOPWORDS)
        content_data = books_data
        content_data = content_data.astype(str)
        content_data['content'] = content_data['original_title'] + ' ' + content_data['authors'] + ' ' + content_data['average_rating']
        content_data = content_data.reset_index()
        indices = pd.Series(content_data.index, index=content_data['original_title'])
        #removing stopwords
        tfidf = TfidfVectorizer()

        #Construct the required TF-IDF matrix by fitting and transforming the data
        tfidf_matrix = tfidf.fit_transform(content_data['authors'])

        cosine_sim_author = linear_kernel(tfidf_matrix, tfidf_matrix)
        def get_recommendations_author_books(book, cosine_sim=cosine_sim_author):
            idx = indices[book]
            # print(idx)
            # Get the pairwsie similarity scores of all books with that book
            sim_scores = list(enumerate(cosine_sim[idx]))
            # print(sim_scores)
            # Sort the books based on the similarity scores
            sim_scores = sorted(sim_scores,key=lambda x: x[1],reverse=True)[:10000]

            # Get the scores of the 10 most similar books
            sim_scores = sim_scores[1:6]

            # Get the book indices
            book_indices = [i[0] for i in sim_scores]

            # Return the top 10 most similar books
            author_book=[]
            author_book.append(content_data['isbn'].iloc[book_indices].tolist())
            author_book.append(content_data['original_title'].iloc[book_indices].tolist())
            author_book.append(content_data['authors'].iloc[book_indices].tolist())
            author_book.append(content_data['average_rating'].iloc[book_indices].tolist())
            author_book.append(content_data['image_url'].iloc[book_indices].tolist())
            #author_book.append(content_data['content'].ilo_indices].tolist())
            return list(author_book)
        author_books = get_recommendations_author_books(book, cosine_sim_author)
        
        count = CountVectorizer()
        count_matrix = count.fit_transform(content_data['content'])
        cosine_sim_content = cosine_similarity(count_matrix, count_matrix)

        def get_recommendations(title, cosine_sim=cosine_sim_content):
            idx = indices[title]

            # Get the pairwsie similarity scores of all books with that book
            sim_scores = list(enumerate(cosine_sim[idx]))

            # Sort the books based on the similarity scores
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[:10000]

            # Get the scores of the 10 most similar books
            sim_scores = sim_scores[1:11]
            # Get the book indices
            book_indices = [i[0] for i in sim_scores]
            recom_book=[]
            recom_book.append(content_data['isbn'].iloc[book_indices].tolist())
            recom_book.append(content_data['original_title'].iloc[book_indices].tolist())
            recom_book.append(content_data['authors'].iloc[book_indices].tolist())
            recom_book.append(content_data['average_rating'].iloc[book_indices].tolist())
            recom_book.append(content_data['image_url'].iloc[book_indices].tolist())
            #author_book.append(content_data['content'].ilo_indices].tolist())
            return list(recom_book)
        recom = get_recommendations(book, cosine_sim_content)

        return author_books,recom


@app.route('/predict', methods=['POST'])
def predict():
    global KNN_Recommended_Books
    if request.method == 'POST':
        ICF = Predict()
        book = request.form['book']
        data = book

        Author_Recommended_Books,Recommended_Books= ICF.Recommend_Books(data)
        print(Author_Recommended_Books)
        return render_template('new.html', books1=Author_Recommended_Books,books2=Recommended_Books)

if __name__ == '__main__':
    app.run(debug=True)

