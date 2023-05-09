import pickle
import numpy as np
import streamlit as st
st.header('Books Recommender system using Machine Learning')
model=pickle.load(open('C:/Users/moham/OneDrive/Bureau/data science/book_recommander/fold/model.pkl','rb'))
books_name=pickle.load(open('C:/Users/moham/OneDrive/Bureau/data science/book_recommander/fold/books_name.pkl','rb'))
final_rating=pickle.load(open('C:/Users/moham/OneDrive/Bureau/data science/book_recommander/fold/final_rating.pkl','rb'))
book_pivot=pickle.load(open('C:/Users/moham/OneDrive/Bureau/data science/book_recommander/fold/book_pivot.pkl','rb'))

def fetch_poster(suggestion):
    book_name=[]
    ids_index=[]
    poster_url=[]
    for book_id in suggestion:
        book_name.append(book_pivot.index[book_id])
    for name in book_name[0]:
        ids=np.where(final_rating['title']==name)[0][0]
        ids_index.append(ids)
    for idx in ids_index:
        url=final_rating.iloc[idx]['img_url']
        poster_url.append(url)
    return poster_url


def recommended_book(book_name):
    book_list=[]
    book_id=np.where(book_pivot.index==book_name)[0][0]
    distance,suggestion=model.kneighbors(book_pivot.iloc[book_id,:].values.reshape(1,-1),n_neighbors=7)
    poster_url=fetch_poster(suggestion)
    for i in range(len(suggestion)):
        books=book_pivot.index[suggestion[i]]
        for j in books:
            book_list.append(j)
    return book_list,poster_url



selected_book=st.selectbox(
    'Type or select your book',books_name
)
if st.button('Show Recommendations'):
    recommendation_book,poster_url=recommended_book(selected_book)
    col1,col2,col3,col4,col5,col6=st.columns(6)
    with col1:
        st.text(recommendation_book[1])
        st.image(poster_url[1])
    with col2:
        st.text(recommendation_book[2])
        st.image(poster_url[2])
    with col3:
        st.text(recommendation_book[3])
        st.image(poster_url[3])
    with col4:
        st.text(recommendation_book[4])
        st.image(poster_url[4])
    with col5:
        st.text(recommendation_book[5])
        st.image(poster_url[5])
    with col6:
        st.text(recommendation_book[6])
        st.image(poster_url[6])



