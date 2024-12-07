
from flask import Flask, request, render_template
import os

from utils import *
# Load environment variables from .env file

app = Flask(__name__)
os.environ['PYTHONUNBUFFERED'] = '1'

@app.route('/')
def home():
    """
    Handles the GET request to the root URL ('/').
    """
    return render_template('index.html')


@app.route('/query', methods=['POST'])
def query():
    """
    Handles the POST request to '/query'. Extracts the query from the request,
    processes it through the search, concatenate, and generate functions,
    and returns the generated answer.
    """
    # get the data/query from streamlit app
    query = request.form.get('query')
    print("Received query: ", query)
    
    # Step 1: Search and scrape articles based on the query
    print("Step 1: searching articles")
    articles = search_articles(query)
    
    
    # fetch_article_content(articles)

    # Step 2: Concatenate content from the scraped articles
    print("Step 2: concatenating content")
    # concatenate_content(articles)
    # Step 3: Generate an answer using the LLM
    print("Step 3: generating answer")
    # generate_answer(1,2)
    # return the jsonified text back to streamlit
    return "something"

if __name__ == '__main__':
    app.run(host='localhost', port=5001, debug=True)



