
# IR Final Project: Wikipedia Search Engine

## Overview
Welcome to our Information Retrieval project. We have developed a search engine for the English Wikipedia corpus. Our system allows users to perform free text queries and retrieves relevant results using classical IR techniques, including inverted indexes and ranking algorithms.

The entire system is hosted on Google Cloud Platform (GCP). We designed it so that all index data is stored in our storage bucket and accessed directly, meaning no external services are required when a query is processed.

## Architecture and Components
The project is split into two main layers to ensure a clean separation of concerns:

### 1. Backend
The backend handles the core logic of the search engine.
* **Preprocessing:** We implemented a standard pipeline including tokenization, stopword removal, lowercasing, and Porter stemming.
* **Retrieval & Ranking:** The backend accesses the inverted indexes to fetch candidates. After experimentation, we selected BM25 as our final ranking method for the body index because it offered superior retrieval quality and speed compared to TF IDF.
* **Search Logic:** We use binary search for the title and anchor indexes, while the body index uses the weighted BM25 model.

### 2. Frontend
The frontend is a Flask based web application that exposes our engine to the world. It provides several endpoints to access different search functionalities:
* **/search:** The main search engine, uses BM25 ranking over page bodies. Showed the best ratio of accuracy to speed
* **/search_body:** tf-idf weighted cosine similarity search and ranking over page bodies.
* **/search_title:** Binary search and ranking over page titles.
* **/search_anchor:** Binary search and ranking over anchor texts.
* **/get_pageview:** Lists the page views of the pages whose id was provided in the json payload during august of 2021.

## Data Storage and Indexes
All our data is stored in the 'ir_project_2025' bucket on Google Cloud Storage. We organized the data into specific directories to keep the structure clean:

* **Body Index:** Stored in the 'body100' directory. This supports our main search.
* **Title Index:** Stored in the 'title1' directory.
* **Anchor Index:** Stored in the 'anchor4' directory.
* **Page Views:** We store a the page views during august of 2021 in the 'pageviews_aug_2021' directory.
* **Auxiliary Maps:** We also store a dictionary mapping document IDs to titles in the 'id_to_title_dict' directory to display results correctly.

## Authors
* Noa Segev
* Niel Melekh
* Shaked Farjun
