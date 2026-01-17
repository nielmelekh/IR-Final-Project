from flask import Flask, request, jsonify
from backend import *

app = Flask(__name__)
backend = Backend_Search()

@app.route("/search")
def search():
    query = request.args.get('query', '')
    if not query:
        return jsonify([])
    return jsonify(backend.search_body_bm25(query))

@app.route("/search_body")
def search_body():
    query = request.args.get('query', '')
    if not query:
        return jsonify([])
    return jsonify(backend.search_body(query))

@app.route("/search_title")
def search_title():
    query = request.args.get('query', '')
    if not query:
        return jsonify([])
    return jsonify(backend.search_title(query))

@app.route("/search_anchor")
def search_anchor():
    query = request.args.get('query', '')
    if not query:
        return jsonify([])
    return jsonify(backend.search_anchor(query))

@app.route("/get_pageview", methods=['POST'])
def get_pageview():
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
      return jsonify(res)
    res = backend.get_pageviews(wiki_ids)
    return jsonify(res)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
