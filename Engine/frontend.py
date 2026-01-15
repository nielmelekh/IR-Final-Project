from flask import Flask, request, jsonify
from Backend import Backend_Search

# יצירת אפליקציית Flask
app = Flask(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

# יצירת backend
backend = Backend_Search()


@app.route("/search")
def search():
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)

    res = backend.search_body(query)
    return jsonify(res)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)
