import pandas as pd
from flask import Flask, request, render_template, redirect
from datetime import datetime

from annotator.clusters import Clusters
from annotator.knn import NNGenerator
from annotator.writer import CSVWriter

DATASET_NAME = "elections"
RECORDS_PATH = f"/app/annotator/static/data/{DATASET_NAME}_records.csv"

app = Flask(
    __name__,
    template_folder="/app/annotator/templates",
    static_folder="/app/annotator/static"
)

csv_writer = CSVWriter(RECORDS_PATH)
clusters = Clusters(RECORDS_PATH)

already_known = set(clusters.df.d1.to_list() if clusters.df is not None else [])
neighbors = NNGenerator.from_csv(DATASET_NAME, already_known)


@app.route('/', methods=['GET'])
def index():
    return redirect(f"/{neighbors.random()}/{0}/{neighbors.random()}/{0}")


@app.route('/<string:docid1>/<int:page1>/<string:docid2>/<int:page2>', methods=['GET', 'POST'])
def pair_get(docid1, page1, docid2, page2):
    if request.method == 'GET':
        relationship = clusters.get_relationship(docid1, docid2)
        return render_template(
            'index.html',
            docid1=docid1,
            page1=page1,
            docid2=docid2,
            page2=page2,
            relationship=relationship,
            experiment_name=DATASET_NAME,
            n_docs_annotated=csv_writer.n_lines()
        )
    elif request.method == 'POST':
        if request.form.get("same"):
            relationship = "same"
        elif request.form.get("different"):
            relationship = "different"
        elif request.form.get("interesting"):
            relationship = "interesting"
        elif request.form.get("reject"):
            relationship = "reject"
        elif request.form.get("next"):
            relationship = "next"
        else:
            raise ValueError("Parameter same not found", request.form)

        if relationship in ["same", "different", "interesting", "reject"]:
            clusters.set_relationship(docid1, docid2, relationship)
            csv_writer.write([str(datetime.now()), docid1, docid2, relationship])
            print(f"Database contains {clusters.n_clusters()} non-trivial clusters.")
        elif relationship == "next":
            df = pd.read_csv(RECORDS_PATH, sep=";", names=["timestamp", "d1", "d2", "rel"])
            already_known = set(df.d1.to_list())
            docid1 = neighbors.random(already_known)

        friend = neighbors.knn(clusters, docid1)
        return redirect(f"/{docid1}/{0}/{friend}/{0}")


if __name__ == '__main__':
    app.run(port=5005, host='0.0.0.0')
