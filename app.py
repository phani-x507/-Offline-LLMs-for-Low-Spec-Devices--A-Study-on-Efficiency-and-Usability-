from flask import Flask, render_template, jsonify
from mlm_evaluation import evaluate_all_mlms
from clm_evaluation import evaluate_all_clms

app = Flask(__name__)

@app.route("/")
def index():
    name = "phani"
    return render_template("index.html",name = name)

@app.route('/clm')
def clm():

    return render_template('clm.html')

@app.route("/evaluate/mlm")
def evaluate_mlm():
    mlm_results = evaluate_all_mlms()
    return jsonify(mlm_results)

@app.route("/evaluate/clm")
def evaluate_clm():
    clm_results = evaluate_all_clms()
    return jsonify(clm_results)

if __name__ == "__main__":
    app.run(debug=True)
