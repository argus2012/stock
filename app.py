from flask import Flask, render_template

from src.business_logic.process_query import create_business_logic

from src.algo.product_model import predict_by_sector_model, update_sector_models, stock_predict


app = Flask(__name__)


@app.route('/', methods=['GET'])
def hello():
    # return f'Hello dear students, you should use an other route:!\nEX: get_stock_val/<ticker>\n'
    return render_template("index.html")


@app.route('/predict', methods=['GET'])
def predict():
    # Redirect to predict page
    return render_template("predict.html")


@app.route('/recommend', methods=['GET'])
def recommend():
    # Redirect to recommend page
    return render_template("recommend.html")


@app.route('/update', methods=['GET'])
def update():
    # Redirect to update page
    return render_template("update.html")


@app.route('/account', methods=['GET'])
def account():
    # Redirect to account page
    return render_template("account.html")


@app.route('/detail', methods=['GET'])
def detail():
    # Redirect to detail page
    return render_template("detail.html")


@app.route('/test/<ticker>', methods=['GET'])
def test(ticker):
    # final_predict, training_ba, test_ba, predict_ba = ticker_predict(ticker)
    final_predict, training_ba, test_ba, predict_ba = predict_by_sector_model(ticker)
    return f'Ticker Predict BA: {predict_ba} <br/>'


@app.route('/update_system', methods=['GET'])
def update_all_sectors():
    output = update_sector_models()
    return render_template('update.html', output=output)


@app.route('/get_stock_val/<ticker>', methods=['GET'])
def get_stock_value(ticker):
    bl = create_business_logic()
    prediction = bl.do_predictions_for(ticker)

    return f'{prediction}\n'


@app.route('/stock_predict/<ticker>', methods=['GET'])
def stock_predict_model(ticker):
    output = stock_predict(ticker)
    return render_template('predict.html', output=output, ticker=ticker)


@app.route('/get_stock_predict/<ticker>', methods=['GET'])
def get_stock_predict(ticker):
    output = stock_predict(ticker)
    return output


if __name__ == '__main__':
    # Used when running locally only. When deploying to Cloud Run,
    # a webserver process such as Gunicorn will serve the app.
    app.run(host='localhost', port=8080, debug=True)
