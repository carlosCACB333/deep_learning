import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)


@app.route('/predict', methods=['GET'])
def home() -> str:
    X = np.array([7.594444821, 7.479555538, 1.616463184, 1.53352356,
                 0.796666503, 0.635422587, 0.362012237, 0.315963835, 2.277026653])
    X = X.reshape(1, -1)
    prediction = model.predict(X)
    return jsonify({'prediction': prediction.tolist()})


if __name__ == '__main__':
    model = joblib.load('models/model.pkl')
    app.run(port=8080)
