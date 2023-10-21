from flask import Flask, render_template, request
import joblib
app = Flask(__name__)

knn_model = joblib.load('models/knn_model.pkl')
pate_model = joblib.load('models/pate_model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        user_input = []
        for feature in ['fLength', 'fWidth', 'fSize', 'fConc', 'fConc1', 'fAsym', 'fM3Long', 'fM3Trans', 'fAlpha', 'fDist']:
            user_input.append(float(request.form[feature]))

        knn_prediction = knn_model.predict([user_input])
        
        pate_prediction = pate_model.predict([user_input])
        if pate_prediction==1:
            pate_prediction='g'
        else:
            pate_prediction='h'

        return render_template('index.html', knn_prediction=knn_prediction[0], pate_prediction=pate_prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
