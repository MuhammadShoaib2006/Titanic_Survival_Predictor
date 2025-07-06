from flask import Flask, request, jsonify
import pandas as pd
import pickle

app = Flask(__name__)

# Load the model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        data = request.get_json()
        
        # Convert to DataFrame with correct feature order
        input_data = pd.DataFrame([{
            'Pclass': data['Pclass'],
            'Sex': data['Sex'],
            'Age': data['Age'],
            'SibSp': data['SibSp'],
            'Parch': data['Parch'],
            'Fare': data['Fare'],
            'Embarked': data['Embarked']
        }])
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        
        return jsonify({
            'status': 'success',
            'prediction': int(prediction),
            'message': '1 = Survived, 0 = Did not survive'
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)