from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import pickle

app = Flask(__name__)
CORS(app)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        required_fields = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
        if not all(field in data for field in required_fields):
            return jsonify({
                'status': 'error',
                'message': 'Missing required fields'
            }), 400
        
        input_data = pd.DataFrame([{
            'Pclass': int(data['Pclass']),
            'Sex': int(data['Sex']),
            'Age': float(data['Age']),
            'SibSp': int(data['SibSp']),
            'Parch': int(data['Parch']),
            'Fare': float(data['Fare']),
            'Embarked': int(data['Embarked'])
        }])
        
        prediction = model.predict(input_data)[0]
        
        return jsonify({
            'status': 'success',
            'prediction': int(prediction),
            'message': '1 = Survived, 0 = Did not survive'
        })
    
    except ValueError as e:
        return jsonify({
            'status': 'error',
            'message': f'Invalid data format: {str(e)}'
        }), 400
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Prediction failed: {str(e)}'
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
