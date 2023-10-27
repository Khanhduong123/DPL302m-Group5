from flask import Flask, request, jsonify, render_template
import tensorflow as tf

app = Flask(__name__, template_folder='./templates')

# model = tf.keras.models.load_model('chatbot.h5')

@app.route('/chatbot', methods=['GET'])
def chat():
    print("Done")
    # Preprocess input text
    # ...

    # Get model prediction
    # prediction = model.predict([input_text])
    
    # Postprocess prediction
    # ...
    
    # response = {'prediction': prediction}
    # return jsonify(response)
    return render_template("index.html")

@app.route('/post', methods=['POST'])
def post():
    data = request.json
    print(data['message'])
    return data

if __name__ == '__main__':
    app.run(debug=True)
    