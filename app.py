from flask import Flask, render_template,request,jsonify
import sqlite3
from price_search import *
# from flask_sqlalchemy import SQLAlchemy
# import datetime

app = Flask(__file__, template_folder="./template", static_folder="./static")
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///instance/giasanpham.db"

# db = SQLAlchemy(app)
# class Users(db.Model):
#     __tablename__ = "Users"
#     id_ = db.Column(db.Integer, primary_key=True)
#     name_ = db.Column(db.String(100))
#     email_ = db.Column(db.String(100))
#     date_ = db.Column(db.Date())
#     count_ = db.Column(db.Integer)
#     def __init__(self, email, name, date=datetime.datetime.now().date(), count=0):
#         self.name_ = name
#         self.email_ = email
#         self.date_ = date
#         self.count_ = count
#     def __repr__(self):
#         return '<User %r>' % self.email_


# with app.app_context():
#     db.create_all()
#     db.session.commit()
#@app.route("/", methods = ['GET'])
#def root():
#    return render_template("index.html")

@app.route('/')
def index():
    # Connect to the SQLite database
    conn = sqlite3.connect('instance/giasanpham.db')
    cursor = conn.cursor()

    # Execute the first SELECT query
    cursor.execute("SELECT * FROM thongtindienthoai WHERE phone_names LIKE '%iPhone 15 Pro Max%'")
    data_from_table1 = cursor.fetchall()
    
    cursor.execute("SELECT * FROM thongtindienthoai WHERE phone_names LIKE '%iPhone 15 Pro 128GB%'")
    data_from_table2 = cursor.fetchall()
    
    conn.close()
    
    return render_template('index.html', data1=data_from_table1, data2=data_from_table2)

@app.route('/', methods=['GET'])
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


@app.route('/', methods=['POST'])
def post():
    data = request.json
    user_message = data['message']
    print(user_message)  # Print user input to the server console
    # You can perform processing on user_message if needed
    # For example, you can pass it to a chatbot model for generating a response
    # For now, let's echo the user input back to the frontend
    result = price(user_message)
    # Construct the response data including the user input
    response_data = {
        'user_message': user_message,
        'bot_response': result  # You can replace this with the actual bot response
    }

    return jsonify(response_data)

if __name__ == '__main__':
    app.run(debug=True)