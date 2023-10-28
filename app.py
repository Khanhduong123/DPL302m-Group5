from flask import Flask, render_template
# from flask_sqlalchemy import SQLAlchemy
# import datetime

app = Flask(__file__, template_folder="./template", static_folder="./static")
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///retail_database.db"

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
@app.route("/", methods = ['GET'])
def root():
    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)