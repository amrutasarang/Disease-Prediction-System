
import flask
import joblib
import pymysql
from sqlalchemy import true
pymysql.install_as_MySQLdb()
from flask import Flask, render_template, redirect, url_for, request, flash
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField, SubmitField
from wtforms.validators import InputRequired, Email, Length, DataRequired
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user
from flask_sqlalchemy import SQLAlchemy
from flask_mail import Mail
from flask_mail import Message
from itsdangerous import TimedJSONWebSignatureSerializer as Serializer
import pandas
import pickle
import numpy as np

from email import message
from urllib import response
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from chat import get_response


app = Flask(__name__)
CORS(app)
filename = 'cancer.pkl'
log = pickle.load(open(filename, 'rb'))
model = pickle.load(open('cancer.pkl', 'rb'))
model1 = pickle.load(open('model1.pkl','rb'))
model2 = pickle.load(open('diabetes.pkl','rb'))

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret'
# app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:@localhost/prediction'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
bootstrap = Bootstrap(app)
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(15), unique=True)
    email = db.Column(db.String(50), unique=True)
    password = db.Column(db.String(80))

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

class LoginForm(FlaskForm):
    username = StringField('Username', validators=[InputRequired(), Length(min=4, max=15)])
    password = PasswordField('Password', validators=[InputRequired(), Length(min=8, max=80)])
    remember = BooleanField('remember me')


class RegisterForm(FlaskForm):
    email = StringField('Email', validators=[InputRequired(), Email(message='Invalid email'), Length(max=50)])
    username = StringField('Username', validators=[InputRequired(), Length(min=4, max=15)])
    password = PasswordField('Password', validators=[InputRequired(), Length(min=8, max=80)])

class ResetRequestForm(FlaskForm):
    username = StringField('Username', validators=[InputRequired(), Length(min=4, max=15)])
    submit = SubmitField(label='Reset Password', validators=[DataRequired()])

@app.route('/')
def index():
    return render_template("index.html")

@app.post("/predict")
def predict():
    text = request.get_json().get("message")
    # TODO: check if text is valid
    response = get_response(text)
    message = {"answer": response}
    return jsonify(message)

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/contact')
def contact():
    return render_template("contact.html")


@app.route('/help')
def help():
    return render_template("help.html")


@app.route('/terms')
def terms():
    return render_template("tc.html")

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user:
            if check_password_hash(user.password, form.password.data):
                login_user(user)
                return redirect(url_for('dashboard'))
            else:
                return render_template("login.html", form=form)
        return render_template("login.html", form=form)
    return render_template("login.html", form=form)


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    form = RegisterForm()
    if form.validate_on_submit():
        hashed_password = generate_password_hash(form.password.data, method='sha256')
        new_user = User(username=form.username.data, email=form.email.data, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()

        return redirect("/login")
    return render_template('signup.html', form=form)

@app.route('/reset_password', methods=['GET', 'POST'])
def reset_request():
    form = ResetRequestForm()
    if form.validate_on_submit():
        flash('Reset request sent. Check your mail.','success')
    return render_template('reset_request.html', title='Reset Request', form=form)


@app.route("/dashboard")
@login_required
def dashboard():
    return render_template("dashboard.html")

@app.route("/disindex")
@login_required
def disindex():
    return render_template("disindex.html")

@app.route("/cancer")
@login_required
def cancer():
    return render_template("cancer.html")

@app.route("/diabetes")
@login_required
def diabetes():
    return render_template("diabetes.html")

@app.route("/heart")
@login_required
def heart():
    return render_template("heart.html")

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

# Cancer
@app.route('/predictt', methods=['POST','GET'])
def predictt():
    if request.method == 'POST':
        tm = request.form['texture_mean']
        am = request.form['area_mean']
        cm = request.form['concavity_mean']
        ase = request.form['area_se']
        cse = request.form['concavity_se']
        fde = request.form['fde']
        sw = request.form['sw']
        cw = request.form['cw']
        syw = request.form['symmetry_w']
        fdw = request.form['fdw']

        data = np.array([[tm,am,cm,ase,cse,fde,sw,cw,syw,fdw]])
        output = log.predict(data)

        return render_template('cancer_result.html', prediction=output)

# Heart
@app.route('/heart',methods=['GET','POST'])
def home():
     if request.method=='POST':
        age=request.form['age']
        sex=request.form['sex']
        cp=request.form['cp']
        trestbps=request.form['trestbps']
        chol=request.form['chol']
        fbs=request.form['fbs']
        restecg=request.form['restecg']
        thalach=request.form['thalach']
        exang=request.form['exang']
        old=request.form['old']
        slope=request.form['slope']
        ca=request.form['ca']
        thal=request.form['thal']

        if(sex=='Male'):
            g=1
        else:
            g=0

        prediction=model1.predict([[age,g,cp,trestbps,chol,fbs,restecg,thalach,exang,old,slope,ca,thal]])
    #     if(prediction==0):
    #         prediction="No"
    #     else:
    #         prediction="Yes"

    #     return render_template("heart.html",prediction_txt="Prediction is {}".format(prediction))
            
    #  else:
    #     return render_template("heart.html")
        return render_template('heart_result.html', prediction=prediction)

# Diabetes
@app.route('/diabetes',methods=['GET','POST'])
def home1():
     if request.method=='POST':
        
        pre=request.form['pre']
        glu=request.form['glu']
        bp=request.form['bp']
        skin=request.form['skin']
        insu=request.form['insu']
        bmi=request.form['bmi']
        dpf=request.form['dpf']
        age=request.form['age']

        prediction=model2.predict([[pre,glu,bp,skin,insu,bmi,dpf,age]])
    #     if(prediction==0):
    #         prediction="No"
    #     else:
    #         prediction="Yes"

    #     return render_template("diabetes.html",prediction_txt="Prediction is {}".format(prediction))
            
    #  else:
    #     return render_template("diabetes.html")
        return render_template('diab_result.html', prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)