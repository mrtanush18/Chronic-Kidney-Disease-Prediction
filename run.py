from flask import Flask, render_template, request, jsonify
import prediction
from flask import Flask, render_template, flash, request, redirect,url_for
from wtforms import Form, TextAreaField, validators, StringField, SubmitField,widgets, SelectMultipleField
from flask_wtf import FlaskForm


app = Flask(__name__)


@app.route('/')
def index():
    return render_template("home.html")

@app.route('/home')
def home():
    return render_template("index.html")

@app.route('/contact')
def contact():
    return render_template("contact.html")

@app.route('/about')
def about():
    return render_template("about.html")


@app.route('/', methods=['POST'])
def predict():
    response = None
    if request.method == 'POST' :
        try:
            bp = request.form['bp'] 
            sg = request.form['sg'] 
            a = request.form['a'] 
            pcc = request.form['pcc'] 
            bgr = request.form['bgr'] 
            sc = request.form['sc'] 
            pcv = request.form['pcv'] 
            wbcc = request.form['wbcc'] 
            dm = request.form['dm'] 
            ap = request.form['ap'] 

            list1=[]
            list1.append(bp)
            list1.append(sg)
            list1.append(a)
            list1.append(pcc)
            list1.append(bgr)
            list1.append(sc)
            list1.append(pcv)
            list1.append(wbcc)
            list1.append(dm)
            list1.append(ap)

            global new_response
            response = prediction.predict(list1)
            print(response)
            message=response
        except Exception as e:
            return respond(e)
    return render_template('home.html', message=message)

@app.route('/display/<filename>')
def display_image(filename):
	print('display_image filename: ' + filename)
	return redirect(url_for('static', filename='stactic/' + filename), code=301)


def respond(err, res=None):
    return_res =  {
        'status_code': 400 if err else 200,
        'body': err.message if err else res,
    }
    return jsonify(return_res)

# start the server with the 'run()' method
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9000)