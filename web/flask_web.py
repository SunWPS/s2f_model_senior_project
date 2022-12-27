import requests
from flask import Flask, render_template , request ,send_file
from database.db import db , db_init
from database.models import Record
import datetime
import tempfile
import cv2
app = Flask(__name__)
db_init(app)
api_url = 'http://127.0.0.1:5000/image'

@app.route('/')
def home():
    return render_template('template.html')

@app.route('/formHandling', methods=['POST' , 'GET'])
def formHandling():
    firstImg = request.files["image"]
    data={'image': firstImg.read()}
    response= requests.post(api_url,files=data)
    if(response.status_code == 200):
        secImg = response.content
        with open("static/temp.png", 'wb') as f:
            f.write(secImg)    
        return send_file("D:\work\s2f_model_senior_project\web\static\\temp.png", mimetype='image/png')
        # return render_template('template.html', sec_img="/static/temp.png")
    return 'data not sent!',404

@app.route('/save', methods=['POST'])
def save():
    content = request.json
    time = datetime.datetime.now()
    data = Record(
        id=content['id'],
        createDate = time,
        updateDate = time,
        originalImg =content['originalImg'],
        genImg = content['genImg']
    )
    db.session.add(data)
    db.session.commit()

    return 'data save!', 200
    
if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0', port=8000)