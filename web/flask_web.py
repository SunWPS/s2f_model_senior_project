import requests
from flask import Flask, render_template , request ,send_file
from database.db import db , db_init
from database.models import Record
import datetime
import json
import os
import boto3
import secrets
from dotenv import load_dotenv

app = Flask(__name__)
db_init(app)
api_url = 'http://127.0.0.1:5000/image'
oriImgPath="static\oriImg.jpg"
genImgPath='static\genImg.png'

load_dotenv()

ACCESS_KEY_ID = os.getenv('ACCESS_KEY')
SECRET_ACCESS_KEY = os.getenv('SECRET_ACCESS_KEY')
BUCKET_NAME = os.getenv('BUCKET_NAME')

session = boto3.Session(
    aws_access_key_id = ACCESS_KEY_ID,
    aws_secret_access_key = SECRET_ACCESS_KEY
)

s3 = session.client('s3')

file_name = 'test.jpg'

@app.route('/')
def home():
    return render_template('template.html')

@app.route('/formHandling', methods=['POST' , 'GET'])
def formHandling():
    firstImg = request.files["image"]
    firstImg.save(oriImgPath)
    
    data={'image': open(oriImgPath, 'rb') }
    try:
        response= requests.post(api_url,files=data)
        secImg = response.content
        with open(genImgPath, 'wb') as f:
            f.write(secImg)    
        return send_file(genImgPath, mimetype='image/png')
    except requests.exceptions.RequestException as e:
        return str(e),500

@app.route('/save', methods=['POST'])
def save():
    content = request.json
    time = datetime.datetime.now()
    data = Record(
        createDate = time,
        updateDate = time,
        originalImg =content['originalImg'],
        genImg = content['genImg']
    )
    db.session.add(data)
    db.session.commit()
    return 'data save!', 200

@app.route('/upload_img_to_cloud' , methods=['GET'])
def uploadImgToCloud():
    secretTok=secrets.token_hex(16)
    oriName="ori_"+secretTok+".jpg"
    genName="gen_"+secretTok+".png"
    try:
        with open(oriImgPath, 'rb') as f:
            s3.upload_fileobj(f, BUCKET_NAME, oriName)
        with open(genImgPath,'rb') as f:
            s3.upload_fileobj(f, BUCKET_NAME, genName)
        my_json_obj = json.dumps({"originalImg" : oriName,"genImg" : genName})
        headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
        response = requests.post("http://192.168.1.5:8000/save",data=my_json_obj , headers=headers)
        return response.text,200
    except:
        return "something wrong",500
    
if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0', port=8000)