from flask import Flask, request, send_file , Response
import os
import sys


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__)) 
if "\\" in CURRENT_DIR:
    ROOT_DIR = "\\".join(CURRENT_DIR.split("\\")[:-1])
else:
    ROOT_DIR = "/".join(CURRENT_DIR.split("/")[:-1])                     

MODEL_DIR = os.path.join(ROOT_DIR, 'model') 

sys.path.append(MODEL_DIR)
from s2fgenerator.model import Generator
import tempfile
import cv2
from helper.helper_func import predict_one_img, plot_one_gen_image
from flask_cors import CORS, cross_origin
from enhancer.gfpgan import GFPGAN

modelPath = os.path.join(*[ROOT_DIR, 'model', 'model_saved', 'generator_weight.h5'])
gfpgan = GFPGAN(os.path.join(*[ROOT_DIR, 'model', 'model_saved', 'GFPGANv1.3.pth']))
predictOnePath= os.path.join(*[ROOT_DIR, 'api', 'firstImg.png'])
predictTwoPath= os.path.join(*[ROOT_DIR, 'api', 'secondImg.png'])

app = Flask(__name__)

cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
generator = Generator().load_model(modelPath)

@app.route('/')
def hello_world():
    return 'Hello, World'

@app.route("/image", methods=["POST"])
@cross_origin()
def process_image():
    file = request.files["image"]
    
    # smt = request.form['name']
    ofile, ofname = tempfile.mkstemp()
    file.save(ofname)
    # Read the image via file.stream
    # print(cv2.imdecode(numpy.fromstring(file.read(), numpy.uint8), cv2.IMREAD_UNCHANGED).shape())
    predict_one_img(generator, cv2.imread(ofname,0), predictOnePath)
    os.close(ofile)
    os.remove(ofname)
    gen_image = cv2.imread(predictOnePath, cv2.IMREAD_COLOR)
    # print(smt)
    # return send_file(predictOnePath, mimetype='image/png')
    gfpgan.enhance(gen_image, predictTwoPath)
    return send_file(predictTwoPath, mimetype='image/png')



@app.route('/<int:id>')
def get_img(id):
    img = Img.query.filter_by(id=id).first()
    if not img:
        return 'Img Not Found!', 404

    return Response(img.img, mimetype=img.mimetype)


if __name__ == "__main__":
    # app.run(debug=True)
    app.run(host="0.0.0.0", debug=True)