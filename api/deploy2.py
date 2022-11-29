from flask import Flask, request, send_file
import sys
sys.path.append('D:\work\s2f_model_senior_project\model')
from s2fgenerator.model import Generator
import tempfile
import cv2
from helper.helper_func import predict_one_img, plot_one_gen_image
from flask_cors import CORS, cross_origin
from enhancer.gfpgan import GFPGAN


modelPath = 'D:\work\s2f_model_senior_project\model\model_saved\generator_weight.h5'
gfpgan = GFPGAN('D:\work\s2f_model_senior_project\model\model_saved\GFPGANv1.3.pth')
predictOnePath='D:\work\s2f_model_senior_project\\api\\firstImg.png'
predictTwoPath='D:\work\s2f_model_senior_project\\api\\secondImg.png'
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
generator = Generator().load_model(modelPath)

@app.route("/image", methods=["POST"])
@cross_origin()
def process_image():
    file = request.files["image"]
    ofile, ofname = tempfile.mkstemp()
    file.save(ofname)
    # Read the image via file.stream
    predict_one_img(generator, cv2.imread(ofname,0), predictOnePath)
    gen_image = cv2.imread(predictOnePath, cv2.IMREAD_COLOR)
    gfpgan.enhance(gen_image, predictTwoPath)
    
    # return "ok"
    return send_file(predictTwoPath, mimetype='image/png')


if __name__ == "__main__":
    app.run(debug=True)