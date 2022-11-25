from flask import Flask, request, send_file
from s2fgenerator.model import Generator
import tempfile
import cv2
from helper.helper_func import predict_one_img, plot_one_gen_image

outputFilePath='img_output\\1_generated\\fromDeploy2.png'

app = Flask(__name__)
generator = Generator().load_model('D:\work\s2f_model_senior_project\model\model_saved\generator_weight.h5')

@app.route("/image", methods=["POST"])
def process_image():
    file = request.files["image"]
    ofile, ofname = tempfile.mkstemp()
    file.save(ofname)
    # Read the image via file.stream
    predict_one_img(generator, cv2.imread(ofname,0), outputFilePath)
    return send_file(outputFilePath, mimetype='image/png')


if __name__ == "__main__":
    app.run(debug=True)