from flask import Flask
from flask_restful import Resource, Api, reqparse
from werkzeug.datastructures import FileStorage
from s2fgenerator.model import Generator
import tempfile
import cv2
from helper.helper_func import predict_one_img, plot_one_gen_image

app = Flask(__name__)
app.logger.setLevel('INFO')

api = Api(app)

parser = reqparse.RequestParser()
parser.add_argument('file',
                    type=FileStorage,
                    location='files',
                    required=True,
                    help='provide a file')

##load model
generator = Generator().load_model('D:\work\s2f_model_senior_project\model\model_saved\generator_weight.h5')

class Image(Resource):

    def post(self):
        args = parser.parse_args()
        the_file = args['file']
        # save a temporary copy of the file
        ofile, ofname = tempfile.mkstemp()
        the_file.save(ofname)
        predict_one_img(generator, cv2.imread(ofname,0), 'D:\work\s2f_model_senior_project\model\img_output\\1_generated\\fromDeploy.png')
        return "worked"

api.add_resource(Image, '/image')

if __name__ == '__main__':
    app.run(debug=True)