from flask import Flask, render_template, request
# if you encounter dependency issues using 'pip install flask-uploads'
# try 'pip install Flask-Reuploaded'
from flask_uploads import UploadSet, configure_uploads, IMAGES
from tensorflow.keras.preprocessing.image import load_img
# the pretrained model
from model import process_image, predict_class

app = Flask(__name__)

photos = UploadSet('photos', IMAGES)

# path for saving uploaded images
app.config['UPLOADED_PHOTOS_DEST'] = 'static/img'
configure_uploads(app, photos)

# professionals have standards :p


@app.route('/')
def home():
    return render_template('index.html')

# the main route for upload and prediction


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST' and 'file-upload-field' in request.files:
        # save the image
        filename = photos.save(request.files['file-upload-field'])
        # load the image
        image = load_img('static/img/'+filename, target_size=(224, 224))
        # process the image
        image = process_image(image)
        # make prediction
        prediction, percentage = predict_class(image)
        # the answer which will be rendered back to the user

        return render_template('index.html', answer="The prediction is : {} with probability = {}".format(prediction, percentage))
    # web page to show before the POST request containing the image
    # return render_template('index.html')



