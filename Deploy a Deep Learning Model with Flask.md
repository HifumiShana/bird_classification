# Deploy a Deep Learning Model with Flask
Trong bài viết này nhóm tôi sẽ hướng dẫn cách tạo một app 
để dự đoán đây là con chim nào với Flask và mô hình CNN
## Installation
Đầu tiên hãy install những thư viện cần thiết
    - tensorflow
    - numpy
    - flask
    - flask_uploads 

## Predict a bird 
Trước hết, bạn cần xây dựng mô hình Deep Learning model. Ở đây, nhóm tôi sử dụng mô hình Google Colab để train mô hình
### B1: import các thư viện
```python=
import tensorflow as tf 
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import *
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import Model
from tensorflow.keras.applications.resnet50 import preprocess_input 
np.random.seed(42)
import matplotlib.pyplot as plt
```

### B2: kết nối với drive của bạn và giải nén file
```python=
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
!unzip "/content/drive/MyDrive/Project/bird.zip"
```
### B3: Lưu dữ liệu vào 3 tập train, valid, test
```python=
from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(shear_range=0.1,
                                    zoom_range=0.1,
                                    horizontal_flip=True)  

valid_datagen = ImageDataGenerator()
test_datagen = ImageDataGenerator()
train_generator = train_datagen.flow_from_directory('/content/train',
                                              color_mode='rgb',
                                              target_size=(224,224),
                                              batch_size=1024,
                                              save_format='jpg',
                                              class_mode='categorical',
                                              shuffle=True,
                                              subset='training',
                                              seed=42)


validation_generator = valid_datagen.flow_from_directory('/content/valid',
                                              color_mode='rgb',
                                              target_size=(224,224),
                                              batch_size=1024,
                                              save_format='jpg',
                                              class_mode='categorical',
                                              shuffle=True,
                                              subset='training',
                                              seed=42)
test_generator = test_datagen.flow_from_directory('/content/test',
                                              color_mode='rgb',
                                              target_size=(224,224),
                                              batch_size=32,
                                              save_format='jpg',
                                              class_mode='categorical',
                                              shuffle=True,
                                              subset='training',
                                              seed=42)
```
### B4: Xây dựng và huấn luyện mô hình 
```python=
earlystopping =  EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model_checkpoint_callback = ModelCheckpoint(filepath='/content/drive/MyDrive/Project/weights_resnet_adamw_cyclical.hdf5',
                                            save_weights_only=True,
                                            monitor='val_accuracy',
                                            mode='max',
                                            save_best_only=True)
callbacks = [model_checkpoint_callback,earlystopping]
base_model = ResNet50(input_shape=(224,224,3),
                        include_top=False,
                        weights='imagenet')
base_model.trainable = False
base_model.summary()
global_average_layer = GlobalAveragePooling2D()
prediction_layer = Dense(260, activation='softmax')
inputs = Input(shape=(224, 224, 3))
x = preprocess_input(inputs)
x = base_model(x, training=False)
x = global_average_layer(x)
outputs = prediction_layer(x)
model = Model(inputs, outputs)
model.summary()
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
history = model.fit(train_generator, validation_data=validation_generator, callbacks=callbacks,epochs=1000)
```
```python=
# Fine-tuning
base_model.trainable = True
print(len(base_model.layers))
for layer in base_model.layers[:100]:
  layer.trainable = False
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_generator, validation_data=validation_generator, callbacks=[checkpoint_acc, stop2, lr], epochs=100)

```
### B5: Lưu lại mô hình, hàm tf.keras.models.load_model tải mô hình mà ta đã train trong google colab 
```python=
model.save('/content/drive/MyDrive/Project/model_resnet50.hdf5')
```
## Classify images with a Flask
Cấu trúc thư mục:
- Bird Classification 
  - static
    - img 
  - model.py
  - upload.py
  - templates
    - index.html
  - weights
    - my_model.h5
 
Trước khi xây dựng app bạn cần tạo ra 2 hàm chứa trong model.py 
```python=
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications.resnet50 import decode_predictions
import numpy as np
class_names = ['AFRICAN CROWNED CRANE', 'AFRICAN FIREFINCH', 'ALBATROSS', 'ALEXANDRINE PARAKEET', 'AMERICAN AVOCET', 'AMERICAN BITTERN', 'AMERICAN COOT', 'AMERICAN GOLDFINCH', 'AMERICAN KESTREL', 'AMERICAN PIPIT', 'AMERICAN REDSTART', 'ANHINGA', 'ANNAS HUMMINGBIRD', 'ANTBIRD', 'ARARIPE MANAKIN', 'ASIAN CRESTED IBIS', 'BALD EAGLE', 'BALI STARLING', 'BALTIMORE ORIOLE', 'BANANAQUIT', 'BANDED BROADBILL', 'BAR-TAILED GODWIT', 'BARN OWL', 'BARN SWALLOW', 'BARRED PUFFBIRD', 'BAY-BREASTED WARBLER', 'BEARDED BARBET', 'BELTED KINGFISHER', 'BIRD OF PARADISE', 'BLACK FRANCOLIN', 'BLACK SKIMMER', 'BLACK SWAN', 'BLACK TAIL CRAKE', 'BLACK THROATED WARBLER', 'BLACK VULTURE', 'BLACK-CAPPED CHICKADEE', 'BLACK-NECKED GREBE', 'BLACK-THROATED SPARROW', 'BLACKBURNIAM WARBLER', 'BLUE GROUSE', 'BLUE HERON', 'BOBOLINK', 'BROWN NOODY', 'BROWN THRASHER', 'CACTUS WREN', 'CALIFORNIA CONDOR', 'CALIFORNIA GULL', 'CALIFORNIA QUAIL', 'CANARY', 'CAPE MAY WARBLER', 'CAPUCHINBIRD', 'CARMINE BEE-EATER', 'CASPIAN TERN', 'CASSOWARY', 'CHARA DE COLLAR', 'CHIPPING SPARROW', 'CHUKAR PARTRIDGE', 'CINNAMON TEAL', 'COCK OF THE  ROCK', 'COCKATOO', 'COMMON FIRECREST', 'COMMON GRACKLE', 'COMMON HOUSE MARTIN', 'COMMON LOON', 'COMMON POORWILL', 'COMMON STARLING', 'COUCHS KINGBIRD', 'CRESTED AUKLET', 'CRESTED CARACARA', 'CRESTED NUTHATCH', 'CROW', 'CROWNED PIGEON', 'CUBAN TODY', 'CURL CRESTED ARACURI', 'D-ARNAUDS BARBET', 'DARK EYED JUNCO', 'DOWNY WOODPECKER', 'EASTERN BLUEBIRD', 'EASTERN MEADOWLARK', 'EASTERN ROSELLA', 'EASTERN TOWEE', 'ELEGANT TROGON', 'ELLIOTS  PHEASANT', 'EMPEROR PENGUIN', 'EMU', 'ENGGANO MYNA', 'EURASIAN GOLDEN ORIOLE', 'EURASIAN MAGPIE', 'EVENING GROSBEAK', 'FIRE TAILLED MYZORNIS', 'FLAME TANAGER', 'FLAMINGO', 'FRIGATE', 'GAMBELS QUAIL', 'GANG GANG COCKATOO', 'GILA WOODPECKER', 'GILDED FLICKER', 'GLOSSY IBIS', 'GO AWAY BIRD', 'GOLD WING WARBLER', 'GOLDEN CHEEKED WARBLER', 'GOLDEN CHLOROPHONIA', 'GOLDEN EAGLE', 'GOLDEN PHEASANT', 'GOLDEN PIPIT', 'GOULDIAN FINCH', 'GRAY CATBIRD', 'GRAY PARTRIDGE', 'GREAT POTOO', 'GREATOR SAGE GROUSE', 'GREEN JAY', 'GREY PLOVER', 'GUINEA TURACO', 'GUINEAFOWL', 'GYRFALCON', 'HARPY EAGLE', 'HAWAIIAN GOOSE', 'HELMET VANGA', 'HIMALAYAN MONAL', 'HOATZIN', 'HOODED MERGANSER', 'HOOPOES', 'HORNBILL', 'HORNED GUAN', 'HORNED SUNGEM', 'HOUSE FINCH', 'HOUSE SPARROW', 'IMPERIAL SHAQ', 'INCA TERN', 'INDIAN BUSTARD',
         'INDIAN PITTA', 'INDIGO BUNTING', 'JABIRU', 'JAVA SPARROW', 'JAVAN MAGPIE', 'KAKAPO', 'KILLDEAR', 'KING VULTURE', 'KIWI', 'KOOKABURRA', 'LARK BUNTING', 'LEARS MACAW', 'LILAC ROLLER', 'LONG-EARED OWL', 'MAGPIE GOOSE', 'MALABAR HORNBILL', 'MALACHITE KINGFISHER', 'MALEO', 'MALLARD DUCK', 'MANDRIN DUCK', 'MARABOU STORK', 'MASKED BOOBY', 'MASKED LAPWING', 'MIKADO  PHEASANT', 'MOURNING DOVE', 'MYNA', 'NICOBAR PIGEON', 'NOISY FRIARBIRD', 'NORTHERN BALD IBIS', 'NORTHERN CARDINAL', 'NORTHERN FLICKER', 'NORTHERN GANNET', 'NORTHERN GOSHAWK', 'NORTHERN JACANA', 'NORTHERN MOCKINGBIRD', 'NORTHERN PARULA', 'NORTHERN RED BISHOP', 'NORTHERN SHOVELER', 'OCELLATED TURKEY', 'OKINAWA RAIL', 'OSPREY', 'OSTRICH', 'OYSTER CATCHER', 'PAINTED BUNTIG', 'PALILA', 'PARADISE TANAGER', 'PARUS MAJ![](https://i.imgur.com/WSECzKf.png)
OR', 'PEACOCK', 'PELICAN', 'PEREGRINE FALCON', 'PHILIPPINE EAGLE', 'PINK ROBIN', 'PUFFIN', 'PURPLE FINCH', 'PURPLE GALLINULE', 'PURPLE MARTIN', 'PURPLE SWAMPHEN', 'QUETZAL', 'RAINBOW LORIKEET', 'RAZORBILL', 'RED BEARDED BEE EATER', 'RED BELLIED PITTA', 'RED FACED CORMORANT', 'RED FACED WARBLER', 'RED HEADED DUCK', 'RED HEADED WOODPECKER', 'RED HONEY CREEPER', 'RED TAILED THRUSH', 'RED WINGED BLACKBIRD', 'RED WISKERED BULBUL', 'REGENT BOWERBIRD', 'RING-NECKED PHEASANT', 'ROADRUNNER', 'ROBIN', 'ROCK DOVE', 'ROSY FACED LOVEBIRD', 'ROUGH LEG BUZZARD', 'RUBY THROATED HUMMINGBIRD', 'RUFOUS KINGFISHER', 'RUFUOS MOTMOT', 'SAMATRAN THRUSH', 'SAND MARTIN', 'SCARLET IBIS', 'SCARLET MACAW', 'SHOEBILL', 'SHORT BILLED DOWITCHER', 'SMITHS LONGSPUR', 'SNOWY EGRET', 'SNOWY OWL', 'SORA', 'SPANGLED COTINGA', 'SPLENDID WREN', 'SPOON BILED SANDPIPER', 'SPOONBILL', 'SRI LANKA BLUE MAGPIE', 'STEAMER DUCK', 'STORK BILLED KINGFISHER', 'STRAWBERRY FINCH', 'STRIPPED SWALLOW', 'SUPERB STARLING', 'SWINHOES PHEASANT', 'TAIWAN MAGPIE', 'TAKAHE', 'TASMANIAN HEN', 'TEAL DUCK', 'TIT MOUSE', 'TOUCHAN', 'TOWNSENDS WARBLER', 'TREE SWALLOW', 'TRUMPTER SWAN', 'TURKEY VULTURE', 'TURQUOISE MOTMOT', 'UMBRELLA BIRD', 'VARIED THRUSH', 'VENEZUELIAN TROUPIAL', 'VERMILION FLYCATHER', 'VICTORIA CROWNED PIGEON', 'VIOLET GREEN SWALLOW', 'VULTURINE GUINEAFOWL', 'WATTLED CURASSOW', 'WHIMBREL', 'WHITE CHEEKED TURACO', 'WHITE NECKED RAVEN', 'WHITE TAILED TROPIC', 'WILD TURKEY', 'WILSONS BIRD OF PARADISE', 'WOOD DUCK', 'YELLOW BELLIED FLOWERPECKER', 'YELLOW CACIQUE', 'YELLOW HEADED BLACKBIRD']
model = tf.keras.models.load_model('model_resnet50.h5', compile=False)


def process_image(image):
    '''
    Make an image ready-to-use by VGG19
    '''
    # convert the image pixels to a numpy array
    img_array= img_to_array(image)
    # reshape data for the model
    # image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2])
    
    # prepare the image for the VGG model
    img_array= preprocess_input(img_array)
    img_array =tf.expand_dims(img_array,axis=0)
    return img_array


def predict_class(image):
    '''
    Predict and render the class of a given image 
    '''
    

    predictions = model.predict(image)
    score = tf.nn.softmax(predictions[0])
    # print(class_names[np.argmax(score)])
    return class_names[np.argmax(score)], 100*np.max(score)
```
Sau đó tạo file upload.py
```python=
from flask import Flask, render_template, request
# if you encounter dependency issues using 'pip install flask-uploads'
# try 'pip install Flask-Reuploaded'
from flask_uploads import UploadSet, configure_uploads, IMAGES
from keras.preprocessing.image import load_img
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
    if request.method == 'POST' and 'image' in request.files:
        # save the image
        filename = photos.save(request.files['image'])
        # load the image
        image = load_img('static/img/'+filename, target_size=(224, 224))
        # process the image
        image = process_image(image)
        # make prediction
        prediction, percentage = predict_class(image)
        # the answer which will be rendered back to the user
        
        return render_template('index.html',answer = "The prediction is : {} With probability = {}".format(prediction, percentage))
    # web page to show before the POST request containing the image
    # return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True) 
```
Tạo file index.html
```htmlembedded=
<body>

    <div class='container'>

        <form method=POST enctype=multipart/form-data action="{{ url_for('upload') }}" class="form">
            <div class="intro">
                <h1>Welcome to Project Bird Classification</h1>
            </div>
            <div class='file-upload-wrapper' data-text="Select your file!">
                <input type="file" class="file-upload-field" accept="image/*" name="file-upload-field" id="file"
                    onchange="loadFile(event)" value="">
            </div>


            <div class="submit"> <input type="submit" value="Done" id="submit"></div>
            <div class="answer">
                <p>{{answer}}</p>
            </div>
        </form>
    </div>
    <script src="script.js"></script>
</body>

```
## Conclusion
Nhờ flask, nhóm chúng tôi có thể tạo một python web đơn giản cho deep learning model. App này là sự kết hợp của google colab, javascript, html, css và python.
### Trước khi Done
Chạy file upload.py để xem kết quả
![](https://i.imgur.com/IC05AUO.png)
### Hình ảnh được chọn 
![](https://i.imgur.com/iAp9ovh.png)

### Sau khi Done
![](https://i.imgur.com/nUDVQzf.png)


## Tham khảo
Bài viết này được tham khảo từ nhiều nguồn. Nguồn tham khảo chình nằm từ [đây](https://towardsdatascience.com/machine-learning-in-production-keras-flask-docker-and-heroku-933b5f885459)
