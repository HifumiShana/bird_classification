# Deploy a Deep Learning Model with Streamlit
Trong bài viết này nhóm tôi sẽ hướng dẫn cách tạo một app 
để dự đoán đây là con chim nào với Streamlit và mô hình CNN
## Installation
Đầu tiên hãy install những thư viện cần thiết
    - tensorflow
    - numpy
    - streamlit
## Xây dựng mô hình
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
train_datagen = ImageDataGenerator(shear_range=0.1, zoom_range=0.1,                       horizontal_flip=True)  

valid_datagen = ImageDataGenerator()
test_datagen = ImageDataGenerator()
train_generator = train_datagen.flow_from_directory('/content/train',   color_mode='rgb', target_size=(224,224), batch_size=128, save_format='jpg', class_mode='categorical', shuffle=True, subset='training',seed=42)


validation_generator = valid_datagen.flow_from_directory('/content/valid', color_mode='rgb', target_size=(224,224), batch_size=128, save_format='jpg', class_mode='categorical', shuffle=True, subset='training', seed=42)
test_generator = test_datagen.flow_from_directory('/content/test', mode='rgb', target_size=(224,224), batch_size=128,  save_format='jpg', class_mode='categorical', shuffle=True, subset='training', seed=42)
```
### B4: Xây dựng và huấn luyện mô hình 
Ở đây ta sử dụng kỹ thuật Transfer Learning và Fine-tuning với mô hình Resnet50 đã được train từ trước.

```python=
# Xây dựng mô hình
stop1 =  EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)
stop2 =  EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=False)
model_checkpoint= ModelCheckpoint(filepath='/content/drive/MyDrive/Project/weights_resnet_adam.hdf5', save_weights_only=True, monitor='val_accuracy', mode='max', save_best_only=True)
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
# Train những layer cuối
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
## Xây dựng app đơn giản bằng streamlit

Trước khi tiếp tục, các bạn hãy tải và giải nén bộ dữ liệu trên [tại đây](https://www.kaggle.com/gpiosenka/100-bird-species).
Lưu ý: Bộ dữ liệu này có thể đã được cập nhật thêm tại thời điểm bạn đọc bài viết này, vì thể hãy tinh chỉnh một chút cho phù hợp nhé!
### B1: Download mô hình về máy
Ta down tập tin "model_resnet50.hdf5" vể máy để sử dụng. Sau đó trên máy tính, mở VS Code và thực hiện xây dựng mô hình lại như trên. Các bạn có thể sử dụng công cụ khác như PyCharm, Sublime Text, ...
```python=
import streamlit as st
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import Model
from tensorflow.keras.layers import *
from tensorflow.keras.applications.resnet50 import preprocess_input

base_model = ResNet50(input_shape=(224,224,3),
                        include_top=False,
                        weights='imagenet')
base_model.trainable = True
global_average_layer = GlobalAveragePooling2D()
prediction_layer = Dense(260, activation='softmax')
inputs = Input(shape=(224, 224, 3))
x = preprocess_input(inputs)
x = base_model(x, training=False)
x = global_average_layer(x)
outputs = prediction_layer(x)
model = Model(inputs, outputs)
base_model.trainable = True
for layer in base_model.layers[:100]:
  layer.trainable = False
model.load_weights('D:\\AI\\Birds\\weights_resnet_acc.hdf5')
```

### B2: Sử dụng thư viện Streamlit xây dựng app
Tôi sẽ gọi tập tin sau đây là $ui.py$
```python=
import streamlit as st
import numpy as np
from load_model import model
import os
import matplotlib.image as mpimg
from tensorflow.keras.applications.resnet50 import preprocess_input 

model.load_weights('D:\\AI\\Project\\weights_resnet_acc.hdf5')

st.title('BIRDS CLASSIFICATION APP')
st.write('260 Species of Birds:')
birds_species = os.listdir('D:\\AI\\Project\\archived\\consolidated')
birds_species = sorted(birds_species)
diction = dict()
for i in range(10):
    diction['collumn '+str(i)] = list()
for i in range(10):
    for j in range(26):
        diction['collumn '+str(i)].append(birds_species[(26*i+j)])
st.write(pd.DataFrame(diction))

opt = st.selectbox(options=birds_species, label='Choose a species to see')
img = mpimg.imread(f'D:\\AI\\Project\\archived\\test\\{opt}\\1.jpg') 
st.image(img)

uploaded_img = st.file_uploader(label="Choose a image of a bird you want to know the names")
if uploaded_img != None:
    img = mpimg.imread(uploaded_img)
    img = preprocess_input(img)
    img = img.reshape(1,224,224,3)
    output = model.predict(img)
    output = birds_species[output.argmax()]
st.write('*The image shows the: *') 
if uploaded_img != None:
    st.write(output)
```
Sau khi code xong, bạn hãy mở Command Prompt nhập dòng lệnh "streamlit run ..." với dấu "..." là đường dẫn của tập tin $ui.py$ trên máy tính của bạn.

![Imgur](https://i.imgur.com/Q7pYaDj.png)

Từ đó, Streamlit sẽ load một Web app đơn giản trên trình duyệt của bạn

![Imgur](https://i.imgur.com/c4RYK1J.png)

![Imgur](https://i.imgur.com/6JUn876.png)

Các bạn sẽ thấy 3 phần:
- Phần bảng liệt kê 260 loài chim mà mô hình đã được train
- Các bạn có thể chọn ở Select Box phía dưới để xem hình ảnh mẫu của từng loài.
- Phần còn lại là một ô chứa để cái bạn có thể duyệt tập tin hình ảnh và xem được kết quả của mô hình.
