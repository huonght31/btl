import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from tensorflow.keras.models import Sequential,load_model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout

data = []
labels = []
classes = 5 #số lương thư mục ảnh
cur_path = os.getcwd()

# Truy xuất hình ảnh và nhãn của chúng
for i in range(classes):
    path = os.path.join(cur_path, 'traindt', str(i))
    images = os.listdir(path)
    for a in images:
        try:
            image = Image.open(path + '\\' + a)
            image = image.resize((240,240))
            image = np.array(image)
            data.append(image)
            labels.append(i)
        except:
            print("Load ảnh lỗi")

# Chuyển đổi danh sách thành mảng trống
data = np.array(data)
labels = np.array(labels)

print(data.shape, labels.shape)
# Tách tập dữ liệu đào tạo và thử nghiệm
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

print(data.shape, labels.shape)


y_train = to_categorical(y_train, 5)
y_test = to_categorical(y_test, 5)

# Xây dựng mô hình
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu', input_shape=X_train.shape[1:]))
model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(5, activation='softmax'))

# Tổng hợp mô hình
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train, y_train,
      epochs=10,
      batch_size=32,
      validation_data=(X_test, y_test))

model.save("flowerclassi.h5")

plt.figure(0)
plt.plot(history.history['accuracy'], label='training accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()


from sklearn.metrics import accuracy_score

y_test = pd.read_csv('test.csv')

labels = y_test["ClassId"].values
imgs = y_test["path"].values

data = []

for img in imgs:
    image = Image.open(img)
    image = image.resize((224,224))
    data.append(np.array(image))

X_test = np.array(data)
pred =np.argmax(model.predict(X_test), axis=1)



from sklearn.metrics import accuracy_score

print(accuracy_score(labels, pred))
