
# Face Mask Detection Using Deep Learning and OpenCV

## Overview

This project implements a deep learning model using **Keras** and **OpenCV** to detect whether a person is wearing a face mask in real-time. The model is trained on a dataset of images of people with and without masks. The system can be extended to implement face mask detection through live video feed from a webcam.

## Model Architecture

- **Conv2D Layers**: Three convolutional layers with ReLU activation for feature extraction.
- **MaxPooling2D**: Used after each convolutional layer to reduce spatial dimensions.
- **Dense Layers**: A fully connected layer followed by an output layer with a sigmoid activation function for binary classification (mask/no mask).

## Dataset

The dataset is structured into two categories:
1. **With Mask**: Images of people wearing masks.
2. **Without Mask**: Images of people not wearing masks.

Data augmentation is applied to improve the model's generalization.

## Key Steps

1. **Data Augmentation**:
   - `ImageDataGenerator` is used to rescale, shear, zoom, and horizontally flip the training images.
   
2. **Training**:
   - The model is trained on images with binary cross-entropy as the loss function and `adam` as the optimizer.

3. **Prediction**:
   - The model predicts whether a person is wearing a mask or not by analyzing images from a folder or real-time video feed using OpenCV.

## Code Explanation

### Model Training

```python
# Model creation and compilation
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(100, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

### Data Augmentation and Model Training

```python
# Data Augmentation
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

# Loading dataset from directories
training_set = train_datagen.flow_from_directory('/path/to/train', target_size=(150, 150), batch_size=16, class_mode='binary')
test_set = test_datagen.flow_from_directory('/path/to/test', target_size=(150, 150), batch_size=16, class_mode='binary')

# Training the model
model.fit(training_set, epochs=10, validation_data=test_set)
```

### Saving and Loading the Model

```python
# Save the trained model
model.save('mymodel.h5')

# Load the model
mymodel = load_model('mymodel.h5')
```

### Image Prediction

```python
# Predicting mask or no mask from an image
test_image = image.load_img('/path/to/image.jpg', target_size=(150, 150, 3))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = mymodel.predict(test_image)[0][0]
```

### Real-Time Mask Detection Using Webcam

```python
# Live detection using webcam
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while cap.isOpened():
    _, img = cap.read()
    faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=4)

    for (x, y, w, h) in faces:
        face_img = img[y:y+h, x:x+w]
        cv2.imwrite('temp.jpg', face_img)
        
        test_image = image.load_img('temp.jpg', target_size=(150, 150, 3))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        
        pred = mymodel.predict(test_image)[0][0]
        label = "MASK" if pred == 0 else "NO MASK"
        color = (0, 255, 0) if pred == 0 else (0, 0, 255)
        
        cv2.rectangle(img, (x, y), (x+w, y+h), color, 3)
        cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow('Mask Detection', img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## Requirements

- **Python 3.x**
- **TensorFlow/Keras**
- **OpenCV**
- **NumPy**

### Install the dependencies:
```bash
pip install -r requirements.txt
```

## How to Use

1. Clone the repository:

```bash
git clone https://github.com/yourusername/face-mask-detection.git
```

2. Train the model or load the pre-trained model:

```python
mymodel = load_model('mymodel.h5')
```

3. Run the real-time detection script:

```bash
python mask_detection.py
```

## Results

- **Accuracy**: ~98% on the validation set.
- Real-time mask detection with high accuracy.

## Future Improvements

- Add additional face mask types for broader coverage.
- Deploy the model on mobile or edge devices for real-time applications.


