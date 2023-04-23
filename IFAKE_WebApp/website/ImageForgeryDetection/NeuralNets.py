from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, UpSampling2D, MaxPooling2D, BatchNormalization

def initClassifier():
    model = Sequential()
    model.add(Conv2D(filters = 16, kernel_size = (3,3),padding = 'same', 
                    activation ='relu', input_shape = (256,256,3)))
    model.add(BatchNormalization())
    model.add(Conv2D(filters = 16, kernel_size = (3,3),padding = 'same', 
                    activation ='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'same', 
                    activation ='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'same', 
                    activation ='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'same', 
                    activation ='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'same', 
                    activation ='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(filters = 128, kernel_size = (3,3),padding = 'same', 
                    activation ='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(filters = 128, kernel_size = (3,3),padding = 'same', 
                    activation ='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(512, activation = "relu"))
    model.add(Dropout(0.50))
    model.add(Dense(2, activation = "softmax"))

    return model

def initSegmenter():
    model = Sequential()

    model.add(Conv2D(filters = 16, kernel_size = (3,3),padding = 'same', 
                    activation ='relu', input_shape = (256,256,1)))
    model.add(BatchNormalization())
    model.add(Conv2D(filters = 16, kernel_size = (3,3),padding = 'same', 
                    activation ='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'same', 
                    activation ='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'same', 
                    activation ='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'same', 
                    activation ='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'same', 
                    activation ='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(filters = 128, kernel_size = (3,3),padding = 'same', 
                    activation ='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(filters = 128, kernel_size = (3,3),padding = 'same', 
                    activation ='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(UpSampling2D(size = (2,2)))
    model.add(Conv2D(filters = 128, kernel_size = (3,3),padding = 'same', 
                    activation ='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(filters = 128, kernel_size = (3,3),padding = 'same', 
                    activation ='relu'))
    model.add(BatchNormalization())

    model.add(UpSampling2D(size = (2,2)))
    model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'same', 
                    activation ='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'same', 
                    activation ='relu'))
    model.add(BatchNormalization())

    model.add(UpSampling2D(size = (2,2)))
    model.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'same', 
                    activation ='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'same', 
                    activation ='relu'))
    model.add(BatchNormalization())

    model.add(UpSampling2D(size = (2,2)))
    model.add(Conv2D(filters = 16, kernel_size = (3,3),padding = 'same', 
                    activation ='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(filters = 16, kernel_size = (3,3),padding = 'same', 
                    activation ='relu'))
    model.add(BatchNormalization())

    model.add(Conv2D(filters = 1, kernel_size = (1,1), activation = 'sigmoid'))

    return model