<!-- Adding a GIF after main title -->
<p align="center">
  <img src="https://raw.githubusercontent.com/shraddhavijay/IFAKE/master/screenshots/text.gif">
</p>

<h1 align="center">IFAKE - Image/Video Forgery Detection Application</h1>

This repository contains two main folders:

1. **IFAKE_AI** - This folder contains the AI Jupyter notebook files used to create the proposed CNN model for forgery detection and classification. The notebook files demonstrate the process of training and testing the model on the FIDAC & CASIA dataset.

2. **IFAKE_WebApp** - This folder contains the web application project. The web application is built on the Django framework and provides a user-friendly interface for detecting image and video forgeries.

## Research Paper and Dataset

Our research paper titled "[Image Forgery Detection and Classification Using Deep Learning and FIDAC Dataset](https://ieeexplore.ieee.org/document/9862034)" is published on IEEE Explore. In this paper, we propose our model that uses CNN for classification after being fed with ELA preprocessed images to detect image forgery, and we also introduce our created dataset - FIDAC (Forged Images Detection And Classification), which consists of original cameraclicked images along with their tampered version. Furthermore, we conducted an experimental analysis wherein we compared our proposed CNN model with famous pre-defined models on various datasets combinations.

The [FIDAC dataset](https://ieee-dataport.org/documents/fidac-forged-images-detection-and-classification) is available on IEEE Dataport and contains original camera-clicked images along with their tampered versions. The dataset was used to train and test our proposed CNN model and compare it with other pre-defined models on various datasets combinations.



## Pre-trained Models

We provide links to download our pre-trained models for image & video forgery detection and classification:

- [Image Model weigths](https://drive.google.com/drive/folders/1B4ODeK_QQ6XMFo6i6EEup1nZC6PllVfu?usp=sharing)
- [Video Model weigths](https://drive.google.com/drive/folders/1irYZbRnr4Y7jKieSyhjxHxwk43oSMqh-?usp=sharing)

## Running the Web Application

To run the web application on Windows, Linux, or Mac, follow these steps:

1. Install Python3 and pip3
2. Clone this repository
3. Open a terminal and navigate to the IFAKE_WebApp folder
4. Run the following command to install the required Python packages:

    ```
    pip3 install -r requirements.txt
    ```

5. Run the following command to start the web application:

    ```
    python3 manage.py runserver
    ```

6. Open a web browser and go to http://127.0.0.1:8000/ to access the web application.

## Screenshots
<img src="https://raw.githubusercontent.com/shraddhavijay/IFAKE/master/screenshots/index.JPG" alt="Image description" width="60%">
<img src="https://raw.githubusercontent.com/shraddhavijay/IFAKE/master/screenshots/imageDetection1.png" alt="Image description" width="60%">
<img src="https://raw.githubusercontent.com/shraddhavijay/IFAKE/master/screenshots/imageDetection2.png" alt="Image description" width="60%">
<img src="https://raw.githubusercontent.com/shraddhavijay/IFAKE/master/screenshots/metadata.JPG" alt="Image description" width="60%">
<img src="https://raw.githubusercontent.com/shraddhavijay/IFAKE/master/screenshots/videoDetection.png" alt="Image description" width="60%">
<img src="https://raw.githubusercontent.com/shraddhavijay/IFAKE/master/screenshots/pdfDetection.png" alt="Image description" width="60%">




The screenshots show different features of our web application, including the image and video forgery detection functionality, and the ability to upload and view results of detected forgeries.

## Contributors
- Shraddha Pawar
- Gaurangi Pradhan
- Bhavin Goswami



## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
