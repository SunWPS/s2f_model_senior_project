# **Facial Image Synthesis from Sketches based on Deep Learning Techniques**

(senior project)

- Wongsakorn Pinvasee
- Pachara Srisomboonchote

## **Abstract**
The system for synthesizing facial images from facial sketches using deep learning techniques has the objective of supporting the work of police investigators in tracking down suspects. The system can convert sketches or facial images of suspects into lifelike images to aid in identifying suspects and provide the public with a clearer image of the suspect. The lifelike images also benefit the public in providing information to the authorities. The system was developed using Generative Adversarial Networks (GAN) models to learn from a dataset of facial sketches and generate realistic facial images. The GAN model consists of two processes: image synthesis using Pix2pix and image refinement using GFPGAN. The system also includes a web application built using the Flask framework to call the image generation model, with Firebase Authentication to ensure secure access for users and Google Storage to store image data. From evaluating the learning efficiency of the model, it was found that the Structural Similarity Index Measure (SSIM) was 0.74.

## **Prepare Data**
To prepare data for model learning, real sketches and sketches created from OpenCV will be used as the dataset, as collecting real sketches resulted in only 188 images, which is a very small amount. The dataset used will be a combination of four datasets: CUHK (X. Wang & X. Tang, 2009), AFAD (Zhenxing et al., 2016), FFHQ (Tero et al., 2019), and CelebA-HQ (Lee et al., 2020), with a total of 55,000 pairs of both sketches and real images, as shown in image below. This is divided into 50,000 images for learning and 5,000 for testing. Before providing the data for the model to learn, the data must be prepared. The first step is to adjust the sketches to have similar image detail values, such as the intensity of black, threshold values, and image size, which in this case is 256x256 pixels. The next step is to normalize the color range of the preprocessed images to a new range of (-1, 1) from the normal range of (0, 255) to help the model learn the features and relationships in the data more effectively. For the real images, the size is adjusted and the color range is also transformed to (-1, 1), just like the sketches

![alt text](https://github.com/SunWPS/s2f_model_senior_project/blob/master/images/0.jpg?raw=true)

## **Model**
In this system, the Pix2pix model is used as the main method for image synthesis. From the image below, it can be seen that it consists of two models, namely the Generator and the Discriminator. However, only the Generator part will be used for web application. Additionally, GFPGAN is also used to further enhance the results obtained from Pix2pix, in order to increase the level of detail in the images.

![alt text](https://github.com/SunWPS/s2f_model_senior_project/blob/master/images/1.jpg?raw=true)

### **Generator**
The Generator component utilizes the U-Net architecture, which operates by taking input images with dimensions of 256x256 pixels, and produces output images with the same dimensions. The input and output layers have an input/output size of (256, 256, 3). The output layer employs the tanh activation function, while the hidden layers are divided into sub-blocks, namely the encoder and decoder. The encoder layers consist of Conv2D, BatchNormalization, and LeakyRelu, while the decoder layers consist of Conv2DTranspose, BatchNormalization, dropout, and relu. Furthermore, the decoder layers employ skip connections with the encoder layers, following the U-Net architecture. In summary, the entire Generator component consists of an input layer, 7 encoder blocks, 7 decoder blocks, and an output layer.

### **Decoder**
In the Discriminator part, it is a classification model for distinguishing between real and generated images. The input layer takes in two images and concatenates them, where input_1 is the sketch image and input_2 is either the real or generated image. This means that the Discriminator not only discriminates between generated and real images but also considers the consistency between the sketch image and the real/generated image. This is called a conditional model. The output layer uses sigmoid as the activation function, and the hidden layers in each sublayer consist of Conv2D, BatchNormalization, and LeakyReLU. Finally, the optimizer used is Adam, and the loss function is binary crossentropy.

### **GAN**
Once we have both the Generator and Discriminator, we combine them into a GAN model for training the Generator specifically. In this model, the Discriminator is set to be non-trainable so that it can be separately trained later. During GAN training, we train to update only the Generator while the Discriminator provides feedback and guidance on how to learn better.

The optimizer used in the GAN is Adam, and the loss function is a total loss, which is calculated from both pixel loss and contextual loss. We have tried using the L1 function, but found that the results from the Generator, which are then sent to the Discriminator for learning, resulted in the Discriminator having a loss value close to 0 or equal to 0, indicating that the model did not learn anything new. Therefore, we switched to using the total loss as mentioned above.

### **GFPGAN**
In this part, we will be adjusting the output from the Pix2pix model as shown in the figure below, in order to enhance the details of the image and make it sharper and clearer. This includes resizing the image. The model used for this task will utilize a pre-trained model from Tencent Applied Research Center call GFPGAN

![alt text](https://github.com/SunWPS/s2f_model_senior_project/blob/master/images/2.jpg?raw=true)

## **Result**
In this case, SSIM (Structural Similarity Index) will be used for testing. From testing with a dataset of 5,000 pairs of synthesized and real images, the result is 0.74, which indicates that the images generated by the model cannot produce background scenes that look similar to the real images. As a result, the SSIM value obtained is not very high.

![alt text](https://github.com/SunWPS/s2f_model_senior_project/blob/master/images/4.jpg?raw=true)

## **Web application**
https://github.com/Pachara2001/s2f_web_senior_project

![alt text](https://github.com/SunWPS/s2f_model_senior_project/blob/master/images/3.jpg?raw=true)

## **Summary**
The team responsible for achieving the objectives and scope of work has completed their work within the designated timeframe. The team has developed a model using Python and Tensorflow to learn data with deep learning techniques, GAN and Pix2pix architecture to generate images, and has also used GFPGAN to enhance the results from Pix2pix, making them more detailed. From the learning process with 50,000 pairs of sketches and real images, the model was tested and achieved an SSIM score of 0.74, which indicates a high degree of similarity between the real and generated images. Additionally, the model developed in this project can generate images at a certain level, but it may not be able to determine the race, skin color, or hair color of the generated image.

Furthermore, the team has developed a web application using Google App Engine, which is connected to the model created as an API in the Google Cloud Run system. The web application also stores data on Google Cloud SQL and Google Cloud Storage, and has authentication with Firebase Authentication.

## **References**
- [1] Yamashita, R., Nishio, M., Do, R.K.G. et al. (2018). Convolutional neural networks: an overview and application in radiology. https://doi.org/10.1007/s13244-018-0639-9.
- [2] Olaf, R., Philipp, F. et al. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. https://doi.org/10.1007/978-3-319-24574-4_28.
- [3] Shane, B., Rishi, S., (2018). A Note on the Inception Score. https://doi.org/10.48550/arXiv.1801.01973.
- [4] Zhou, W., Hamid, S., (2004). Image Quality Assessment: From Error Visibility to Structural Similarity. https://doi.org/10.1109/TIP.2003.819861.
- [5] Kusam, L., et al. (2019). Image-to-Image Translation Using Generative Adversarial Network. https://ieeexplore.ieee.org/document/8822195.
- [6] Shuai, Y., et al. (2020). Deep Plastic Surgery: Robust and Controllable Image Editing with Human-Drawn Sketches. https://arxiv.org/abs/2001.02890.
- [7] Yan, Y., et al. (2022). S2FGAN: Semantically Aware Interactive Sketch-to-Face Translation. [ออนไลน์] เข้าถึงได้จาก https://doi.org/10.48550/arXiv.2011.14785.
- [8] Shetty, M., Raghavendra, K., et al. (2022). TRANSFER LEARNING WITH PIX2PIX GAN FOR GENERATING REALISTIC PHOTOGRAPHS FROM VIEWED SKETCH ARTS. https://doi.org/10.35741/issn.0258-2724.57.4.17.
- [9] Xintao, W., et al. (2021). Towards Real-World Blind Face Restoration with Generative Facial Prior. https://doi.org/10.48550/arXiv.2101.04061.
- [10] Wang, X., Tang, X., (2009). Face Photo-Sketch Synthesis and Recognition. https://doi.org/10.1109/TPAMI.2008.222.
- [11] Zhenxing, N., et al. (2016). Ordinal Regression with a Multiple Output CNN for Age Estimation. https://doi.org/10.1109/CVPR.2016.532.
- [12] Tero, K., et al. (2019). A Style-Based Generator Architecture for Generative Adversarial Networks. https://doi.org/10.48550/arXiv.1812.04948.
- [13] Cheng-Han, L., et al. (2020). MaskGAN: Towards Diverse and Interactive Facial Image Manipulation. https://doi.org/10.1109/CVPR42600.2020.00559
