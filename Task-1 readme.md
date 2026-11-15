Image Classification Using a Pretrained Model
A beginner-friendly project that uses a small pretrained model (MobileNetV2 from PyTorch /
torchvision) to classify a few sample images and display the predicted labels.
How to run (VS Code)
1. Create a folder, put image_classifier.py and requirements.txt in it, and open it in VS
Code (File > Open Folder).
2. Open the terminal (Ctrl + `) and install the libraries (needs internet):
pip install -r requirements.txt
3. Run:
python image_classifier.py
The first run downloads 4 small sample images (into sample_images/) and the model (about 14 MB).
After that it works from the saved copies.
Use your own images: copy any .jpg or .png files into the sample_images folder and run again.
What the program does
1. Downloads a tiny dataset of 4 sample images (dog, Labrador, cat, sunflower).
2. Loads the pretrained MobileNetV2 model.
3. Classifies each image and prints the top 3 predicted labels with percentages.
4. Shows a picture grid with the best prediction as each title and saves it as results.png.
Sample output (format)
Image: dog.jpg
1. Samoyed                   85.3%
2. Arctic fox                 4.2%
3. white wolf                 2.1%
(The exact labels and percentages depend on the images and the model.)
How the pretrained model works (in simple words)
A pretrained model is one that someone has already taught. MobileNetV2 was trained on ImageNet, a
collection of about 1.2 million photos sorted into 1000 categories such as dog breeds, vehicles, fruits
and tools. While learning, it adjusted millions of internal numbers (weights) until it could guess the
category of a photo correctly most of the time.
Inside, the image goes through many layers. The first layers pick up simple patterns like edges and
colours. The middle layers combine them into textures and shapes (fur, wheels, petals). The last layers
combine those into whole objects. At the end, the model gives one score for each of the 1000
categories, and a function called softmax turns these scores into percentages that add up to 100%.
The categories with the highest percentages are the predictions.
Why MobileNetV2? It is designed to be small and fast, so it runs on a normal laptop without a GPU. It
uses a lighter type of convolution that needs far fewer calculations than older models.
Why use a pretrained model? Training from scratch needs millions of images and many hours of
computing. Using a pretrained model gives good results immediately with just a few lines of code.
Limitations: the model only knows the 1000 ImageNet categories, so it can only answer with one of
them. If a picture shows something outside those categories, it will pick the closest match, which may
look wrong.
Files
• image_classifier.py - the complete program, with comments on every step
• requirements.txt - libraries to install