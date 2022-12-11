
The dataset is imbalanced, so it affects the accuracy. To handle this we need to do image agumentation using ImageDataGenerator from tensorflow preprocessing techniques, which can handle balancing the dataset by flipping the images.But manuall work is need to annotate the text file according to the generated image. Label 1, label3 and label4 are highly skewed

image_973.jpg 0 1 0 0 , image_953.jpg 0 1 0 0

We can take the images like image_973.jpg, image_953.jpg( having 0 1 0 0, because attribute 1, attribute 3,attribute 4 having imbalanced '0' ) and do the image generator 


Missing values, preprocessing methods are already handled in the program itself.  