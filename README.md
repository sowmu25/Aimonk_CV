
The dataset is imbalanced, so it affects the accuracy. To handle this we need to do image agumentation using ImageDataGenerator from tensorflow preprocessing techniques, which can handle balancing the dataset by preprocessing the images(flipping, zooming etc). Attribute 1, Attribute 3 and Attribute 4 are highly skewed


Missing values, preprocessing methods  and model building are already handled in Attribute.ipynb file.



*****-----How to run this file------******
activate tensorflow environment---> conda create and activate tensorflow environment
Run the requirements.txt file  ---> pip install -r requirements.txt
Run the file                   ---> python app.py