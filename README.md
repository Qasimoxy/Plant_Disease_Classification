# Plant_Disease_Classification description
The project involves the classification model to detect disease plant leaves among four plants classs that includes Tomato, Cassava, Maize and Cashew.
The Cashew plant dataset includes anthracnose, gummosis, leaf miner, red rust and healthy leaves. 
The Cassava plant dataset includes bacterial blight, brown spot, green mite, mosaic and healthy leaves. 
The Maize dataset includes fall armyworm, grasshopper, leaf beetle, leaf blight, leaf spot, streak virus and health leaves. 
The Tomato plant dataset includes leaf blight, leaf curl, septoria leaf spot, verticillium wilt and healthy leaves.

# To Get the dataset
https://data.mendeley.com/datasets/bwh3zbpkpv/1
Go into the “Raw Data”directory and download the CCMT Dataset from the above link. Rename the CCMT
Folder to “DataPlant”. 

# To Get the Model 
https://drive.google.com/file/d/1zhuYaTJVzmqZ8YqGswB9w8nJC3SwzznD/view?usp=drive_link
To Download the Classification model (Plant_disease_detection_model.h5), follow the above link to get it from Google Drive.

# To Run the Python script and Classification model.
To run the python script, use command “python3 predict_disease_class.py image_name”, 
where image_name is the name of image from above link's dataset or from browser
Requirement: Image from browser should be from any above class's description.
Requirement: The image_name and the model (Plant_disease_detection_model.h5) should be in the directory as the python script (predict_disease_class.py).
