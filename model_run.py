from keras.models import model_from_json, load_model, Sequential
import numpy as np
import cv2 as cv

bengin1 = '''A benign skin patch is a general term for any area of discolored or raised skin that isn't cancerous. There are many different types of benign skin patches, each with its own cause and appearance. Some of the most common benign skin patches include:

    Birthmarks, Moles, Seborrheic keratoses, Skin tags, Cherry angiomas, Freckles\n'''

benign2 = '''In most cases, benign skin patches do not require treatment. However, if a benign skin patch is bothering you or you are concerned about it, see a doctor. They can examine the patch and determine if it is benign.'''


malignant = '''**Warning: Potential Melanoma Detected**\n\n

This image analysis suggests the possibility of melanoma. Melanoma is a serious form of skin cancer, and early detection is crucial.  \n\nPlease see a doctor immediately for a professional diagnosis if you multiple scans suggest Malignant Melanoma.\n\n'''

malignant2 = '''Try taking several images of the same lesion from different angles and under good lighting. This can help improve the accuracy of the model.\n
If you continue to receive a melanoma warning after trying different images, a visit to the doctor is strongly recommended.\n
Additional symptoms like Asymmetry, blurred border, uneven colouration, change in shape of the patch makes it important that you approach a medical professional.\n
If you have any concerns about a mole or lesion, it is important to see a dermatologist for a skin exam. Early diagnosis and treatment of melanoma is essential for improving the chances of a cure.\n
Melanoma can occur anywhere on the body, but has a higher chance on fave, arms and legs.\n
Anyone can get melanoma, but people with fair skin, a history of sunburn, and a family history of melanoma are at higher risk.\n
The best way to prevent melanoma is to protect your skin from the sun. This includes wearing sunscreen with an SPF of 30 or higher, seeking shade during peak sun hours (10 am to 4 pm), and wearing protective clothing, such as hats and sunglasses.\n
Keep in mind that this is not a diagnosis, but we highly recommend that you seek professional help for a skin examination\n
'''

class CNN:
    def __init__(self):
        self.json = open("Melanoma.json", "r")
        self.file = self.json.read()
        self.model = model_from_json(self.file)
        self.json.close()

        self.model.load_weights("Melanoma.h5")

    def predict(self, img):
        self.image = cv.resize(img, (112, 112), interpolation=cv.INTER_LINEAR)

        img = np.expand_dims(self.image, axis=0)

        self.prediction = self.model.predict(img/255)
        if self.prediction == 0.:
            return "Model fault", "Error"
        if self.prediction > 0.5:
            return "Malignant Melanoma", malignant, malignant2
        else:
            return "Benign Skin Lesion", bengin1, benign2

