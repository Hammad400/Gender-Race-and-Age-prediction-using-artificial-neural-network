from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img
import numpy as np

app = Flask(__name__)



model = load_model('cnn_model_for_gender_race_and_age.h5')
gender_dict = {0:'Male', 1:'Female'}
race_dict={0:'Black', 1:'White',2:'Asian',3:'Indian',4:'Others (like Hispanic, Latino, Middle Eastern)'}
model.make_predict_function()

def predict_label(img_path):
	img = load_img(img_path, grayscale=True)
	img = img.resize((128, 128), Image.ANTIALIAS)
	img = np.array(img)
	img=img.reshape(128,128,1)
	img=img/255.0
	pred = model.predict(img.reshape(1,128, 128, 1))
	pred_gender = gender_dict[round(pred[0][0][0])]
	pred_race = race_dict[round(pred[1][0][0])]
	pred_age = round(pred[2][0][0])
	list_=[pred_gender,pred_race,pred_age]
	return list_


# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")



@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "static/" + img.filename	
		img.save(img_path)

		p = predict_label(img_path)

	return render_template("index.html", gender = p[0],race= p[1],age = p[2], img_path = img_path)


if __name__ =='__main__':
	#app.debug = True
	app.run(debug = True)