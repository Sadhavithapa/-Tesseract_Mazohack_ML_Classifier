from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image

app = Flask(__name__)
output_class = ["E-waste\n Yay!! we can recycle this product", "Sorry not classifed as E-waste", "E-waste\n Yay!! we can recycle this product", "Sorry not classifed as E-waste", "Sorry not classifed as E-waste", "Sorry not classifed as E-waste", "Sorry not classifed as E-waste", "Sorry not classifed as E-waste", "Sorry not classifed as E-waste", "Sorry not classifed as E-waste"]


import numpy as np
model = load_model('classifyWaste.h5')

model.make_predict_function()

def predict_label(img_path,model):
	img = image.load_img(img_path, target_size=(224, 224))	
	 # Preprocessing the image
	x = image.img_to_array(img) / 255
	 # x = np.true_divide(x, 255)
	x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
	predicted_array = model.predict(x)
	predicted_value = output_class[np.argmax(predicted_array)]
	return predicted_value


# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")

@app.route("/about")
def about_page():
	return "Please subscribe  Artificial Intelligence Hub..!!!"

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "static/" + img.filename	
		img.save(img_path)

		p = predict_label(img_path,model)

	return render_template("index.html", prediction = p, img_path = img_path)


if __name__ =='__main__':
	#app.debug = True
	app.run(debug = True)