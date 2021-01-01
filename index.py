from flask import Flask, request,jsonify,send_file,send_from_directory
import os
from flask_cors import CORS
import matplotlib.pyplot as plt
import uuid
import cv2
import numpy as np
from tqdm import tqdm
import os

app = Flask(__name__)
CORS(app)

import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf

tfinter = tf.lite.Interpreter('mango.tflite')
tfinter.allocate_tensors()


@app.route('/getFile',methods=['POST'])
def getFile():
	file = request.files['picture']
	
	filename=str(uuid.uuid4())+".jpeg"
	if(not os.path.exists("input")):
		os.system("mkdir input")

	file.save('./input/'+filename)
	print(cv2.imread(file).shape)
	return jsonify({"status":1,"message":"Success","fileName":filename})

@app.route('/')
def home():
	return "Working API"


def pathed_image (image):
  patchs=[]
  h=15
  w=20
  for i in range(1,h+1):
      for j in range(1,w+1):
          img = image[ 200*i-200: 200 *i , 200*j-200 : 200*j]
          patchs.append(img)
  return np.array(patchs)

def predict(image):
  patch = pathed_image(image)
  h1=[]
  h=15
  pred=[]
  inp_index = tfinter.get_input_details()[0]['index']
  out_index = tfinter.get_output_details()[0]['index']
  
  for patch_img in tqdm(patch):
    img = np.array([patch_img],dtype='float32')
    tfinter.set_tensor(inp_index,img)
    tfinter.invoke()
    out = tfinter.get_tensor(out_index)
    pred.append(out[0])
  pred = np.array(pred)


  for i in tqdm(range(1,h+1)):
    img1 = np.hstack(pred[20*i-20 : 20*i])
    h1.append(img1)
    
  fimg = np.vstack(h1)
  #plt.imshow(fimg,cmap='gray')
  return fimg



@app.route('/api',methods=['POST'])
def demo():
	img_file_name = request.json['file']
	print(img_file_name)

	if(not os.path.exists("output")):
		os.system("mkdir output")
	if(not os.path.exists("preds")):
		os.system("mkdir preds")
	img = cv2.imread("input/"+img_file_name)
	img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
	print(img.shape)
	if img.shape[0]!=4000 and img.shape[1]!=3000:
		img = cv2.resize(img, (4000,3000), interpolation = cv2.INTER_AREA)
		print(img.shape)

	
	out = predict(img)
	
	plt.imsave("preds/"+img_file_name,out)
	outs = cv2.imread("preds/"+img_file_name,0)
	outs = cv2.threshold(outs,175,255,cv2.THRESH_BINARY)[1]
	con, hier = cv2.findContours(outs , cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	#cv2.drawContours(img,con,-1,(255,0,0),10)
	for c in con:
		x,y,w,h = cv2.boundingRect(c)
		cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),10)

	
	plt.imsave('output/'+img_file_name,img)
	output_name = request.base_url+'/output/'+img_file_name
	preds_name = request.base_url+'/preds/'+img_file_name


	return jsonify({'status':1,"message":"Success","output":output_name,"preds":preds_name,"count":str(len(con))})


@app.route('/api/output/<path:path>',methods=['GET'])
def getOutputFile(path):
	return send_from_directory('output',path)

@app.route('/api/preds/<path:path>',methods=['GET'])
def getPredsFile(path):
	return send_from_directory('preds',path)

if __name__ =='__main__':
	app.run(debug=False,host='0.0.0.0')
