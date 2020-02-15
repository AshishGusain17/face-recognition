# python count-vector.py -d caffe_model -m openface_nn4.small2.v1.t7  -e output/vector_embeddings.pickle

from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse, imutils, pickle, time, cv2, os

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--detector",           required=True,help="path to OpenCV's deep learning face detector")
ap.add_argument("-m", "--embedding-model",    required=True,help="path to OpenCV's deep learning face embedding model")
ap.add_argument("-e", "--embeddings",         required=True,help="path to serialized db of facial embeddings")
ap.add_argument("-c", "--confidence", type=float, default=0.5,help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# load our serialized face detector from disk
print("Loading face detector...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"], "res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# load our serialized face embedding model from disk
print("Loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch(args["embedding_model"])

# initialize the video stream, then allow the camera sensor to warm up
print("Starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# start the FPS throughput estimator
fps = FPS().start()


print("Loading face embeddings from other file...")
data = pickle.loads(open(args["embeddings"], "rb").read())

names=["ashish",'shirley']
persons=len(names)
flag=[0]*persons
for ii in range(persons):
	ans=data['names'].count(names[ii])
	flag[ii]=ans
print(flag)
print(names)
print(data['names'])

embeddings=data['embeddings']

while True:
	# break
	frame = vs.read()

	# resize the frame to have a width of 600 pixels (while maintaining the aspect ratio), and then grab the image dimensions
	frame = imutils.resize(frame, width=600)
	(h, w) = frame.shape[:2]

	# construct a blob from the image
	imageBlob = cv2.dnn.blobFromImage(
		cv2.resize(frame, (300, 300)), 1.0, (300, 300),
		(104.0, 177.0, 123.0), swapRB=False, crop=False)

	# detect all faces in the image frame
	detector.setInput(imageBlob)
	detections = detector.forward()

	for i in range(0, detections.shape[2]):
		confidence = detections[0, 0, i, 2]

		if confidence > args["confidence"]:

			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# extract the face ROI
			face = frame[startY:endY, startX:endX]
			(fH, fW) = face.shape[:2]

			# ensure the face width and height are sufficiently large
			if fW < 20 or fH < 20:
				continue

			# construct a blob for the face ROI
			faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
			embedder.setInput(faceBlob)
			vec = embedder.forward()


			ch=[0]*len(embeddings)
			for ind,ll in enumerate(embeddings):
				sub=np.sum((np.array(vec)-np.array(ll))**2)**(1/2)
				ch[ind]=sub
				print(sub)

			go=0
			preds=[0]*persons
			for ind,ii in enumerate(flag):
				ans=ch[go:go+ii]
				preds[ind]=np.sum(ans)/ii
				print(ans,ind,ii)
				go=ii
			print(preds)

	# break



			# classifying the face
			j = np.argmin(preds)
			proba = preds[j]
			name = names[j]

			text = "{}: {:.2f}%".format(name, proba * 100)
			y = startY - 10 if startY - 10 > 10 else startY + 10
			cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
			cv2.putText(frame, text, (startX, y),  cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

	# update the FPS counter
	fps.update()

	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	if key == ord("q"):
		break

# stop the timer and display FPS information
fps.stop()
print("Total time elapsed: {:.2f}".format(fps.elapsed()))
print("Aprox. FPS: {:.2f}".format(fps.fps()))

cv2.destroyAllWindows()
vs.stop()