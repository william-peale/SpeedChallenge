import cv2
import numpy as np

vidcap = cv2.VideoCapture('data/train.mp4')
sped = open("data/train.txt", "r")
sped_data = sped.read().split("\n")
print(len(sped_data))
success,image_old = vidcap.read()
count = 0
success = True
max_entries = len(sped_data)-2
image_size = 224
x_first = np.zeros((max_entries,image_size,image_size,3),dtype=np.uint8)
x_second = np.zeros((max_entries,image_size,image_size,3),dtype=np.uint8)
y = np.zeros((max_entries,1))
while count < max_entries and success:
  success,image_new = vidcap.read()
  fact = np.random.uniform()
  resized_image_new = cv2.resize(image_new[120:-120], (image_size,image_size))
  resized_image_old = cv2.resize(image_old[120:-120], (image_size,image_size))
  x_first[count] = resized_image_old
  x_second[count] = resized_image_new
  y[count] = sped_data[count+1]
  if cv2.waitKey(10) == 27:                     # exit if Escape is hit
      break
  image_old = image_new
  print(count/len(sped_data))
  print(count)
  count += 1


np.save("x_first.npy",x_first)
np.save("x_second.npy",x_second)
np.save("y.npy",y)
