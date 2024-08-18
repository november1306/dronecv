import cv2
from matplotlib import pyplot as plt

img_path = "../video/60m_return_frames/frame_0415.jpg"
# img_path = "../video/m_short_frames/frame_0000.jpg"
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# plt.figure(figsize=(10, 8))  # Optional: set the figure size
plt.imshow(img, cmap='gray')
plt.show()
