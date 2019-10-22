import numpy as np
import cv2
path='C:/Users/sanghoon/AssIstoon/'
img = cv2.imread(path+'e125_p102.jpg')
mask = cv2.imread(path+'white_masked.jpg', 0)


dst = cv2.inpaint(img, mask, 3,cv2.INPAINT_TELEA)
                  #cv2.INPAINT_TELEA)

cv2.imshow('dst', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()