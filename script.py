import cv2
import numpy as np

data = {
    '1' : r'data\1\0.jpg',
    '2' : r'data\2\0.jpg',
    '3' : r'data\3\0.jpg',
    '4' : r'data\4\0.jpg',
    '5' : r'data\5\0.jpg',
    '6' : r'data\6\0.jpg',
    '7' : r'data\7\0.jpg',
    '8' : r'data\8\0.jpg',
    '9' : r'data\9\0.jpg',
    'A' : r'data\A\0.jpg',
    'B' : r'data\B\0.jpg',
    'C' : r'data\C\0.jpg',
    'D' : r'data\D\0.jpg',
    'E' : r'data\E\0.jpg',
    'F' : r'data\F\0.jpg',
    'G' : r'data\G\0.jpg',
    'H' : r'data\H\0.jpg',
    'I' : r'data\I\0.jpg',
    'J' : r'data\J\0.jpg',
    'K' : r'data\K\0.jpg',
    'L' : r'data\L\0.jpg',
    'M' : r'data\M\0.jpg',
    'N' : r'data\N\0.jpg',
    'O' : r'data\O\0.jpg',
    'P' : r'data\P\0.jpg',
    'Q' : r'data\Q\0.jpg',
    'R' : r'data\R\0.jpg',
    'S' : r'data\S\0.jpg',
    'T' : r'data\T\0.jpg',
    'U' : r'data\U\0.jpg',
    'V' : r'data\V\0.jpg',
    'W' : r'data\W\0.jpg',
    'X' : r'data\X\0.jpg',
    'Y' : r'data\Y\0.jpg',
    'Z' : r'data\Z\0.jpg'
}

input_string = input("Enter the input: ")
input_string = input_string.upper()

blank_img = np.zeros((420,420)) # Black image to represent space
cv2.imshow("Video", blank_img)
cv2.waitKey(2000)

for ch in input_string:
    if(ch == ' '):
        cv2.imshow("Video", blank_img)
        cv2.waitKey(1000)
    else:
        img = cv2.imread(data[ch])
        img = cv2.resize(img, (420, 420))
        cv2.imshow("Video", img)
        cv2.waitKey(1500)   