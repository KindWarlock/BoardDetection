import cv2
import numpy as np

from board import Board


def draw_borders(frame, scale):
    rect_shape = (scale * 12, scale * 10)
    frame_center = np.array([frame.shape[0] // 2, frame.shape[1] // 2])
    l = frame_center[1] - rect_shape[1] // 2
    # Вычитание сотни - временное решение для телефонной камеры
    t = frame_center[0] - rect_shape[0] // 2 - 100
    borders = (l, t, rect_shape[1], rect_shape[0])
    cv2.rectangle(frame, borders, (255, 255, 255), 3)
    return borders


def returnCameraIndices():
    arr = []
    for i in range(10):
        # Пытаемся считать данные с каждой камеры поочередно. Если удается, принимаем ее за работающую. 
        cap = cv2.VideoCapture(i)
        if cap.read()[0]:
            arr.append(i)
            cap.release()
        i -= 1
    return arr


# cap = cv2.VideoCapture(returnCameraIndices[-1])
url = "http://192.168.1.39:8080/video"
cap = cv2.VideoCapture(url)

# Камера может снимать в высоком разрешении, поэтому его лучше предварительно уменьшить
resize_scale = 2

if (cap.isOpened()== False): 
    print("Error opening video stream or file")

board_present = False
my_board = Board()
borders = tuple()

while (cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        frame = cv2.resize(frame, (frame.shape[1] // resize_scale, frame.shape[0] // resize_scale))
        if cv2.waitKey(1) & 0xFF == ord('b'):
            b_present = my_board.find_board(frame, borders)
            if not b_present:
                print('No board detected!')
                continue
            my_board.rotate(frame, True)
            my_board.find_board(frame, borders)
            board_present = True

        if board_present:
            my_board.rotate(frame)
            my_board.check_lights(frame)
            my_board.draw(frame)
            
        else:
            borders = draw_borders(frame, 20)

        cv2.imshow('Frame', frame)         
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else: 
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue
 
cap.release()
 
cv2.destroyAllWindows()