# -*- coding: utf-8 -*-
import cv2
import dlib
import numpy as np
from imutils import face_utils
import socket
import time
import tkinter as tk
from transitions import Machine


host = "127.0.0.1"
port1 = 1234
port2 = 2345

sensorSize = 8.467
pixSize = 2.0*0.000000001

scaler = 0.5

mode = 0

face_landmark_path = './shape_predictor_68_face_landmarks.dat'
face_pose_sampleVideo = './face_estimation.mp4'

K = [6.5308391993466671e+002, 0.0, 3.1950000000000000e+002,
     0.0, 6.5308391993466671e+002, 2.3950000000000000e+002,
     0.0, 0.0, 1.0]
D = [7.0834633684407095e-002, 6.9140193737175351e-002, 0.0, 0.0, -1.3073460323689292e+000]
#カメラの情報（キャリブレーション）
cam_matrix = np.array(K).reshape(3, 3).astype(np.float32)
dist_coeffs = np.array(D).reshape(5, 1).astype(np.float32)

client1 = socket.socket(socket.AF_INET, socket.SOCK_STREAM) #オブジェクトの作成をします
client2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

client1.connect((host, port1)) #これでサーバーに接続します
client2.connect((host, port2))

object_pts = np.float32([[6.825897, 6.760612, 4.402142],
                         [1.330353, 7.122144, 6.903745],
                         [-1.330353, 7.122144, 6.903745],
                         [-6.825897, 6.760612, 4.402142],
                         [5.311432, 5.485328, 3.987654],
                         [1.789930, 5.393625, 4.413414],
                         [-1.789930, 5.393625, 4.413414],
                         [-5.311432, 5.485328, 3.987654],
                         [2.005628, 1.409845, 6.165652],
                         [-2.005628, 1.409845, 6.165652],
                         [2.774015, -2.080775, 5.048531],
                         [-2.774015, -2.080775, 5.048531],
                         [0.000000, -3.116408, 6.097667],
                         [0.000000, -7.415691, 4.070434]])


reprojectsrc = np.float32([[10.0, 10.0, 10.0],
                           [10.0, 10.0, -10.0],
                           [10.0, -10.0, -10.0],
                           [10.0, -10.0, 10.0],
                           [-10.0, 10.0, 10.0],
                           [-10.0, 10.0, -10.0],
                           [-10.0, -10.0, -10.0],
                           [-10.0, -10.0, 10.0]])

line_pairs = [[0, 1], [1, 2], [2, 3], [3, 0],
              [4, 5], [5, 6], [6, 7], [7, 4],
              [0, 4], [1, 5], [2, 6], [3, 7]]


def HeadAngle(function,Focallength,Coordinate,wide,widePropotion):
    wide = wide*widePropotion
    if function == "horizontal":
        fx = Focallength
        x = (Coordinate - 0.5*wide)*pixSize/widePropotion
        fxmm = fx*pixSize
        tan_x1 = x/fxmm
        tan_x2 = ((fxmm + 35*0.01) *tan_x1) / (20*0.01)
        return np.arctan(tan_x2)

    if function == "vertical":
        fy = Focallength
        y = (Coordinate - 0.5*wide)*pixSize/widePropotion
        fymm = fy*pixSize
        tan_y1 = y/fymm
        tan_y2 = ((fymm + 35*0.01) *tan_y1 + 20*0.01) / (20*0.01)
        return np.arctan(tan_y2)

def get_head_pose(shape):
    image_pts = np.float32([shape[17], shape[21], shape[22], shape[26], shape[36],
                            shape[39], shape[42], shape[45], shape[31], shape[35],
                            shape[48], shape[54], shape[57], shape[8]])

    _, rotation_vec, translation_vec = cv2.solvePnP(object_pts, image_pts, cam_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

    # calc euler angle
    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    pose_mat = cv2.hconcat((rotation_mat, translation_vec))
    _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_mat)

    (nose_end_point2D, _) = cv2.projectPoints(np.array([(0.0, 0.0, 500.0)]), rotation_vec,
                                                             translation_vec, cam_matrix, dist_coeffs)

    return euler_angle


def main():
    # return
    n = 0
    cap = cv2.VideoCapture(0)#face_pose_sampleVideo)
    if not cap.isOpened():
        print("Unable to connect to camera.")
        return
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(face_landmark_path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,120)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("fps:{}".format(fps))

    while cap.isOpened():#0.04
        start = time.time()
        ret, frame = cap.read() #0.003
        frame = cv2.flip(frame, -1)
        frame = cv2.resize(frame, (int(width*0.5), int(height*0.5)))


        if ret:
            face_rects = detector(frame) #0.03 -> 86%

            if len(face_rects) > 0:

                shape = predictor(frame, face_rects[0])     #
                shape = face_utils.shape_to_np(shape)       #
                cv2.circle(frame, shape[28],8,(0,0,0),-1)  #0.0025
                euler_angle = get_head_pose(shape)

                if euler_angle[1,0]<18 and euler_angle[1,0]>-18:
                    data_x = HeadAngle("horizontal",cam_matrix[0,0],shape[28,0],width,0.5)  #
                    data_y = HeadAngle("vertical",cam_matrix[1,1],shape[28,1],height,0.5)   #0.00003

                    print('Send_x : %s' % str(data_x))                  #
                    print('Send_y : %s' % str(data_y))                  #
                    print("18度以内")
                    client1.send(str(data_x).encode()) #データを送信します #
                    client2.send(str(data_y).encode())                  #0.0001
                    print(euler_angle[1,0])

                else:
                    print("do not look")
                    n = n + 0.02
                    seek_x = np.sin(n)
                    client1.send(str(seek_x).encode()) #データを送信します #
                    client2.send(str('-1.0').encode())
                    print(euler_angle[1,0])
                    print("18度以上")

            else:
                print('do not find face')
                n = n + 0.02
                seek_x = np.sin(n)
                client1.send(str(seek_x).encode()) #データを送信します #
                client2.send(str('-1.0').encode())


            cv2.imshow("demo", frame)               #0.00017


            if cv2.waitKey(1) & 0xFF == ord('q'):
                break                                   #0.0027
        elapsed_time = time.time() - start
        print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")



if __name__ == '__main__':

    main()
