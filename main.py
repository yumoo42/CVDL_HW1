from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
# from PyQt5.QtCore import *
# from PyQt5.QtGui import *
import sys, hw1_ui
import glob
import os
import cv2
import numpy as np
from copy import copy

class Main(QMainWindow, hw1_ui.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.folder_imgs = [] # 存儲 loadFolder 加載的圖像
        self.imageL = None
        self.imageR = None
        self.image1 = None
        self.image2 = None

        self.loadImage = self.LoadImage(self)
        self.loadFolder.clicked.connect(self.loadImage.load_folder)
        self.loadImageL.clicked.connect(self.loadImage.load_imageL)
        self.loadImageR.clicked.connect(self.loadImage.load_imageR)

        self.calibration = self.Calibration(self)
        self.calibration_corners.clicked.connect(self.calibration.find_corners)
        self.calibration_intrinsic.clicked.connect(self.calibration.find_intrinsic)
        self.calibration_extrinsic.clicked.connect(self.calibration.find_extrinsic)
        self.calibration_distortion.clicked.connect(self.calibration.find_distortion)
        self.calibration_showResult.clicked.connect(self.calibration.show_result)

        self.augmentationReality = self.AugmentationReality(self)
        self.augmented_showOnBoard.clicked.connect(self.augmentationReality.show_words_on_board)
        self.augmented_showVertical.clicked.connect(self.augmentationReality.show_words_vertical)

        self.stereoDisparityMap = self.StereoDisparityMap(self)
        self.stereo_disparity_map.clicked.connect(self.stereoDisparityMap.find_disparity_map)

        self.sift = self.SIFT(self)
        self.sift_loadImage1.clicked.connect(self.loadImage.load_image1)
        self.sift_loadImage2.clicked.connect(self.loadImage.load_image2)
        self.sift_keypoints.clicked.connect(self.sift.find_keypoint)
        self.sift_matchedKeypoints.clicked.connect(self.sift.match_keypoint)

    class LoadImage:
        def __init__(self, main):
            self.main = main

        def load_folder(self):
            folderPath = QtWidgets.QFileDialog.getExistingDirectory()
            if not folderPath:
                return
            self.main.folderPath = folderPath
            img_file = [file.split("\\")[-1] for file in os.listdir(folderPath) if file.endswith('.bmp')]
            self.img_names = sorted(img_file, key=lambda x: int(x.split(".")[0]))
            self.main.folder_imgs = [cv2.imread(folderPath + "/" + file) for file in self.img_names]

            self.main.calibration_comboBox.clear()
            for i in range(1, len(self.main.folder_imgs) + 1):
                self.main.calibration_comboBox.addItem(str(i))  # 加入選項，從1開始

            print("The images have been loaded.")

        def load_imageL(self):
            filePath , filterType = QtWidgets.QFileDialog.getOpenFileName()
            self.main.imageL = cv2.imread(filePath)
            print("The images have been loaded.")
        
        def load_imageR(self):
            filePath , filterType = QtWidgets.QFileDialog.getOpenFileName()
            self.main.imageR = cv2.imread(filePath)
            print("The images have been loaded.")
        
        def load_image1(self):
            filePath , filterType = QtWidgets.QFileDialog.getOpenFileName()
            self.main.image1 = cv2.imread(filePath)
            print("The images have been loaded.")
        
        def load_image2(self):
            filePath , filterType = QtWidgets.QFileDialog.getOpenFileName()
            self.main.image2 = cv2.imread(filePath)
            print("The images have been loaded.")

    # 相機校準
    class Calibration:
        def __init__(self, main):
            self.main = main
            self.pattern_size = (11, 8) # 棋盤格(w, h)
            self.objectPoints = [] # 存放3D座標點
            self.imgPoints = [] # 存放2D圖像座標點

            # 定義棋盤格的3D坐標點，每個方格大小為 0.02m
            self.objp = np.zeros((self.pattern_size[0] * self.pattern_size[1], 3), np.float32)
            self.objp[:, :2] = np.mgrid[0:self.pattern_size[0], 0:self.pattern_size[1]].T.reshape(-1, 2) * 0.02

        # 1. 利用棋盤格檢測角點，進行校準
        def find_corners(self):
            if not self.main.folder_imgs:
                print("Please load a folder with images first.")
                return

            # 初始化參數
            winSize = (5, 5)
            zeroZone = (-1, -1)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            images = [img.copy() for img in self.main.folder_imgs]

            for img in images:
                grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 圖像轉為灰值
                ret, corners = cv2.findChessboardCorners(grayImg, self.pattern_size, None) # detect the corner 
                if ret:
                    corners = cv2.cornerSubPix(grayImg, corners, winSize, zeroZone, criteria) # increase accuracy
                    self.imgPoints.append(corners)
                    self.objectPoints.append(self.objp) # 每張圖像對應相同的3D點
                    cv2.drawChessboardCorners(img, self.pattern_size, corners, ret) # draw the corner 
                    # display the corners
                    cv2.namedWindow('Corner', cv2.WINDOW_NORMAL)
                    cv2.imshow("Corner", img)
                    cv2.waitKey(500)
            cv2.destroyAllWindows() 

        # 2. 計算內部參數、畸變參數
        def find_intrinsic(self):
            if not self.objectPoints or not self.imgPoints:
                print("Please press 'Find Corners' button first.")
                return

            gray = cv2.cvtColor(self.main.folder_imgs[0], cv2.COLOR_BGR2GRAY)
            image_size = gray.shape[::-1]  # (w, h)   

            ret, self.mtx, self.dist, self.rvecs, self.tvecs = cv2.calibrateCamera(self.objectPoints, self.imgPoints, image_size, None, None)
            print("Intrinsic matrix: \n", self.mtx)

        def find_extrinsic(self):
            if not hasattr(self, 'rvecs') or not hasattr(self, 'tvecs'):
                print("Please press 'Find Intrinsic' button first.")
                return

            select_img = int(self.main.calibration_comboBox.currentText())
            select_rvec = self.rvecs[select_img - 1]
            select_tvec = self.tvecs[select_img - 1]

            rotation_matrix, _ = cv2.Rodrigues(select_rvec)
            extrinsic_matrix = np.column_stack((rotation_matrix, select_tvec))
            print("Extrinsic\n", extrinsic_matrix)

        def find_distortion(self):
            if not hasattr(self, 'dist'):
                print("Please press 'Find Intrinsic' button first.")
                return

            print("Distortion\n", self.dist)
        
        # 3. 校正圖像畸變，生成更準確的影像
        def show_result(self):
            if not hasattr(self, 'mtx') or not hasattr(self, 'dist'):
                print("Please press 'Find Intrinsic' button first.")
                return

            images = [img.copy() for img in self.main.folder_imgs]
            for img in images:
                grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                undistort_img = cv2.undistort(grayImg, self.mtx, self.dist)

                combined_img = np.concatenate((grayImg, undistort_img), axis=1)
                cv2.namedWindow('Show result', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('Show result', 512, 256)
                cv2.imshow("Show result", combined_img)
                cv2.waitKey(500)
            cv2.destroyAllWindows()

    class AugmentationReality:
        def __init__(self, main):
            self.main = main
            self.pattern_size = (11, 8)
        
        def check_words_len(self):
            word = self.main.augmented_lineEdit.text()
            if len(word) > 6 or len(word) < 0:
                print("The word length should be between 0 and 6")
                return
            else:
                return word
        
        def get_new_word_position(self, i, letter):
            start_positions = [(7, 5), (4, 5), (1, 5), (7, 2), (4, 2), (1, 2)] # 初始位置
            
            col, row = start_positions[i]
            letter = letter.upper()
            points = np.array(self.fs.getNode(letter).mat()) # 讀取字母的3D坐標
            points[:, :, 0] += col  # x偏移
            points[:, :, 1] += row  # y偏移

            return np.array(points)

        def get_calibrateCamera(self):
            objp = np.zeros((self.pattern_size[0] * self.pattern_size[1], 3), np.float32)
            objp[:, :2] = np.mgrid[0:self.pattern_size[0], 0:self.pattern_size[1]].T.reshape(-1, 2)
            obj_points = []
            img_points = []

            images = [img.copy() for img in self.main.folder_imgs]
            for img in images:
                grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                ret, corners = cv2.findChessboardCorners(grayImg, self.pattern_size, None)
                if ret:
                    img_points.append(corners)
                    obj_points.append(objp)
            ret, self.mtx, self.dist, self.rvecs, self.tvecs = cv2.calibrateCamera(obj_points, img_points, grayImg.shape[::-1], None, None)

        def show_ar_results(self):
            words = self.check_words_len()
            # 1. 利用相機的參數，定位虛擬物的位置
            self.get_calibrateCamera()
            images = [img.copy() for img in self.main.folder_imgs]
            for img, rvec, tvec in zip(images, self.rvecs, self.tvecs):
                for i, letter in enumerate(words):
                    # 2. 讀取虛擬物體的 3D 坐標
                    charPoints = self.get_new_word_position(i, letter)
                    charPoints = np.float32(charPoints).reshape(-1, 3) # cv2.projectPoints() 期望輸入的 3D 點的形狀是 (N, 3)
                    # 3. 將 3D 坐標投影到 2D 圖像平面
                    newCharPoints, _ = cv2.projectPoints(charPoints, rvec, tvec, self.mtx, self.dist)
                    imgPoints = np.int32(newCharPoints).reshape(-1, 2) # cv2.projectPoints() 的輸出是 3D 點投影到 2D 平面的結果，通常形狀是 (N, 1, 2)，將其轉為 (N, 2)
                    for i in range(0, len(imgPoints) - 1, 2):
                        pointA = tuple(imgPoints[i].ravel())
                        pointB = tuple(imgPoints[i + 1].ravel())
                        cv2.line(img, pointA, pointB, (0, 0, 255), 15)
                cv2.namedWindow('Augmented Reality', cv2.WINDOW_NORMAL)
                cv2.imshow("Augmented Reality", img)
                cv2.waitKey(1000)
            self.fs.release()
            cv2.destroyAllWindows()

        def show_words_on_board(self):
            if not self.main.folder_imgs:
                print("Please load a folder with images first.")
                return
            self.fs = cv2.FileStorage(f'{self.main.folderPath}/Q2_db/alphabet_db_onboard.txt', cv2.FILE_STORAGE_READ)
            self.show_ar_results()
            
                
        def show_words_vertical(self):
            if not self.main.folder_imgs:
                print("Please load a folder with images first.")
                return
            self.fs = cv2.FileStorage(f'{self.main.folderPath}/Q2_db/alphabet_db_vertical.txt', cv2.FILE_STORAGE_READ)
            self.show_ar_results()

    # 立體視差圖
    class StereoDisparityMap:
        def __init__(self, main):
            self.main = main

        def find_disparity_map(self):
            if self.main.imageL is None:
                print("Please load ths left image first.")
                return
            if self.main.imageR is None:
                print("Please load the right image first.")
                return

            grayL = cv2.cvtColor(self.main.imageL, cv2.COLOR_BGR2GRAY)
            grayR = cv2.cvtColor(self.main.imageR, cv2.COLOR_BGR2GRAY)

            # 初始化參數
            # numDisparities = 432 (16的倍數) 值越大，視差範圍越廣，但計算時間增加
            # blockSize = 25 (奇數，範圍在 [5, 51] 之間) 值越大，計算越穩定，但細節可能丟失

            numDisparities = 64 
            blockSize = 13

            stereo = cv2.StereoBM_create(numDisparities, blockSize)
            disparity = stereo.compute(grayL, grayR) # 1. 計算視差圖(每個像素的深度資訊，視差值越大，距離越近)
            disparity_normalized = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U) # 2. 正規化視差圖[0, 255]

            resized_imageL = cv2.resize(self.main.imageL, (512, 512))
            resized_imageR = cv2.resize(self.main.imageR, (512, 512))
            resized_disparity = cv2.resize(disparity_normalized, (512, 512))

            cv2.imshow("ImageL", resized_imageL)
            cv2.imshow("ImageR", resized_imageR)
            cv2.imshow("Disparity Map", resized_disparity)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    # 特徵檢測與匹配
    class SIFT:
        def __init__(self, main):
            self.main = main

        # 1. 檢測圖像中的特徵點、計算特徵點的描述符
        def find_keypoint(self):
            if self.main.image1 is None:
                print("Please load ths image1 first.")
                return

            gray1 = cv2.cvtColor(self.main.image1, cv2.COLOR_BGR2GRAY)
            sift = cv2.SIFT_create()
            keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
            kp_img1 = cv2.drawKeypoints(gray1, keypoints1, None, color=(0, 255, 0))

            cv2.namedWindow('Keypoints', cv2.WINDOW_NORMAL)
            cv2.imshow("Keypoints", kp_img1)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # 2. 匹配不同圖像中的特徵點，以進行圖像配準
        def match_keypoint(self):
            if self.main.image1 is None:
                print("Please load ths image1 first.")
                return
            if self.main.image2 is None:
                print("Please load ths image2 first.")
                return

            gray1 = cv2.cvtColor(self.main.image1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(self.main.image2, cv2.COLOR_BGR2GRAY)
            sift = cv2.SIFT_create()
            keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
            keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

            bf = cv2.BFMatcher() # 暴力匹配 (Brute-Force Matcher) 進行特徵點匹配
            matches = bf.knnMatch(descriptors1, descriptors2, k=2)

            # 濾除不好的匹配點
            good_matches = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)

            # 繪製匹配的關鍵點
            match_img = cv2.drawMatches(gray1, keypoints1, gray2, keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

            cv2.namedWindow('Matched Keypoints', cv2.WINDOW_NORMAL)
            cv2.imshow("Matched Keypoints", match_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = Main()
    window.setWindowTitle('cvdlhw1')
    window.show()
    sys.exit(app.exec_())