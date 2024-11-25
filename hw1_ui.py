# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'cvdl_hw1.ui'
#
# Created by: PyQt5 UI code generator 5.15.11
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(40, 20, 151, 261))
        self.groupBox.setObjectName("groupBox")
        self.loadFolder = QtWidgets.QPushButton(self.groupBox)
        self.loadFolder.setGeometry(QtCore.QRect(20, 40, 111, 41))
        self.loadFolder.setObjectName("loadFolder")
        self.loadImageL = QtWidgets.QPushButton(self.groupBox)
        self.loadImageL.setGeometry(QtCore.QRect(20, 100, 111, 41))
        self.loadImageL.setObjectName("loadImageL")
        self.loadImageR = QtWidgets.QPushButton(self.groupBox)
        self.loadImageR.setGeometry(QtCore.QRect(20, 160, 111, 41))
        self.loadImageR.setObjectName("loadImageR")
        self.groupBox_2 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_2.setGeometry(QtCore.QRect(220, 19, 151, 261))
        self.groupBox_2.setObjectName("groupBox_2")
        self.calibration_corners = QtWidgets.QPushButton(self.groupBox_2)
        self.calibration_corners.setGeometry(QtCore.QRect(30, 20, 93, 28))
        self.calibration_corners.setObjectName("calibration_corners")
        self.calibration_intrinsic = QtWidgets.QPushButton(self.groupBox_2)
        self.calibration_intrinsic.setGeometry(QtCore.QRect(30, 50, 93, 28))
        self.calibration_intrinsic.setObjectName("calibration_intrinsic")
        self.groupBox_3 = QtWidgets.QGroupBox(self.groupBox_2)
        self.groupBox_3.setGeometry(QtCore.QRect(10, 90, 131, 91))
        self.groupBox_3.setObjectName("groupBox_3")
        self.calibration_extrinsic = QtWidgets.QPushButton(self.groupBox_3)
        self.calibration_extrinsic.setGeometry(QtCore.QRect(20, 50, 93, 28))
        self.calibration_extrinsic.setObjectName("calibration_extrinsic")
        self.calibration_comboBox = QtWidgets.QComboBox(self.groupBox_3)
        self.calibration_comboBox.setGeometry(QtCore.QRect(30, 20, 69, 22))
        self.calibration_comboBox.setObjectName("calibration_comboBox")
        self.calibration_distortion = QtWidgets.QPushButton(self.groupBox_2)
        self.calibration_distortion.setGeometry(QtCore.QRect(30, 190, 93, 28))
        self.calibration_distortion.setObjectName("calibration_distortion")
        self.calibration_showResult = QtWidgets.QPushButton(self.groupBox_2)
        self.calibration_showResult.setGeometry(QtCore.QRect(30, 220, 93, 28))
        self.calibration_showResult.setObjectName("calibration_showResult")
        self.groupBox_4 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_4.setGeometry(QtCore.QRect(220, 310, 151, 181))
        self.groupBox_4.setObjectName("groupBox_4")
        self.sift_loadImage1 = QtWidgets.QPushButton(self.groupBox_4)
        self.sift_loadImage1.setGeometry(QtCore.QRect(10, 20, 131, 28))
        self.sift_loadImage1.setObjectName("sift_loadImage1")
        self.sift_loadImage2 = QtWidgets.QPushButton(self.groupBox_4)
        self.sift_loadImage2.setGeometry(QtCore.QRect(12, 60, 131, 28))
        self.sift_loadImage2.setObjectName("sift_loadImage2")
        self.sift_keypoints = QtWidgets.QPushButton(self.groupBox_4)
        self.sift_keypoints.setGeometry(QtCore.QRect(10, 100, 131, 28))
        self.sift_keypoints.setObjectName("sift_keypoints")
        self.sift_matchedKeypoints = QtWidgets.QPushButton(self.groupBox_4)
        self.sift_matchedKeypoints.setGeometry(QtCore.QRect(10, 140, 131, 28))
        self.sift_matchedKeypoints.setObjectName("sift_matchedKeypoints")
        self.groupBox_6 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_6.setGeometry(QtCore.QRect(400, 20, 171, 261))
        self.groupBox_6.setObjectName("groupBox_6")
        self.augmented_lineEdit = QtWidgets.QLineEdit(self.groupBox_6)
        self.augmented_lineEdit.setGeometry(QtCore.QRect(20, 30, 131, 31))
        self.augmented_lineEdit.setObjectName("augmented_lineEdit")
        self.augmented_showOnBoard = QtWidgets.QPushButton(self.groupBox_6)
        self.augmented_showOnBoard.setGeometry(QtCore.QRect(20, 100, 131, 28))
        self.augmented_showOnBoard.setObjectName("augmented_showOnBoard")
        self.augmented_showVertical = QtWidgets.QPushButton(self.groupBox_6)
        self.augmented_showVertical.setGeometry(QtCore.QRect(20, 140, 131, 28))
        self.augmented_showVertical.setObjectName("augmented_showVertical")
        self.groupBox_7 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_7.setGeometry(QtCore.QRect(600, 20, 151, 261))
        self.groupBox_7.setObjectName("groupBox_7")
        self.stereo_disparity_map = QtWidgets.QPushButton(self.groupBox_7)
        self.stereo_disparity_map.setGeometry(QtCore.QRect(10, 40, 131, 28))
        self.stereo_disparity_map.setObjectName("stereo_disparity_map")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 25))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.groupBox.setTitle(_translate("MainWindow", "Load Image"))
        self.loadFolder.setText(_translate("MainWindow", "Load Folder"))
        self.loadImageL.setText(_translate("MainWindow", "Load Image L"))
        self.loadImageR.setText(_translate("MainWindow", "Load Image R"))
        self.groupBox_2.setTitle(_translate("MainWindow", "1. Calibration"))
        self.calibration_corners.setText(_translate("MainWindow", "1.1 Find Corners"))
        self.calibration_intrinsic.setText(_translate("MainWindow", "1.2 Find Intrinsic"))
        self.groupBox_3.setTitle(_translate("MainWindow", "1.3 Find Extrinsic"))
        self.calibration_extrinsic.setText(_translate("MainWindow", "1.3 Find Extrinsic"))
        self.calibration_distortion.setText(_translate("MainWindow", "1.4 Find Distortion"))
        self.calibration_showResult.setText(_translate("MainWindow", "1.5 Show Result"))
        self.groupBox_4.setTitle(_translate("MainWindow", "4. SIFT"))
        self.sift_loadImage1.setText(_translate("MainWindow", "Load Image 1"))
        self.sift_loadImage2.setText(_translate("MainWindow", "Load Image 2"))
        self.sift_keypoints.setText(_translate("MainWindow", "4.1 Keypoints"))
        self.sift_matchedKeypoints.setText(_translate("MainWindow", "4.2 Matched Keypoints"))
        self.groupBox_6.setTitle(_translate("MainWindow", "2. Augmented Reality"))
        self.augmented_showOnBoard.setText(_translate("MainWindow", "2.1 Show Words on Board"))
        self.augmented_showVertical.setText(_translate("MainWindow", "2.2 Show Words Vertical"))
        self.groupBox_7.setTitle(_translate("MainWindow", "3. Stereo Disparity Map"))
        self.stereo_disparity_map.setText(_translate("MainWindow", "3.1 Stereo Disparity Map"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
