# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'test.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

import sys
import cv2
import argparse
import random
import torch
import numpy as np
import torch.backends.cudnn as cudnn

from PyQt5 import QtCore, QtGui, QtWidgets

from utils.torch_utils import select_device
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.datasets import letterbox
from utils.plots import plot_one_box
print('...')
class Ui_MainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(Ui_MainWindow, self).__init__(parent)
        print('Starting Initialization.. ')
        self.timer_video = QtCore.QTimer()
        self.setupUi(self)
        self.init_slots()
        self.cap = cv2.VideoCapture()
        self.out = None
        print('Initialization successful!')
        # self.out = cv2.VideoWriter('prediction.avi', cv2.VideoWriter_fourcc(*'XVID'), 20.0, (640, 480))

        parser = argparse.ArgumentParser()
        parser.add_argument('--weights', nargs='+', type=str,default='./weights/best1280.pt', help='model.pt path(s)')
        # file/folder, 0 for webcam
        parser.add_argument('--source', type=str,default='data/images', help='source')
        parser.add_argument('--img-size', type=int,default=1280, help='inference size (pixels)')
        parser.add_argument('--conf-thres', type=float,default=0.25, help='object confidence threshold')
        parser.add_argument('--iou-thres', type=float,default=0.45, help='IOU threshold for NMS')
        parser.add_argument('--device', default='',help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument('--view-img', action='store_true', help='display results')
        parser.add_argument('--save-txt', action='store_true',help='save results to *.txt')
        parser.add_argument('--save-conf', action='store_true',help='save confidences in --save-txt labels')
        parser.add_argument('--nosave', action='store_true',help='do not save images/videos')
        parser.add_argument('--classes', nargs='+', type=int,help='filter by class: --class 0, or --class 0 2 3')
        parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
        parser.add_argument('--augment', action='store_true',help='augmented inference')
        parser.add_argument('--update', action='store_true',help='update all models')
        parser.add_argument('--project', default='runs/detect',help='save results to project/name')
        parser.add_argument('--name', default='exp',help='save results to project/name')
        parser.add_argument('--exist-ok', action='store_true',help='existing project/name ok, do not increment')
        self.opt = parser.parse_args()
        print(self.opt)

        source, weights, view_img, save_txt, imgsz = self.opt.source, self.opt.weights, self.opt.view_img, self.opt.save_txt, self.opt.img_size

        self.device = select_device(self.opt.device)
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA
        print('self_device',self.device)
        cudnn.benchmark = True

        # Load model
        self.model = attempt_load(weights, map_location=self.device)  # load FP32 model
        stride = int(self.model.stride.max())  # model stride
        self.imgsz = check_img_size(imgsz, s=stride)  # check img_size
        if self.half:
            self.model.half()  # to FP16

        # Get names and colors
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]


    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(993, 502)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(0, 0, 991, 461))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout.setContentsMargins(5, 5, 5, 5)
        self.horizontalLayout.setObjectName("horizontalLayout")

        self.csvlabel = QtWidgets.QLabel(self.horizontalLayoutWidget)
        self.csvlabel.setObjectName("csvlabel")

        self.horizontalLayout.addWidget(self.csvlabel)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setContentsMargins(5, 20, 5, 20)
        self.verticalLayout.setObjectName("verticalLayout")

        self.button_startstop_detection = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.button_startstop_detection.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.button_startstop_detection.sizePolicy().hasHeightForWidth())
        self.button_startstop_detection.setSizePolicy(sizePolicy)
        self.button_startstop_detection.setObjectName("button_startstop_detection")
        self.verticalLayout.addWidget(self.button_startstop_detection)

        spacerItem = QtWidgets.QSpacerItem(20, 70, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        self.verticalLayout.addItem(spacerItem)
        self.line = QtWidgets.QFrame(self.horizontalLayoutWidget)
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line1")
        self.verticalLayout.addWidget(self.line)

        self.openfile = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.openfile.setObjectName("openfile")
        self.verticalLayout.addWidget(self.openfile)

        self.createfile = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        self.createfile.setObjectName("createfile")
        self.verticalLayout.addWidget(self.createfile)

        self.horizontalLayout.addLayout(self.verticalLayout)
        self.cameralabel = QtWidgets.QLabel(self.horizontalLayoutWidget)
        self.cameralabel.setObjectName("cameralabel")
        self.horizontalLayout.addWidget(self.cameralabel)

        self.horizontalLayout.setStretch(0, 3)
        self.horizontalLayout.setStretch(1, 1)
        self.horizontalLayout.setStretch(2, 3)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 993, 21))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionSettings = QtWidgets.QAction(MainWindow)
        self.actionSettings.setObjectName("actionSettings")
        self.actionExit = QtWidgets.QAction(MainWindow)
        self.actionExit.setObjectName("actionExit")
        self.menuFile.addAction(self.actionSettings)
        self.menuFile.addAction(self.actionExit)
        self.menubar.addAction(self.menuFile.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Test"))
        self.csvlabel.setText(_translate("MainWindow", "TextLabel"))
        self.button_startstop_detection.setText(_translate("MainWindow", "Start Detection"))
        self.openfile.setText(_translate("MainWindow", "Open File"))
        self.createfile.setText(_translate("MainWindow", "Create File"))
        self.cameralabel.setText(_translate("MainWindow", "TextLabel"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.actionSettings.setText(_translate("MainWindow", "Settings"))
        self.actionExit.setText(_translate("MainWindow", "Exit"))

    def init_slots(self):
        self.button_startstop_detection.clicked.connect(self.startstop_detection)
        self.timer_video.timeout.connect(self.show_frames)

    def startstop_detection(self):
        if not self.timer_video.isActive():
            flag = self.cap.open(0)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            if flag == False:
                QtWidgets.QMessageBox.warning(
                    self, u"Warning", u"Failed to open camera", buttons=QtWidgets.QMessageBox.Ok, defaultButton=QtWidgets.QMessageBox.Ok)
            else:
                self.out = cv2.VideoWriter('prediction.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 25, (int(self.cap.get(3)), int(self.cap.get(4))))
                self.timer_video.start(30)
                self.button_startstop_detection.setText(u"Stop Detection")
        else:
            self.timer_video.stop()
            self.cap.release()
            self.out.release()
            self.cameralabel.clear()
            self.button_startstop_detection.setText(u"Start Detection")

    def show_frames(self):
        name_list = []
        flag, img = self.cap.read()
        if img is not None:
            im0 = img.copy()
            with torch.no_grad():
                print(img.shape)
                img = letterbox(img, new_shape=self.opt.img_size)[0]
                # Convert
                # BGR to RGB, to 3x416x416
                img = img[:, :, ::-1].transpose(2, 0, 1)
                img = np.ascontiguousarray(img)
                img = torch.from_numpy(img).to(self.device)
                img = img.half() if self.half else img.float()  # uint8 to fp16/32
                img /= 255.0  # 0 - 255 to 0.0 - 1.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)
                # Inference
                pred = self.model(img, augment=self.opt.augment)[0]

                # Apply NMS
                pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres, classes=self.opt.classes,
                                           agnostic=self.opt.agnostic_nms)
                # Process detections
                for i, det in enumerate(pred):  # detections per image
                    if det is not None and len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_coords(
                            img.shape[2:], det[:, :4], im0.shape).round()
                        # Write results
                        for *xyxy, conf, cls in reversed(det):
                            label = '%s %.2f' % (self.names[int(cls)], conf)
                            name_list.append(self.names[int(cls)])
                            print(label)
                            plot_one_box(xyxy, im0, label=label, color=self.colors[int(cls)], line_thickness=2)

            self.out.write(im0)
            show = cv2.resize(im0, (1280, 720))
            self.result = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
            showImage = QtGui.QImage(self.result.data, self.result.shape[1], self.result.shape[0],
                                     QtGui.QImage.Format_RGB888)
            self.cameralabel.setPixmap(QtGui.QPixmap.fromImage(showImage))

        else:
            self.timer_video.stop()
            self.cap.release()
            self.out.release()
            self.cameralabel.clear()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    ui = Ui_MainWindow()
    ui.show()
    sys.exit(app.exec_())

