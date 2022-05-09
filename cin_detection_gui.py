# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'test.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

import os
import gc
from operator import contains
import sys
import cv2
import argparse
import copy
import random
import torch
import numpy as np
import pandas as pd
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
from PIL import Image
from MyUtils import get_grayscale, remove_noise, thresholding, variance_of_laplacian, sharpen_image, dilate, erode, opening, canny, deskew, white_pad, scale
from arabicocr import arabic_ocr, cleanup_text
from PyQt5 import QtCore, QtGui, QtWidgets

from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact

from utils.torch_utils import select_device
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.datasets import letterbox
from utils.plots import plot_one_box

from Dialogs import new_file_dialog, settings_dialog

class Ui_MainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(Ui_MainWindow, self).__init__(parent)
        print('Starting Initialization.. ')
        self.setupUi(self)
        self.init_slots()
        self.out = None
        # self.out = cv2.VideoWriter('prediction.avi', cv2.VideoWriter_fourcc(*'XVID'), 20.0, (640, 480))

        self.opt={
            'weights':'./weights/best1280.pt',
            'source':0,
            'img_size':1280,
            'conf_thres':0.25,
            'iou_thres':0.45,
            'device':'',
            'view_img':False,
            'save_txt':False,
            'save_conf':False,
            'nosave':False,
            'classes':None,
            'agnostic_nms':False,
            'augment':False,
            'update':False,
            'project':'runs/detect',
            'name':'exp',
            'exist-ok':False,
            'cap_h':720,
            'cap_w':1280,
            'cin_conf_thres':0.93,
            'cin_sharp_thres':300,
            'save_rec':True
            }

        self.opt['device'] = select_device(self.opt['device'])
        self.half = self.opt['device'].type != 'cpu'  # half precision only supported on CUDA
        self.opt['half'] = self.half
        cudnn.benchmark = True

        # Load model
        self.model = attempt_load(self.opt['weights'], map_location=self.opt['device'])  # load FP32 model
        stride = int(self.model.stride.max())  # model stride
        self.imgsz = check_img_size(self.opt['img_size'], s=stride)  # check img_size
        if self.half:
            self.model.half()  # to FP16
        
        self.esrgan_model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        self.netscale = 2
        model_path = './BasicSR/Real-ESRGAN-master/experiments/pretrained_models/net_g_latest_ver3.pth'
        self.enhancer = RealESRGANer(scale=self.netscale, model_path=model_path, model=self.esrgan_model)

        # Get names and colors
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]

        self.opt['model']=self.model
        self.opt['names']=self.names
        self.opt['colors']=self.colors

        print('Initialization successful!')
        
        

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1241, 529)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setContentsMargins(5, 5, 5, 5)
        self.horizontalLayout.setObjectName("horizontalLayout")

        self.tableDisplay = QtWidgets.QTableWidget(self.centralwidget)
        self.tableDisplay.setObjectName("tableDisplay")
        self.tableDisplay.setColumnCount(0)
        self.tableDisplay.setRowCount(0)
        self.horizontalLayout.addWidget(self.tableDisplay)

        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setContentsMargins(5, 20, 5, 20)
        self.verticalLayout.setObjectName("verticalLayout")

        self.button_startstop_detection = QtWidgets.QPushButton(self.centralwidget)
        self.button_startstop_detection.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.button_startstop_detection.sizePolicy().hasHeightForWidth())
        self.button_startstop_detection.setSizePolicy(sizePolicy)
        self.button_startstop_detection.setObjectName("button_startstop_detection")
        self.verticalLayout.addWidget(self.button_startstop_detection)

        spacerItem = QtWidgets.QSpacerItem(20, 250, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.verticalLayout.addItem(spacerItem)
        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line1")
        self.verticalLayout.addWidget(self.line)

        self.openfile = QtWidgets.QPushButton(self.centralwidget)
        self.openfile.setObjectName("openfile")
        self.verticalLayout.addWidget(self.openfile)

        self.createfile = QtWidgets.QPushButton(self.centralwidget)
        self.createfile.setObjectName("createfile")
        self.verticalLayout.addWidget(self.createfile)

        self.line_2 = QtWidgets.QFrame(self.centralwidget)
        self.line_2.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.verticalLayout.addWidget(self.line_2)
        self.checkBox = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox.setObjectName("checkBox")
        self.verticalLayout.addWidget(self.checkBox)

        self.horizontalLayout.addLayout(self.verticalLayout)
        self.cameralabel = QtWidgets.QLabel(self.centralwidget)
        self.cameralabel.setObjectName("cameralabel")
        self.cameralabel.setScaledContents(True)
        self.horizontalLayout.addWidget(self.cameralabel)

        self.horizontalLayout.setStretch(0, 3)
        self.horizontalLayout.setStretch(1, 1)
        self.horizontalLayout.setStretch(2, 1)

        self.gridLayout.addLayout(self.horizontalLayout, 0, 0, 1, 1)
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
        MainWindow.setWindowTitle(_translate("MainWindow", "Smart CIN Detection"))
        self.button_startstop_detection.setText(_translate("MainWindow", "Start Detection"))
        self.openfile.setText(_translate("MainWindow", "Open File"))
        self.openfile.setStatusTip(_translate("MainWindow", "Open table file of type (CSV, Excel, ..etc)."))
        self.createfile.setText(_translate("MainWindow", "Create File"))
        self.createfile.setStatusTip(_translate("MainWindow", "Create table file of type (CSV, Excel, ..etc)."))
        self.checkBox.setText(_translate("MainWindow", "Auto-save changes"))
        self.checkBox.setStatusTip(_translate("MainWindow", "Auto-save changes made on the table manually."))
        self.cameralabel.setText(_translate("MainWindow", "Press 'Start Detection' button\nto start CIN detection process"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.actionSettings.setText(_translate("MainWindow", "Settings"))
        self.actionSettings.setShortcut(_translate("MainWindow", "F1"))
        self.actionExit.setText(_translate("MainWindow", "Exit"))

    def init_slots(self):
        self.actionExit.triggered.connect(self.exit_clicked)
        self.actionSettings.triggered.connect(self.settings_clicked)
        self.button_startstop_detection.clicked.connect(self.startstop_detection)
        self.openfile.clicked.connect(self.Open_file)
        self.createfile.clicked.connect(self.Create_file)
        self.checkBox.stateChanged.connect(self.save_table_changes)

    ########### File Menu Setup ##############

    def exit_clicked(self):
        close = QtWidgets.QMessageBox.question(self,
                                        "Exit",
                                        "Are you sure want to exit?",
                                        QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        if close == QtWidgets.QMessageBox.Yes:
            self.close()
        else:
            return None

    def settings_clicked(self):
        settings = settings_dialog.Ui_Dialog()
        resolution, camera_index, conf_thresh, sharp_thresh, save = settings.get_inputs()
        if camera_index:
            res_dic={720:1280, 480:720, 576:720, 1080:1920}
            self.opt['source']=int(camera_index)
            self.opt['cap_h']=int(resolution[:-1])
            self.opt['cap_w']=res_dic[int(resolution[:-1])]
            self.opt['cin_conf_thres']=float(int(conf_thresh[:2])/100)
            self.opt['cin_sharp_thres']=int(sharp_thresh[:3])
            self.opt['save_rec']=save
        
    ##########################################

    ######### Camera Label Setup #############
    def startstop_detection(self):
        worker_exists = False
        for obj in gc.get_objects():
            if isinstance(obj, Worker1):
                worker_exists = True
                print('Worker Exist')
                break
        if not worker_exists:
            cap_test = cv2.VideoCapture(self.opt['source'])
            if not cap_test.isOpened():
                QtWidgets.QMessageBox.warning(self, u"Warning", u"Failed to open camera", 
                    buttons=QtWidgets.QMessageBox.Ok, defaultButton=QtWidgets.QMessageBox.Ok)
                assert cap_test.isOpened(), f'Failed to open camera!'
            self.CamThread = Worker1(self.opt)
            self.CamThread.start()
            self.CamThread.ImageUpdate.connect(self.signal_receptor)
            self.button_startstop_detection.setText(u"Stop Detection")
            self.horizontalLayout.setStretch(0, 3)
            self.horizontalLayout.setStretch(1, 1)
            self.horizontalLayout.setStretch(2, 3)
        else:
            self.CamThread.ImageUpdate.disconnect()
            self.CamThread.stop()
            self.button_startstop_detection.setText(u"Start Detection")
            self.horizontalLayout.setStretch(0, 3)
            self.horizontalLayout.setStretch(1, 1)
            self.horizontalLayout.setStretch(2, 1)
            del(self.CamThread)
            gc.collect()
            self.cameralabel.clear()
    


    def signal_receptor(self, dic):
        self.cameralabel.setPixmap(QtGui.QPixmap.fromImage(dic['Qt_frame']))

        if len(dic['results'])==2:
            self.CamThread.ImageUpdate.disconnect()
            self.CamThread.stop()
            self.button_startstop_detection.setText(u"Start Detection")
            del(self.CamThread)
            gc.collect()
            self.cameralabel.setText("MainWindow", "Press 'Start Detection' button\nto start CIN detection process")

            self.horizontalLayout.setStretch(0, 3)
            self.horizontalLayout.setStretch(1, 1)
            self.horizontalLayout.setStretch(2, 1)

            results = {'Field': [], 'Value':[], 'Confidence_Level':[]}
            
            for key in dic['results'].keys():
                crop_imgs_enhanced = []
                bboxes_info, conf_scores, class_indexes, obj_names, img, crop_imgs = dic['results'][key]
                # enhancement using real-esrgan
                if len(img.shape) == 3 and img.shape[2] == 4:
                    img_mode = 'RGBA'
                else:
                    img_mode = None
                img = scale(img, 50)
                output, _ = self.enhancer.enhance(img, outscale=2)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
                for box in bboxes_info:
                    crop_imgs_enhanced.append(output[box[2]:box[3], box[0]:box[1]])


                preproc_imgs = crop_imgs_enhanced.copy()
                nbr_obj=len(obj_names)

                fig, axs = plt.subplots(nbr_obj, 1, figsize=(12,10))
                for i in range(nbr_obj):
                    axs[i].imshow(crop_imgs[i])
                plt.savefig(f'cropped_{str(key)}.png')

                fig, axs = plt.subplots(nbr_obj, 1, figsize=(12,10))
                for i in range(nbr_obj):
                    axs[i].imshow(crop_imgs_enhanced[i])
                plt.savefig(f'enhanced_cropped_{str(key)}.png')

                for i, img in enumerate(preproc_imgs):
                    pil_img = Image.fromarray(img)
                    pil_img = pil_img.resize(np.multiply(pil_img.size,2))
                    preproc_imgs[i] = thresholding(get_grayscale(np.array(pil_img)))
                fig2, axs2 = plt.subplots(nbr_obj,1, figsize=(12,10))
                for i in range(nbr_obj):
                    axs2[i].imshow(preproc_imgs[i], cmap='gray')
                plt.savefig(f'preprocessed_{str(key)}.png')

                easyocr_results={}
                for i, img in enumerate(preproc_imgs):
                    print(f'Object number {i}:')
                    temp_res = arabic_ocr(preproc_imgs[i])
                    easyocr_results[f'Object {i}'] = temp_res
                    print('='*25)
                
                easyocr_results_sorted = copy.deepcopy(easyocr_results)
                for obj_res in easyocr_results_sorted.values():
                    for i in obj_res:
                        i[1] = cleanup_text(i[1])
                for obj_res in easyocr_results_sorted.keys():
                    concat_text = ''
                    conf_avg = []
                    obj_list = easyocr_results_sorted[obj_res]
                    for i in range(len(obj_list)):
                        for j in range(i+1,len(obj_list)):
                            if obj_list[j][0][2][0] > obj_list[i][0][2][0]:
                                aux = obj_list[i]
                                obj_list[i] = obj_list[j]
                                obj_list[j] = aux
                    for i in range(len(obj_list)):
                        obj_list[i].pop(0)
                    easyocr_results_sorted[obj_res] = obj_list
                    
                    for i in easyocr_results_sorted[obj_res]:
                        concat_text = concat_text+' '+i[0]
                        conf_avg.append(i[1])
                    easyocr_results_sorted[obj_res] = [concat_text, np.mean(conf_avg)]
                
                names = ['cin_number','last_name','first_name','date_of_birth','place_of_birth','mother_name','job','address','cin_date']
                for i, key in enumerate(easyocr_results_sorted.keys()):
                    print(self.opt['names'][class_indexes[i]])
                    results['Field'].append(self.opt['names'][class_indexes[i]])
                    results['Value'].append(easyocr_results_sorted[key][0])
                    results['Confidence_Level'].append(np.round(easyocr_results_sorted[key][1],4))
            results_df = pd.DataFrame(results)

            if (self.tableDisplay.rowCount(), self.tableDisplay.columnCount())==(0,0):
                self.table_path = f'./created_table_files/temp.csv'
                with open(self.table_path, 'w', encoding = 'utf-8') as file:
                    file.write('cin_number,last_name,first_name,date_of_birth,place_of_birth,mother_name,job,address,cin_date\n')
                self.table = pd.read_csv(self.table_path)
                self.Show_Table_Data()
            
            rowPosition = self.tableDisplay.rowCount()
            self.tableDisplay.insertRow(rowPosition)
            
            
            for c in range(self.tableDisplay.columnCount()):
                self.tableDisplay.setItem(rowPosition, c, QtWidgets.QTableWidgetItem(str(results_df[results_df['Field']==names[c]]['Value'].values[0])))




            
            

    ##########################################

    ############# Open & Create table file Setup ################
    def Open_file(self):
        path = QtWidgets.QFileDialog.getOpenFileName(self, "Open File", "./created_table_files", "CSV (*.csv);; Excel (*.xls *.xlsx *.xlsm *.xlsb)")
        if "CSV" in path[1]:
            self.table_path = path[0]
            self.table = pd.read_csv(self.table_path)
            self.Show_Table_Data()
            self.table_file_name = self.table_path.split('/')[-1].split('.')[0]
            self.table_file_type = self.table_path.split('/')[-1].split('.')[1]
        elif "Excel" in path[1]:
            self.table = pd.read_excel(self.table_path)
            self.table.fillna('',inplace=True)
            self.Show_Table_Data()
            self.table_file_name = self.table_path.split('/')[-1].split('.')[0]
            self.table_file_type = self.table_path.split('/')[-1].split('.')[1]
        elif len(path[1])==0 and len(path[0])==0:
            return None
        else:
            print(path)
            QtWidgets.QMessageBox.warning(self, u"Warning", u"The selected file type is not supported!", 
                    buttons=QtWidgets.QMessageBox.Ok, defaultButton=QtWidgets.QMessageBox.Ok)
    

    def Create_file(self):
        new_file_ui = new_file_dialog.Ui_Dialog()
        new_file_name, new_file_type = new_file_ui.get_inputs()
        if (not new_file_name is None) and (not new_file_type is None):
            msg = self.validate_file_name(new_file_name, new_file_type)
        else:
            msg= None
        if isinstance(msg, str):
            QtWidgets.QMessageBox.warning(self, u"Warning", msg, 
                    buttons=QtWidgets.QMessageBox.Ok, defaultButton=QtWidgets.QMessageBox.Ok)
        else:
            self.table_path = f'./created_table_files/{new_file_name}.{new_file_type}'
            with open(self.table_path, 'w', encoding = 'utf-8') as file:
                file.write('cin_number,last_name,first_name,date_of_birth,place_of_birth,mother_name,job,address,cin_date\n')
            self.table_file_name = new_file_name
            self.table_file_type = new_file_type

            if new_file_type == "CSV":
                self.table = pd.read_csv(self.table_path)
                self.Show_Table_Data()
            elif new_file_type == "Excel":
                self.table = pd.read_excel(self.table_path)
                self.table.fillna('',inplace=True)
                self.Show_Table_Data()

    @staticmethod
    def validate_file_name(name:str, type:str):
        if len(name)==0 or len(name)>15:
            return u"File name should be between 1 and 15 characters!"

        chars = set("#%&{}\<>*?/$!':@`|=").union(set('"'))
        if any((c in chars) for c in name):
            return u"File name should not contain special characters (#%&{}\<>*?/$!':@`|=\")"
        
        dir_list = os.listdir("./created_table_files")
        if (len(dir_list) != 0) and any((f'{name}.{type}' in f) for f in dir_list):
            return u"File already exist, use Open file instead."

        return True

    def Show_Table_Data(self):
        nbr_cols = self.table.shape[1]
        nbr_rows = self.table.shape[0]
        self.tableDisplay.setColumnCount(nbr_cols)
        self.tableDisplay.setRowCount(nbr_rows)
        self.tableDisplay.setHorizontalHeaderLabels(self.table.columns)

        for j in range(nbr_cols):
            for i in range(nbr_rows):
                self.tableDisplay.setItem(i,j, QtWidgets.QTableWidgetItem(str(self.table.iloc[i,j])))
    
        self.tableDisplay.resizeColumnsToContents()
        self.tableDisplay.itemChanged.connect(self.save_table_changes)

    def save_table_changes(self):
        if self.checkBox.isChecked():
            for currentQTableWidgetItem in self.tableDisplay.selectedItems():
                i, j = currentQTableWidgetItem.row(),  currentQTableWidgetItem.column()
                changed_item = self.tableDisplay.item(i, j).text()
                self.table.iloc[i, j] = changed_item
                if self.table_path.split(".")[-1]=="CSV":
                    self.table.to_csv(self.table_path, index=False)
        else:
            return None
    ##########################################
        
    


            


class Worker1(QtCore.QThread):
    # Signal
    ImageUpdate = QtCore.pyqtSignal(dict)

    # Initialization
    def __init__(self, yolo_config):
        super().__init__()
        self.opt = yolo_config
        if self.opt['save_rec']:
            i = 0
            while os.path.exists(f'./recorded detections/prediction_{i}.mp4'):
                i += 1
            os.listdir('./recorded detections')
            self.out = cv2.VideoWriter(f'./recorded detections/prediction_{i}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 25, (1280,720))

    # Running Yolov5
    def run(self):
        self.ThreadActive = True
        cap = cv2.VideoCapture(self.opt['source'])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.opt['cap_h'])
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.opt['cap_w'])
        name_list = []
        self.emission={}
        cin_front_detected = False
        cin_back_detected = False
        color_text_front = (192,192,192)
        color_text_back = (192,192,192)
        yolo_res={}
        conf_front = 0
        conf_back = 0

        while self.ThreadActive:
            flag, img = cap.read()
            if flag == False:
                self.emission['Qt_frame'] = None
                self.emission['Cv_frame'] = None
                self.ImageUpdate.emit(self.emission)
            else:
                im0 = img.copy()
                im00 = img.copy()
                with torch.no_grad():
                    img = letterbox(img, new_shape=640, stride=64)[0]
                    # Convert
                    # BGR to RGB, to 3x416x416
                    img = img[:, :, ::-1].transpose(2, 0, 1)
                    img = np.ascontiguousarray(img)
                    img = torch.from_numpy(img).to(self.opt['device'])
                    img = img.half() if self.opt['half'] else img.float()  # uint8 to fp16/32
                    img /= 255.0  # 0 - 255 to 0.0 - 1.0
                    if img.ndimension() == 3:
                        img = img.unsqueeze(0)
                    
                    # Inference
                    pred = self.opt['model'](img, augment=self.opt['augment'])[0]

                    # Apply NMS
                    pred = non_max_suppression(pred, self.opt['conf_thres'], self.opt['iou_thres'], classes=self.opt['classes'],
                                            agnostic=self.opt['agnostic_nms'])
                    
                    # Process detections
                    for i, det in enumerate(pred):  # detections per image
                        bboxes_info = []
                        confidence_scores = []
                        class_indexes = []
                        objects_names = []
                        cropped_imgs = []
                        conf_front = 0
                        conf_back = 0

                        if det is not None and len(det):
                            # Rescale boxes from img_size to im0 size
                            det[:, :4] = scale_coords(
                                img.shape[2:], det[:, :4], im0.shape).round()
                            # Write results
                            for *xyxy, conf, cls in reversed(det):
                                label = '%s %.2f' % (self.opt['names'][int(cls)], conf)
                                name_list.append(self.opt['names'][int(cls)])
                                plot_one_box(xyxy, im0, label=label, color=self.opt['colors'][int(cls)], line_thickness=1)
                                x1 = int(xyxy[0].item())
                                y1 = int(xyxy[1].item())
                                x2 = int(xyxy[2].item())
                                y2 = int(xyxy[3].item())
                                bboxes_info.append([x1,x2,y1,y2])
                                confidence_scores.append(float(conf))
                                class_indexes.append(int(cls))
                                objects_names.append(self.opt['names'][int(cls)])
                                cropped_imgs.append(im00[y1:y2, x1:x2])

                        # Checking if a CIN is detected and pausing capture if positive,
                        # and returning detected objects
                        #im0 = sharpen_image(im0, 1)

                        gray = cv2.cvtColor(im0, cv2.COLOR_BGR2GRAY)
                        fm = variance_of_laplacian(gray)
                        blurry = False
                        if fm<self.opt['cin_sharp_thres']:
                            blurry = True
                        if confidence_scores:
                            if np.mean(confidence_scores)>0:
                                print(f'mean conf: {np.mean(confidence_scores)}, blur: {fm}')
                        if {'first_name', 'last_name', 'date_of_birth','place_of_birth', 'cin_number'}.issubset(set(objects_names))\
                             and np.mean(confidence_scores)>=self.opt['cin_conf_thres'] and np.mean(confidence_scores)>conf_front and blurry == False:
                            conf_front = np.mean(confidence_scores)
                            cin_front_detected = True
                            color_text_front = (0,255,0)
                            yolo_res['front']=[bboxes_info, confidence_scores, class_indexes, objects_names, im00, cropped_imgs]
                        if {'mother_name', 'job', 'address','cin_date'}.issubset(set(objects_names)) and np.mean(confidence_scores)>=self.opt['cin_conf_thres'] \
                            and np.mean(confidence_scores)>conf_back and blurry == False:
                            conf_back = np.mean(confidence_scores)
                            cin_back_detected = True
                            color_text_back = (0,255,0)
                            yolo_res['back']=[bboxes_info, confidence_scores, class_indexes, objects_names, im00, cropped_imgs]
                        
                        #cv2.rectangle(im0, (0,0), (600,170), (255,255,255), -1, cv2.LINE_AA)
                        cv2.putText(im0, 'CIN Front Detected!', (25,100), cv2.FONT_HERSHEY_DUPLEX, 1, color_text_front, 2, lineType=cv2.LINE_AA)
                        cv2.putText(im0, 'CIN Back Detected!', (25,150), cv2.FONT_HERSHEY_DUPLEX, 1, color_text_back, 2, lineType=cv2.LINE_AA)

                        if not cin_front_detected or not cin_back_detected:    
                            cv2.putText(im0, 'Detecting..', (25,50), cv2.FONT_HERSHEY_DUPLEX, 1, (30,250,250), 2, lineType=cv2.LINE_AA)
                        else:
                            cv2.putText(im0, 'Detection completed successfully!', (25,50), cv2.FONT_HERSHEY_DUPLEX, 1, (0,255,0), 2, lineType=cv2.LINE_AA)

                        im0 = np.asarray(im0)
                        

                img = cv2.resize(im0, (1280, 720))
                cv_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                qt_img = QtGui.QImage(cv_img.data, cv_img.shape[1], cv_img.shape[0], QtGui.QImage.Format_RGB888)
                self.emission['Qt_frame'] = qt_img.scaled(640, 480, QtCore.Qt.KeepAspectRatio)
                self.emission['Cv_frame'] = cv_img
                self.emission['results'] = yolo_res
                print(im0.shape)
                self.out.write(im0)
                self.ImageUpdate.emit(self.emission)
    
    def stop(self):
        self.ThreadActive = False
        self.out.release()
        self.quit()



if __name__ == "__main__":
    qss_file = open('./style/Combinear.qss').read()
    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(qss_file)
    app.setApplicationName("Smart CIN Detection")
    app.setWindowIcon(QtGui.QIcon('Icon.png'))
    ui = Ui_MainWindow()
    ui.show()
    sys.exit(app.exec_())

