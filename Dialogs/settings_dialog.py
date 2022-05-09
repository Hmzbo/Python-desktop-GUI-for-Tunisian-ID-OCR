from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog(QtWidgets.QDialog):
    def __init__(self):
        super(Ui_Dialog, self).__init__()
        self.setupUi(self)
        self.setFixedSize(640, 251)

    def setupUi(self, Dialog):
        Dialog.setObjectName("Settings")
        #Dialog.resize(640, 251)
        self.gridLayout = QtWidgets.QGridLayout(Dialog)
        self.gridLayout.setObjectName("gridLayout")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setContentsMargins(5, -1, 5, -1)
        self.verticalLayout.setSpacing(6)
        self.verticalLayout.setObjectName("verticalLayout")
        self.label = QtWidgets.QLabel(Dialog)
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.verticalLayout.addWidget(self.label)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setSpacing(6)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setContentsMargins(5, 0, -1, 0)
        self.verticalLayout_2.setSpacing(15)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.cem_res_setting = QtWidgets.QLabel(Dialog)
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.cem_res_setting.setFont(font)
        self.cem_res_setting.setObjectName("cem_res_setting")
        self.verticalLayout_2.addWidget(self.cem_res_setting)
        self.label_2 = QtWidgets.QLabel(Dialog)
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.verticalLayout_2.addWidget(self.label_2)
        self.label_3 = QtWidgets.QLabel(Dialog)
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.verticalLayout_2.addWidget(self.label_3)
        self.label_4 = QtWidgets.QLabel(Dialog)
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.verticalLayout_2.addWidget(self.label_4)
        self.label_5 = QtWidgets.QLabel(Dialog)
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(True)
        font.setWeight(75)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.verticalLayout_2.addWidget(self.label_5)
        self.horizontalLayout.addLayout(self.verticalLayout_2)
        self.line = QtWidgets.QFrame(Dialog)
        self.line.setFrameShape(QtWidgets.QFrame.VLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.horizontalLayout.addWidget(self.line)
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setSpacing(15)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.radioButton_480p = QtWidgets.QRadioButton(Dialog)
        self.radioButton_480p.setObjectName("radioButton_480p")
        self.horizontalLayout_2.addWidget(self.radioButton_480p)
        self.radioButton_576p = QtWidgets.QRadioButton(Dialog)
        self.radioButton_576p.setObjectName("radioButton_576p")
        self.horizontalLayout_2.addWidget(self.radioButton_576p)
        self.radioButton_720p = QtWidgets.QRadioButton(Dialog)
        self.radioButton_720p.setObjectName("radioButton_720p")
        self.horizontalLayout_2.addWidget(self.radioButton_720p)
        self.radioButton_720p.setChecked(True)
        self.radioButton_1080p = QtWidgets.QRadioButton(Dialog)
        self.radioButton_1080p.setObjectName("radioButton_1080p")
        self.horizontalLayout_2.addWidget(self.radioButton_1080p)
        self.verticalLayout_3.addLayout(self.horizontalLayout_2)
        self.spinBox = QtWidgets.QSpinBox(Dialog)
        self.spinBox.setObjectName("spinBox")
        self.verticalLayout_3.addWidget(self.spinBox)
        self.comboBox_2 = QtWidgets.QComboBox(Dialog)
        self.comboBox_2.setObjectName("comboBox_2")
        self.comboBox_2.addItem("90%")
        self.comboBox_2.addItem("91%")
        self.comboBox_2.addItem("92%")
        self.comboBox_2.addItem("93% (Default/Recommended)")
        self.comboBox_2.addItem("94%")
        self.comboBox_2.addItem("95%")
        self.comboBox_2.addItem("96%")
        self.comboBox_2.addItem("97%")
        self.comboBox_2.addItem("98%")
        self.comboBox_2.addItem("99%")
        self.verticalLayout_3.addWidget(self.comboBox_2)
        self.comboBox = QtWidgets.QComboBox(Dialog)
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("250")
        self.comboBox.addItem("300 (default/Recommended)")
        self.comboBox.addItem("350")
        self.comboBox.addItem("400")
        self.comboBox.addItem("450")
        self.comboBox.addItem("500")
        self.verticalLayout_3.addWidget(self.comboBox)
        self.checkBox = QtWidgets.QCheckBox(Dialog)
        self.checkBox.setObjectName("checkBox")
        self.checkBox.setChecked(True)
        self.verticalLayout_3.addWidget(self.checkBox)
        self.horizontalLayout.addLayout(self.verticalLayout_3)
        self.horizontalLayout.setStretch(0, 2)
        self.horizontalLayout.setStretch(1, 1)
        self.horizontalLayout.setStretch(2, 3)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.buttonBox = QtWidgets.QDialogButtonBox(Dialog)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.verticalLayout.addWidget(self.buttonBox)
        self.gridLayout.addLayout(self.verticalLayout, 0, 0, 1, 1)

        self.retranslateUi(Dialog)
        self.buttonBox.accepted.connect(Dialog.accept)
        self.buttonBox.rejected.connect(Dialog.reject)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Settings"))
        self.label.setText(_translate("Dialog", "General settings:"))
        self.cem_res_setting.setText(_translate("Dialog", "Camera capture resolution:"))
        self.label_2.setText(_translate("Dialog", "Camera index (0 default):"))
        self.label_3.setText(_translate("Dialog", "Detection confidence threshold:"))
        self.label_4.setText(_translate("Dialog", "Sharpness threshold:"))
        self.label_5.setText(_translate("Dialog", "Save detection recording:"))
        self.radioButton_480p.setText(_translate("Dialog", "480p"))
        self.radioButton_576p.setText(_translate("Dialog", "576p"))
        self.radioButton_1080p.setText(_translate("Dialog", "1080p"))
        self.radioButton_720p.setText(_translate("Dialog", "720p"))
        self.comboBox_2.setItemText(0, _translate("Dialog", "90%"))
        self.comboBox_2.setItemText(1, _translate("Dialog", "91%"))
        self.comboBox_2.setItemText(2, _translate("Dialog", "92%"))
        self.comboBox_2.setItemText(3, _translate("Dialog", "93% (Default/Recommended)"))
        self.comboBox_2.setItemText(4, _translate("Dialog", "94%"))
        self.comboBox_2.setItemText(5, _translate("Dialog", "95%"))
        self.comboBox_2.setItemText(6, _translate("Dialog", "96%"))
        self.comboBox_2.setItemText(7, _translate("Dialog", "97%"))
        self.comboBox_2.setItemText(8, _translate("Dialog", "98%"))
        self.comboBox_2.setItemText(9, _translate("Dialog", "99%"))
        self.comboBox.setItemText(0, _translate("Dialog", "250"))
        self.comboBox.setItemText(1, _translate("Dialog", "300 (Default/Recommended)"))
        self.comboBox.setItemText(2, _translate("Dialog", "350"))
        self.comboBox.setItemText(3, _translate("Dialog", "400"))
        self.comboBox.setItemText(4, _translate("Dialog", "450"))
        self.comboBox.setItemText(5, _translate("Dialog", "500"))
        self.comboBox_2.setCurrentIndex(3)
        self.comboBox.setCurrentIndex(1)
        self.checkBox.setText(_translate("Dialog", "Save"))
    
    def get_inputs(self):
        if self.exec_() == QtWidgets.QDialog.Accepted:
            radio_buttons=[self.radioButton_480p, self.radioButton_576p, self.radioButton_1080p, self.radioButton_720p]
            for button in radio_buttons:
                if button.isChecked():
                    resolution =button.text()
                    break

            camera_index = self.spinBox.value()
            conf_thresh = self.comboBox_2.currentText()
            sharp_thresh = self.comboBox.currentText()
            save = self.checkBox.isChecked()
            return resolution, camera_index, conf_thresh, sharp_thresh, save
        else:
            return None, None, None, None, None 


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    ui = Ui_Dialog()
    ui.show()
    sys.exit(app.exec_())