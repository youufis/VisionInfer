# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'd:\MyPython\VisionInfer\visioninfer.ui'
#
# Created by: PyQt5 UI code generator 5.15.1
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1024, 768)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.lblsrc = QtWidgets.QLabel(self.centralwidget)
        self.lblsrc.setText("")
        self.lblsrc.setObjectName("lblsrc")
        self.horizontalLayout_2.addWidget(self.lblsrc)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem)
        self.lbldst = QtWidgets.QLabel(self.centralwidget)
        self.lbldst.setText("")
        self.lbldst.setObjectName("lbldst")
        self.horizontalLayout_2.addWidget(self.lbldst)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        spacerItem1 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem1)
        self.txtmsg = QtWidgets.QTextEdit(self.centralwidget)
        self.txtmsg.setMaximumSize(QtCore.QSize(16777215, 50))
        self.txtmsg.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.txtmsg.setObjectName("txtmsg")
        self.verticalLayout.addWidget(self.txtmsg)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setMaximumSize(QtCore.QSize(100, 16777215))
        self.label_2.setObjectName("label_2")
        self.horizontalLayout_3.addWidget(self.label_2)
        self.rbtimg = QtWidgets.QRadioButton(self.centralwidget)
        self.rbtimg.setMaximumSize(QtCore.QSize(100, 16777215))
        self.rbtimg.setObjectName("rbtimg")
        self.horizontalLayout_3.addWidget(self.rbtimg)
        self.rbtvideo = QtWidgets.QRadioButton(self.centralwidget)
        self.rbtvideo.setMaximumSize(QtCore.QSize(100, 16777215))
        self.rbtvideo.setObjectName("rbtvideo")
        self.horizontalLayout_3.addWidget(self.rbtvideo)
        self.rbtcam = QtWidgets.QRadioButton(self.centralwidget)
        self.rbtcam.setMaximumSize(QtCore.QSize(100, 16777215))
        self.rbtcam.setObjectName("rbtcam")
        self.horizontalLayout_3.addWidget(self.rbtcam)
        self.rbtipcam = QtWidgets.QRadioButton(self.centralwidget)
        self.rbtipcam.setMaximumSize(QtCore.QSize(80, 16777215))
        self.rbtipcam.setObjectName("rbtipcam")
        self.horizontalLayout_3.addWidget(self.rbtipcam)
        self.gboxipcam = QtWidgets.QGroupBox(self.centralwidget)
        self.gboxipcam.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.gboxipcam.setTitle("")
        self.gboxipcam.setObjectName("gboxipcam")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.gboxipcam)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.label_3 = QtWidgets.QLabel(self.gboxipcam)
        self.label_3.setObjectName("label_3")
        self.horizontalLayout_4.addWidget(self.label_3)
        self.leip = QtWidgets.QLineEdit(self.gboxipcam)
        self.leip.setMaximumSize(QtCore.QSize(150, 16777215))
        self.leip.setInputMask("")
        self.leip.setObjectName("leip")
        self.horizontalLayout_4.addWidget(self.leip)
        self.label_4 = QtWidgets.QLabel(self.gboxipcam)
        self.label_4.setObjectName("label_4")
        self.horizontalLayout_4.addWidget(self.label_4)
        self.cboxcamid = QtWidgets.QComboBox(self.gboxipcam)
        self.cboxcamid.setObjectName("cboxcamid")
        self.horizontalLayout_4.addWidget(self.cboxcamid)
        self.label_5 = QtWidgets.QLabel(self.gboxipcam)
        self.label_5.setObjectName("label_5")
        self.horizontalLayout_4.addWidget(self.label_5)
        self.leuser = QtWidgets.QLineEdit(self.gboxipcam)
        self.leuser.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.leuser.setObjectName("leuser")
        self.horizontalLayout_4.addWidget(self.leuser)
        self.label_6 = QtWidgets.QLabel(self.gboxipcam)
        self.label_6.setObjectName("label_6")
        self.horizontalLayout_4.addWidget(self.label_6)
        self.lepwd = QtWidgets.QLineEdit(self.gboxipcam)
        self.lepwd.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.lepwd.setEchoMode(QtWidgets.QLineEdit.Password)
        self.lepwd.setObjectName("lepwd")
        self.horizontalLayout_4.addWidget(self.lepwd)
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_4.addItem(spacerItem2)
        self.horizontalLayout_3.addWidget(self.gboxipcam)
        self.verticalLayout.addLayout(self.horizontalLayout_3)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.cboxscaled = QtWidgets.QCheckBox(self.centralwidget)
        self.cboxscaled.setMaximumSize(QtCore.QSize(100, 16777215))
        self.cboxscaled.setObjectName("cboxscaled")
        self.horizontalLayout.addWidget(self.cboxscaled)
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setMaximumSize(QtCore.QSize(80, 16777215))
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.cboxtask = QtWidgets.QComboBox(self.centralwidget)
        self.cboxtask.setObjectName("cboxtask")
        self.horizontalLayout.addWidget(self.cboxtask)
        self.cboxmodel = QtWidgets.QComboBox(self.centralwidget)
        self.cboxmodel.setObjectName("cboxmodel")
        self.horizontalLayout.addWidget(self.cboxmodel)
        self.btnopen = QtWidgets.QPushButton(self.centralwidget)
        self.btnopen.setObjectName("btnopen")
        self.horizontalLayout.addWidget(self.btnopen)
        self.btnsnap = QtWidgets.QPushButton(self.centralwidget)
        self.btnsnap.setMaximumSize(QtCore.QSize(80, 16777215))
        self.btnsnap.setObjectName("btnsnap")
        self.horizontalLayout.addWidget(self.btnsnap)
        self.btnrec = QtWidgets.QPushButton(self.centralwidget)
        self.btnrec.setMaximumSize(QtCore.QSize(80, 16777215))
        self.btnrec.setObjectName("btnrec")
        self.horizontalLayout.addWidget(self.btnrec)
        self.btnclose = QtWidgets.QPushButton(self.centralwidget)
        self.btnclose.setObjectName("btnclose")
        self.horizontalLayout.addWidget(self.btnclose)
        self.verticalLayout.addLayout(self.horizontalLayout)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1024, 23))
        self.menubar.setObjectName("menubar")
        self.menuhelp = QtWidgets.QMenu(self.menubar)
        self.menuhelp.setObjectName("menuhelp")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.action = QtWidgets.QAction(MainWindow)
        self.action.setObjectName("action")
        self.action_2 = QtWidgets.QAction(MainWindow)
        self.action_2.setObjectName("action_2")
        self.menuhelp.addAction(self.action)
        self.menuhelp.addSeparator()
        self.menuhelp.addAction(self.action_2)
        self.menubar.addAction(self.menuhelp.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label_2.setText(_translate("MainWindow", "数据源："))
        self.rbtimg.setText(_translate("MainWindow", "图像文件"))
        self.rbtvideo.setText(_translate("MainWindow", "视频文件"))
        self.rbtcam.setText(_translate("MainWindow", "摄像头"))
        self.rbtipcam.setText(_translate("MainWindow", "录像机"))
        self.label_3.setText(_translate("MainWindow", "IP:"))
        self.leip.setPlaceholderText(_translate("MainWindow", "192.168.1.1"))
        self.label_4.setText(_translate("MainWindow", "通道："))
        self.label_5.setText(_translate("MainWindow", "帐号："))
        self.leuser.setPlaceholderText(_translate("MainWindow", "帐号"))
        self.label_6.setText(_translate("MainWindow", "密码："))
        self.lepwd.setPlaceholderText(_translate("MainWindow", "密码"))
        self.cboxscaled.setText(_translate("MainWindow", "图像缩放"))
        self.label.setText(_translate("MainWindow", "任务模型："))
        self.btnopen.setText(_translate("MainWindow", "预览/预测"))
        self.btnsnap.setText(_translate("MainWindow", "拍照"))
        self.btnrec.setText(_translate("MainWindow", "开始录制"))
        self.btnclose.setText(_translate("MainWindow", "关闭"))
        self.menuhelp.setTitle(_translate("MainWindow", "帮助"))
        self.action.setText(_translate("MainWindow", "说明"))
        self.action_2.setText(_translate("MainWindow", "关于"))
import xml_rc
