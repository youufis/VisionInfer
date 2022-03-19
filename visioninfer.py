import sys
from PyQt5 import QtCore, QtGui,QtWidgets,QtNetwork
from Ui_visioninfer import Ui_MainWindow
import cv2
import os
import datetime
import xml_rc
import paddlex as pdx 
 
class MainWindow(QtWidgets.QMainWindow,Ui_MainWindow):
    isstop,issnap,isrec=False,False,False  #中断执行标记 #拍照标记 #录制标记 
    isscaled,iscam,isimg,isipcam,isvideo=False,True,False,False,False #勾选标记
    ip,camid,user,pwd="","","","" #录像机登录信息
    cap=cv2.VideoCapture()
    Video=cv2.VideoWriter()

    
    def __init__(self):
        super().__init__()
        self.setupUi(self)        
        #状态栏
        self.statusBar().showMessage("可视化推理预测系统")
        #默认勾选摄像头
        self.rbtcam.setChecked(True)       
        
        
        #槽信号
        self.btnclose.clicked.connect(self.Close)
        self.btnopen.clicked.connect(self.Open)
        self.cboxtask.currentIndexChanged.connect(self.schange)
        self.menuhelp.triggered.connect(self.winaction)
        self.cboxcamid.currentIndexChanged.connect(self.changecamid)
        self.btnsnap.clicked.connect(self.snap)
        self.btnrec.clicked.connect(self.rec)
        
        
        self.cboxtask.addItems(["请选择模型"])        
        if os.path.exists("models"):
            lst=os.listdir("models")
            self.cboxtask.addItems(lst)
        else:
            os.makedirs("models")
            os.makedirs("models/cls-model")
            os.makedirs("models/det-model")
            os.makedirs("models/face-model")
        #默认64通道数
        self.cboxcamid.addItems([str(id).rjust(2,"0") for id in range(1,65)])
        #释放资源文件
        if  not os.path.exists("haarcascade_frontalface_alt.xml"):
            QtCore.QFile.copy(":harr/haarcascade_frontalface_alt.xml","haarcascade_frontalface_alt.xml")
            
        os.environ["CUDA_VISIBLE_DEVICES"]="0,1" 
    
    def winaction(self,action):
        q=action.text()
        if q=="说明":
            QtWidgets.QMessageBox.information(self,
                "程序说明：",
                "1、程序主要实现对推理模型可视化预测，任务模型分三大类：人脸识别、图像分类、目标检测。\n\n\
2、存放任务模型默认目录分别是：cls-model(图像分类模型)、det-model(目标检测模型)、face-model(人脸分类模型),默认目录名不能更改\n\n\
3、任务模型默认目录里可以增加用户自己的模型目录，但模型目录里的inference_model默认目录名不能更改，用来存放用户自已的推理模型文件\n")
        if q=="关于":
            img=QtGui.QImage(":img/start.jpg")    
            self.lblsrc.setPixmap(QtGui.QPixmap.fromImage(img).scaled(1024,480 , QtCore.Qt.KeepAspectRatio))
            self.lblsrc.setScaledContents(True) #自适应大小
            QtWidgets.QMessageBox.about(self, "关于", "可视化推理预测系统")     
                    
            
                    
    def changecamid(self,i):
        if self.rbtipcam.isChecked():
            self.Open()
        
    def schange(self,i):
        #print(i,self.cboxtask.itemText(i)) 
        self.cboxmodel.clear()
        try:
            lst=os.listdir(os.path.join("models",self.cboxtask.itemText(i)))
            self.cboxmodel.addItems(lst)
        except:
            pass
    
    #获取录像机的信息
    def getipcam(self):
        self.ip=self.leip.displayText()
        self.camid=self.cboxcamid.currentText()
        self.user=self.leuser.displayText()
        self.lepwd.setEchoMode(self.lepwd.Normal)
        self.pwd=self.lepwd.displayText() #获取密码
        self.lepwd.setEchoMode(self.lepwd.Password)
        url="rtsp://"+self.user+":"+self.pwd+"@"+self.ip+"/Streaming/Channels/"+self.camid+"01?transportmode=multicas"
        ret=cv2.VideoCapture(url)
        if ret.grab():
            return url
        else:
            self.rbtcam.setChecked(True)
            return 0
    
    #获取录像机的通道数
    def getcamid(self):
        self.ip=self.leip.displayText()      
        self.user=self.leuser.displayText()
        self.lepwd.setEchoMode(self.lepwd.Normal)
        self.pwd=self.lepwd.displayText() #获取密码
        self.lepwd.setEchoMode(self.lepwd.Password)
        camidlst=[]
        for camid in range(64):
            url="rtsp://"+self.user+":"+self.pwd+"@"+self.ip+"：554/Streaming/Channels/"+str(camid).rjust(2,"0")+"01?transportmode=multicas"
            ret=cv2.VideoCapture(url)
            grabbed=ret.grab()
            if grabbed:
                camidlst.append(str(camid).rjust(2,"0"))
            QtWidgets.QApplication.processEvents()
        return camidlst
     
    #开启拍照或录制 
    def snap(self):
        if self.cap.isOpened():
            self.issnap=True
    def rec(self):
        if self.cap.isOpened():
            self.isrec =not self.isrec
            if self.isrec:
                #创建video writer
                ret,frame=self.cap.read()
                self.createVideo(frame)
                self.btnrec.setText("停止录制")    
                self.statusBar().showMessage("文件录制中…………\n")                    
            else:
                self.video.release() #释放video writer
                self.btnrec.setText("开始录制") 
                self.statusBar().showMessage("文件保存在output目录里\n")     
        
    #摄像头或监控拍照
    def imgsave(self,frame):
        if self.rbtcam.isChecked() or self.rbtipcam.isChecked():
            fname=datetime.datetime.now().strftime("%Y%m%d%H%M%S")+".jpg"
            if not os.path.exists("output"):
                os.makedirs("output")                
            cv2.imwrite(os.path.join("output",fname),frame)
            self.statusBar().showMessage(fname+"文件保存在output目录里\n")
            self.issnap=False            
    #摄像头或监控录制            
    def createVideo(self,frame):
        if self.rbtcam.isChecked() or self.rbtipcam.isChecked():
            fname=datetime.datetime.now().strftime("%Y%m%d%H%M%S")+".mp4"
            if not os.path.exists("output"):
                os.makedirs("output")
            self.video=cv2.VideoWriter(
                filename=os.path.join("output",fname),
                fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
                fps=15,
                frameSize=(frame.shape[1],frame.shape[0])
                ) 
            #video.write(frame)                    
            #video.release()
                
    
    #在lable上显示图像 
    def imgshow(self,frame,lbl):
         #img=QtGui.QImage(frame.data,frame.shape[1],frame.shape[0],QtGui.QImage.Format_BGR888)    
        #重载修复图像显示变形问题      
        img = QtGui.QImage(frame.data, frame.shape[1], frame.shape[0], frame.shape[1] * 3,QtGui.QImage.Format_BGR888)
        #按比例缩放
        if self.isscaled:  
            lbl.setPixmap(QtGui.QPixmap.fromImage(img).scaled(480, 480, QtCore.Qt.KeepAspectRatio))
        else:
            lbl.setPixmap(QtGui.QPixmap.fromImage(img))
            lbl.setScaledContents(True) #自适应大小
        QtWidgets.QApplication.processEvents()
            
    #打开输入数据类型
    def Open(self):        
        self.Close()
        self.isstop=False      
        self.isscaled=self.cboxscaled.isChecked()
        #获取选项按钮状态
        self.iscam=self.rbtcam.isChecked()
        self.isimg=self.rbtimg.isChecked()
        self.isvideo=self.rbtvideo.isChecked()
        self.isipcam=self.rbtipcam.isChecked()
        
        taskid=self.cboxtask.currentIndex()
        task=self.cboxtask.currentText()
        model_dir=os.path.join("models",self.cboxtask.currentText(),self.cboxmodel.currentText(),"inference_model")
        #print(model_dir) 
        #图像缩放
        self.isscaled=self.cboxscaled.isChecked()                         
        
        if taskid==0:#预览
            if self.isimg:
                self.fileName, self.fileType = QtWidgets.QFileDialog.getOpenFileName(self, '选择','', "图像文件(*.jpg *.png)")
                if self.fileName!="":
                    frame=cv2.imread(self.fileName)                            
                    self.imgshow(frame,self.lblsrc)            
            else:
                 self.Display()
        else:#预测
            if self.iscam or self.isipcam:
                if task in ["face-model"] :
                    self.cls_videodetect(model_dir,isface=True)
                if task in ["cls-model"]:
                    self.cls_videodetect(model_dir,isface=False)
                if  task in ["det-model"]:
                    self.det_videodetect(model_dir)
                
            if self.isimg:
                self.fileName, self.fileType = QtWidgets.QFileDialog.getOpenFileName(self, '选择','', "图像文件(*.jpg *.png)")
                if self.fileName!="":
                    frame=cv2.imread(self.fileName)                            
                    self.imgshow(frame,self.lblsrc)            
                
                #print(task)
                if task in ["face-model"]:#调用图像分类-人脸检测与识别任务,结果显示在dst上                
                    self.cls_imgpredict(model_dir,self.fileName,isface=True)#预测返回结果
                                        
                if task in ["cls-model"]: #调用图像分类-猫狗分类
                    self.cls_imgpredict(model_dir,self.fileName,isface=False)#预测返回结果
                
                if  task in ["det-model"]:#调用目标检测-人头检测
                    self.det_imgpredict(model_dir,self.fileName)
                                                                                
            if self.isvideo:
                self.fileName, self.fileType = QtWidgets.QFileDialog.getOpenFileName(self, '选择视频文件','', '*.mp4')
                if self.fileName!="":
                    if task in ["face-model"] :
                        self.cls_videodetect(model_dir,self.fileName,isface=True)
                    if task in ["cls-model"]:
                        self.cls_videodetect(model_dir,self.fileName,isface=False)
                    if  task in ["det-model"]:
                        self.det_videodetect(model_dir,self.fileName)
            
    #显示数据            
    def Display(self):
        isscaled=self.cboxscaled.isChecked()
        if self.isipcam:
            self.cap=cv2.VideoCapture(self.getipcam())
        if self.iscam:
            self.cap = cv2.VideoCapture(0)
        if self.isvideo:
            self.fileName, self.fileType = QtWidgets.QFileDialog.getOpenFileName(self, '选择视频文件','', '*.mp4')
            self.cap=cv2.VideoCapture(self.fileName)
        while self.cap.isOpened():
            ret,frame=self.cap.read()
            if not ret:
                break            
            if self.issnap:
                self.imgsave(frame)
            if self.isrec:#开始录制
                self.video.write(frame)              
                                                     
            if self.rbtipcam.isChecked():
                (h,w)= frame.shape[:-1] #获取图片大小
                frame=cv2.resize(frame, (int(w/2), int(h/2)))#缩小图像
            self.imgshow(frame,self.lblsrc)    
            cv2.waitKey(0)
            if self.isstop:
                if self.isrec:
                    self.video.release()
                    self.btnrec.setText("开始录制")
                    self.statusBar().showMessage("文件保存在output目录里\n") 
                    self.isrec=False
                
                self.cap.release()
                break
        self.cap.release()
    #关闭显示   
    def Close(self):             
        if self.rbtcam.isChecked() or self.rbtvideo.isChecked() or self.rbtipcam.isChecked():
            self.isstop=True 
        try:
            self.cap.release()    
        except:
            pass        
        self.lblsrc.clear()
        self.lbldst.clear()
        self.resize(1024,768) 
                  
    #图像文件分类检测（人脸检测与识别）    
    def cls_imgpredict(self,model_dir,imgfile:str,isface=True):
        #创建推理
        self.predictor=pdx.deploy.Predictor(model_dir,use_gpu=True)
        if isface:
            #加载人脸检测分类器
            face_detector = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
            img=cv2.imread(imgfile)
            gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            faces=face_detector.detectMultiScale(gray,1.3,5)
            # 如果有检测有结果，画框        
            if len(faces)>0:
                for (x,y,w,h) in faces:
                    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),1)
                    faceimg=img[y:y+h,x:x+w] #人脸图像
                    res=self.predictor.predict(faceimg) #人脸识别
                    if res:
                        score="score:"+str(round(res[0]["score"],2))
                        category="Faceid:"+res[0]["category"]
                        cv2.putText(img,category+" "+score,(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,0),1)            
        else:
            img=cv2.imread(imgfile)
            result=self.predictor.predict(img)
            cv2.putText(img,result[0]["category"]+":"+str(round(result[0]["score"],2)),(10,20),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),1)
            
        self.imgshow(img,self.lbldst) 

    #图像分类-视频-人脸检测并识别
    def cls_videodetect(self,model_dir,*videofile:str,isface=True):        
        self.isscaled=self.cboxscaled.isChecked()
        self.predictor=pdx.deploy.Predictor(model_dir,use_gpu=True)
        if isface:
            #加载人脸检测分类器
            face_detector = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml') 
            if  videofile:
                self.cap=cv2.VideoCapture(videofile[0])
            else:
                if self.isipcam:
                    self.cap=cv2.VideoCapture(self.getipcam())
                if self.iscam:
                    self.cap = cv2.VideoCapture(0)
            while self.cap.isOpened:
                #读取数据
                ret,frame=self.cap.read()
                if not ret:
                    break
                gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                if self.rbtipcam.isChecked():
                    (h,w)= frame.shape[:-1] #获取图片大小
                    frame=cv2.resize(frame, (int(w/2), int(h/2)))#缩小图像                
                self.imgshow(frame,self.lblsrc)
                faces=face_detector.detectMultiScale(gray,1.3,5)
                # 如果有检测有结果，画框        
                if len(faces)>0:
                    for (x,y,w,h) in faces:
                        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),1)
                        faceimg=frame[y:y+h,x:x+w] #人脸图像
                        res=self.predictor.predict(faceimg) #人脸识别
                        if res:
                            score="score:"+str(round(res[0]["score"],2))
                            category="Faceid:"+res[0]["category"]
                            cv2.putText(frame,category+" "+score,(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,0),1)
                self.imgshow(frame,self.lbldst)               
                cv2.waitKey(15)
                #
                if self.isstop:
                    break
            self.cap.release()
                
        else:#不是人脸检测识别，是普通图像分类
            if  videofile:
                self.cap=cv2.VideoCapture(videofile[0])
            else:
                if self.isipcam:
                    self.cap=cv2.VideoCapture(self.getipcam())
                if self.iscam:
                    self.cap = cv2.VideoCapture(0)
            while self.cap.isOpened:
                #读取数据
                ret,frame=self.cap.read()
                if not ret:
                    break
                if self.rbtipcam.isChecked():
                    (h,w)= frame.shape[:-1] #获取图片大小
                    frame=cv2.resize(frame, (int(w/2), int(h/2)))#缩小图像
                self.imgshow(frame,self.lblsrc)

                res=self.predictor.predict(frame)
                if res:
                    score="score:"+str(round(res[0]["score"],2))
                    category="category:"+res[0]["category"]
                    cv2.putText(frame,category+" "+score,(20,30),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,0),1)
                self.imgshow(frame,self.lbldst)                   
                cv2.waitKey(15)
                #
                if self.isstop:
                    break
            self.cap.release()
    #目标检测-图象
    def det_imgpredict(self,model_dir,imgfile:str):
        self.predictor=pdx.deploy.Predictor(model_dir,use_gpu=True)
        img=cv2.imread(imgfile)   
        result=self.predictor.predict(img)
        #print(result)
        vis_img=pdx.det.visualize(img,result,threshold=0.5,save_dir=None)
        self.imgshow(vis_img,self.lbldst)
    
    #目标检测-视频 
    def det_videodetect(self,model_dir,*videofile:str):
        self.isscaled=self.cboxscaled.isChecked()
        self.predictor=pdx.deploy.Predictor(model_dir,use_gpu=True)
        if  videofile:
            self.cap=cv2.VideoCapture(videofile[0])
        else:
            if self.isipcam:
                self.cap=cv2.VideoCapture(self.getipcam())
            if self.iscam:
                self.cap = cv2.VideoCapture(0)
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if self.rbtipcam.isChecked():
                (h,w)= frame.shape[:-1] #获取图片大小
                frame=cv2.resize(frame, (int(w/2), int(h/2)))#缩小图像
            self.imgshow(frame,self.lblsrc)
            if ret:
                result = self.predictor.predict(frame)
                if result:
                    score = result[0]['score']
                    if score >= 0.5:
                        pass
                    frame = pdx.det.visualize(frame, result, threshold=0.5, save_dir=None)
                self.imgshow(frame,self.lbldst)
            else:
                break            
            cv2.waitKey(15)
            if self.isstop:
                break
        self.cap.release()
        
if __name__ == '__main__':
    try:
        app=QtWidgets.QApplication(sys.argv)        
        app.processEvents()
        serverName="AppServer"
        socket=QtNetwork.QLocalSocket()
        socket.connectToServer(serverName)
        #防止程序实例重复启动
        if socket.waitForConnected(500):
            app.quit()
        else:
            localServer=QtNetwork.QLocalServer()
            localServer.listen(serverName)
            splash = QtWidgets.QSplashScreen(QtGui.QPixmap(':img/start.jpg'))
            splash.show()
            QtWidgets.QApplication.processEvents()
            # 可以显示启动信息
            splash.showMessage('正在加载……')
            # 关闭启动画面
            splash.close() 
            mywin=MainWindow()
            mywin.setWindowTitle("可视化推理预测系统")            
            mywin.show()                     
            sys.exit(app.exec())
             
    except:
        pass
