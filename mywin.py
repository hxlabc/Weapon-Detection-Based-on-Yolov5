import shutil
import sys
import threading
from pathlib import Path

import cv2
from PyQt5.QtGui import QPixmap
from pyqt5_plugins.examplebuttonplugin import QtGui

from torch.backends import cudnn

from mywindow import res_rc
import torch
import os.path as osp

from PyQt5.QtWidgets import QApplication, QFileDialog, QMessageBox
from PyQt5.QtWidgets import QMainWindow
from pyqt5_tools.examples.exampleqmlitem import QtCore


from models.common import DetectMultiBackend
from mywindow.Wea_window import Ui_MainWindow
from utils.dataloaders import LoadStreams, LoadImages
from utils.general import check_img_size, check_imshow, non_max_suppression, scale_boxes, xyxy2xywh, LOGGER
from utils.plots import Annotator, colors
from utils.torch_utils import select_device, time_sync


class UsingTest(QMainWindow, Ui_MainWindow):
    def __init__(self, *args, **kwargs):
        super(UsingTest, self).__init__(*args, **kwargs)
        # 图片读取进程
        self.output_size = 480
        self.img2predict = ""
        self.device = 'cpu'
        # # 初始化视频读取线程
        self.vid_source = '0'  # 初始设置为摄像头
        self.stopEvent = threading.Event()
        self.webcam = True
        self.stopEvent.clear()
        self.model = self.model_load(weights="weights/MSPPF_NAM_yolov5m.pt",
                                     device=self.device)  # todo 指明模型加载的位置的设备

        # 无视原始边框
        self.setWindowFlag(QtCore.Qt.FramelessWindowHint)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)

        # 初始化ui
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # 定义按钮的事件
        self.ui.Button_pic.clicked.connect(lambda: self.ui.stackedWidget.setCurrentIndex(0))
        self.ui.Button_video.clicked.connect(lambda: self.ui.stackedWidget.setCurrentIndex(1))
        self.ui.Button_camera.clicked.connect(lambda: self.ui.stackedWidget.setCurrentIndex(2))

        self.ui.Button_upload.clicked.connect(self.upload_img)
        self.ui.Button_detect.clicked.connect(self.detect_img)
        self.ui.Button_opencam.clicked.connect(self.open_cam)
        self.ui.Button_openvid.clicked.connect(self.open_mp4)
        self.ui.Button_stopc.clicked.connect(self.close_vid)
        self.ui.Button_stopv.clicked.connect(self.close_vid)

        # Conf与IoU的调节
        self.ui.ConfSpinBox.setRange(0, 1)
        self.ui.ConfSpinBox.setSingleStep(0.01)
        self.ui.IouSpinBox.setRange(0, 1)
        self.ui.IouSpinBox.setSingleStep(0.01)

        self.ui.ConfSlider.valueChanged.connect(self.update_Conf)
        self.ui.IouSlider.valueChanged.connect(self.update_Iou)

        # 更换模型
        self.ui.comboBox.currentIndexChanged.connect(self.selectionchange)

    def selectionchange(self, i):
        model_name = self.ui.comboBox.currentText()
        self.model = self.model_load(weights="weights/" + model_name,
                                     device=self.device)  # todo 指明模型加载的位置的设备

    def update_Conf(self, value):
        self.ui.ConfSpinBox.setValue(value / 100.0)

    def update_Iou(self, value):
        self.ui.IouSpinBox.setValue(value / 100.0)

    # 拖动窗口
    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton and self.isMaximized() == False:
            self.m_flag = True
            self.m_Position = event.globalPos() - self.pos()  # 获取鼠标相对窗口的位置
            event.accept()
            self.setCursor(QtGui.QCursor(QtCore.Qt.OpenHandCursor))

    def mouseMoveEvent(self, mouse_event):
        if QtCore.Qt.LeftButton and self.m_flag:
            self.move(mouse_event.globalPos() - self.m_Position)
            mouse_event.accept()

    def mouseReleaseEvent(self, mouse_event):
        self.m_flag = False
        self.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))

    '''
    ***模型初始化***
    '''
    @torch.no_grad()
    def model_load(self, weights="",  # model.pt path(s)
                   device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
                   half=False,  # use FP16 half-precision inference
                   dnn=False,  # use OpenCV DNN for ONNX inference
                   ):
        device = select_device(device)
        # 加载模型
        model = DetectMultiBackend(weights, device=device, dnn=dnn)
        stride, names, pt, jit, onnx = model.stride, model.names, model.pt, model.jit, model.onnx  # names表示标签
        half &= pt and device.type != 'cpu'    # 这里pt表示pytorch，值为True
        if pt:
            model.model.half() if half else model.model.float()
        print("模型加载完成!")
        return model

    '''
    ***上传图片***
    '''
    def upload_img(self):
        # 选择录像文件进行读取
        fileName, fileType = QFileDialog.getOpenFileName(self, 'Choose file', '', '*.jpg *.png *.tif *.jpeg')
        if fileName:
            im0 = cv2.imread(fileName)
            # 调整图片的大小 按长边进行缩放
            resize_scale = min(self.output_size / im0.shape[0], self.output_size / im0.shape[1])
            im0 = cv2.resize(im0, (0, 0), fx=resize_scale, fy=resize_scale)
            cv2.imwrite("images/tmp/upload_show_result.jpg", im0)
            self.img2predict = fileName
            self.ui.pic_org.setPixmap(QPixmap("images/tmp/upload_show_result.jpg"))
            self.ui.show_result.setPlainText('')

    '''
    ***检测图片***
    '''
    def detect_img(self):
        model = self.model
        output_size = self.output_size
        source = self.img2predict  # file/dir/URL/glob, 0 for webcam
        imgsz = [640, 640]  # inference size (pixels)
        # conf_thres = 0.25  # confidence threshold
        # iou_thres = 0.45  # NMS IOU threshold

        conf_thres = self.ui.ConfSpinBox.value()
        iou_thres = self.ui.IouSpinBox.value()

        max_det = 1000  # maximum detections per image
        device = self.device  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img = False  # show results
        save_crop = False  # save cropped prediction boxes
        nosave = False  # do not save images/videos
        classes = None  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms = False  # class-agnostic NMS
        augment = False  # ugmented inference
        visualize = False  # visualize features
        line_thickness = 3  # bounding box thickness (pixels)
        hide_labels = False  # hide labels
        hide_conf = False  # hide confidences
        half = False  # use FP16 half-precision inference
        print(source)
        if source == "":
            QMessageBox.warning(self, "请上传", "请先上传图片再进行检测")
        else:
            source = str(source)
            device = select_device(self.device)
            stride, names, pt, jit, onnx = model.stride, model.names, model.pt, model.jit, model.onnx
            imgsz = check_img_size(imgsz, s=stride)  # check image size
            save_img = not nosave and not source.endswith('.txt')  # save inference images
            # Dataloader
            dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt and not jit)

            # Run inference
            if pt and device.type != 'cpu':        # *imgsz带一个星号, 表示输入的是元组
                model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))  # warmup
            dt, seen = [0.0, 0.0, 0.0], 0
            for path, im, im0s, vid_cap, s in dataset:
                t1 = time_sync()
                # from_numpy将np数组转换为张量tensor
                # 将所有最开始读取数据时的tensor变量copy一份到device所指定的GPU(CPU)上去，之后的运算都在GPU(CPU)上进行
                im = torch.from_numpy(im).to(device)
                im = im.half() if half else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim
                t2 = time_sync()
                dt[0] += t2 - t1
                # Inference
                # visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
                # visualize表示在模型推断的过程中把中间图片给保存下来
                # augment表示模型推断时用数据增强
                # pred返回的是检测框
                pred = model(im, augment=augment, visualize=visualize)
                t3 = time_sync()
                dt[1] += t3 - t2
                # NMS    过滤无用的检测框
                pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
                dt[2] += time_sync() - t3

                # Process predictions
                for i, det in enumerate(pred):  # per image
                    seen += 1
                    p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
                    s += '%gx%g ' % im.shape[2:]  # print string
                    annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                    if len(det):
                        # print("检测框：", det)
                        # Rescale boxes from img_size to im0 size  坐标映射回原图
                        det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                        # Print results
                        for c in det[:, -1].unique():
                            n = (det[:, -1] == c).sum()  # detections per class
                            s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                        showrec = ''
                        # Write results   *xyxy框的坐标   conf置信度   cls类别
                        for *xyxy, conf, cls in reversed(det):
                            showrec += names[int(cls)] + ':' + str(conf.numpy()) + '\n'
                            # print(conf, names[int(cls)])
                            if save_img or save_crop or view_img:  # Add bbox to image
                                c = int(cls)  # integer class
                                label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                                annotator.box_label(xyxy, label, color=colors(c, True))

                        # 显示检测结果
                        self.ui.show_result.setPlainText(showrec)

                    LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')
                    # Stream results
                    im0 = annotator.result()
                    resize_scale = min(output_size / im0.shape[0], output_size / im0.shape[1])
                    im0 = cv2.resize(im0, (0, 0), fx=resize_scale, fy=resize_scale)
                    cv2.imwrite("images/tmp/single_result.jpg", im0)
                    self.ui.pic_detect.setPixmap(QPixmap("images/tmp/single_result.jpg"))

    '''
    ### 视频关闭事件 ### 
    '''
    def open_cam(self):
        self.ui.Button_opencam.setEnabled(False)
        self.ui.Button_stopc.setEnabled(True)
        self.vid_source = '0'
        self.webcam = True
        self.ui.show_result.setPlainText('')
        th = threading.Thread(target=self.detect_vid)
        th.start()

    '''
    ### 开启视频文件检测事件 ### 
    '''
    def open_mp4(self):
        fileName, fileType = QFileDialog.getOpenFileName(self, 'Choose file', '', '*.mp4 *.avi')
        if fileName:
            self.ui.Button_openvid.setEnabled(False)
            self.ui.Button_stopv.setEnabled(True)
            self.vid_source = fileName
            self.webcam = False
            self.ui.show_result.setPlainText('')
            th = threading.Thread(target=self.detect_vid)
            th.start()

    '''
    ### 视频开启事件 ### 
    '''
    def detect_vid(self):
        model = self.model
        output_size = self.output_size + 200
        imgsz = [640, 640]  # inference size (pixels)
        # conf_thres = 0.25  # confidence threshold
        # iou_thres = 0.45  # NMS IOU threshold
        max_det = 1000  # maximum detections per image
        view_img = False  # show results
        save_crop = False  # save cropped prediction boxes
        nosave = False  # do not save images/videos
        classes = None  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms = False  # class-agnostic NMS
        augment = False  # ugmented inference
        visualize = False  # visualize features
        line_thickness = 3  # bounding box thickness (pixels)
        hide_labels = False  # hide labels
        hide_conf = False  # hide confidences
        half = False  # use FP16 half-precision inference
        source = str(self.vid_source)
        webcam = self.webcam
        device = select_device(self.device)
        stride, names, pt, jit, onnx = model.stride, model.names, model.pt, model.jit, model.onnx
        imgsz = check_img_size(imgsz, s=stride)  # check image size
        save_img = not nosave and not source.endswith('.txt')  # save inference images
        # Dataloader
        if webcam:
            view_img = check_imshow()
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt and not jit)
        else:
            dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt and not jit)
        # Run inference
        if pt and device.type != 'cpu':
            model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))  # warmup
        dt, seen = [0.0, 0.0, 0.0], 0
        for path, im, im0s, vid_cap, s in dataset:
            t1 = time_sync()
            im = torch.from_numpy(im).to(device)
            im = im.half() if half else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            t2 = time_sync()
            dt[0] += t2 - t1
            # Inference
            pred = model(im, augment=augment, visualize=visualize)
            t3 = time_sync()
            dt[1] += t3 - t2
            # NMS
            pred = non_max_suppression(pred, self.ui.ConfSpinBox.value(), self.ui.IouSpinBox.value(), classes, agnostic_nms, max_det=max_det)
            dt[2] += time_sync() - t3
            # Second-stage classifier (optional)
            # Process predictions
            for i, det in enumerate(pred):  # per image
                seen += 1
                if webcam:  # batch_size >= 1
                    p, im0, frame = path[i], im0s[i].copy(), dataset.count
                    s += f'{i}: '
                else:
                    p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
                s += '%gx%g ' % im.shape[2:]  # print string
                annotator = Annotator(im0, line_width=line_thickness, example=str(names))

                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    showrec = ''

                    for *xyxy, conf, cls in reversed(det):
                        showrec += names[int(cls)] + ':' + str(conf.numpy()) + '\n'
                        if save_img or save_crop or view_img:  # Add bbox to image
                            c = int(cls)  # integer class
                            label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                            annotator.box_label(xyxy, label, color=colors(c, True))

                    # self.ui.show_result.setPlainText(showrec)

                LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')
                # Stream results
                # Save results (image with detections)
                im0 = annotator.result()
                frame = im0
                resize_scale = min(output_size / im0.shape[0], output_size / im0.shape[1])
                frame_resized = cv2.resize(frame, (0, 0), fx=resize_scale, fy=resize_scale)
                cv2.imwrite("images/tmp/single_result_vid.jpg", frame_resized)
                if webcam:
                    self.ui.cam_detect.setPixmap(QPixmap("images/tmp/single_result_vid.jpg"))
                else:
                    self.ui.vid_detect.setPixmap(QPixmap("images/tmp/single_result_vid.jpg"))

            if cv2.waitKey(25) & self.stopEvent.is_set() == True:
                self.stopEvent.clear()
                self.ui.Button_openvid.setEnabled(True)
                self.ui.Button_opencam.setEnabled(True)
                self.reset_vid()
                break

    '''
    ### 界面重置事件 ### 
    '''
    def reset_vid(self):
        self.ui.Button_opencam.setEnabled(True)
        self.ui.Button_openvid.setEnabled(True)
        self.ui.vid_detect.setPixmap(QPixmap(""))
        self.ui.cam_detect.setPixmap(QPixmap(""))
        self.vid_source = '0'
        self.webcam = True

    '''
    ### 视频重置事件 ### 
    '''
    def close_vid(self):
        self.stopEvent.set()
        self.reset_vid()


if __name__ == '__main__':  # 程序的入口
    app = QApplication(sys.argv)
    win = UsingTest()
    win.show()
    win.reset_vid()
    sys.exit(app.exec_())
