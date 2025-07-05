# -*- coding: utf-8 -*-
"""
主界面模块
基于PyQt5的主要用户界面，提供数据爬取、文件操作、可视化等功能
"""

import shutil
import sys
from viwe import line1, line2, tu
from PyQt5.QtCore import QUrl, QPoint
from PyQt5 import QtWebEngineWidgets
from second import second_UI
from all_Thread import spider_Thread
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPixmap, QRegExpValidator, QPalette, QBrush, QMouseEvent
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QTableWidgetItem, QTableWidget, QMainWindow
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtCore import Qt

get_data = False
being_data = 0
filepath = None
view = False


class First_Form(object):
    """主界面表单类"""
    
    def setupUi(self, Form):
        """设置UI界面"""
        Form.setObjectName("Form")
        Form.resize(1490, 857)
        Form.setStyleSheet("#Form{\n"
                           "    ;\n"
                           "    ;\n"
                           "    border-image: url(:/111/photo/20220816203419_5b1c3.jpg);\n"
                           "}")
        
        # 设置各种UI控件
        self._setup_buttons(Form)
        self._setup_labels(Form)
        self._setup_inputs(Form)
        self._setup_table(Form)
        self._setup_browser(Form)
        
        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def _setup_buttons(self, Form):
        """设置按钮"""
        # 主要功能按钮
        self.pushButton_3 = QtWidgets.QPushButton(Form)
        self.pushButton_3.setGeometry(QtCore.QRect(10, 10, 111, 51))
        self.pushButton_3.setStyleSheet(self._get_button_style())
        self.pushButton_3.setObjectName("pushButton_3")
        
        # 其他按钮设置...
        
    def _setup_labels(self, Form):
        """设置标签"""
        self.label_3 = QtWidgets.QLabel(Form)
        self.label_3.setGeometry(QtCore.QRect(340, 10, 791, 151))
        self.label_3.setStyleSheet("image: url(:/111/photo/img1717558191173.png);")
        self.label_3.setText("")
        self.label_3.setObjectName("label_3")
        
    def _setup_inputs(self, Form):
        """设置输入控件"""
        self.lineEdit = QtWidgets.QLineEdit(Form)
        self.lineEdit.setGeometry(QtCore.QRect(519, 103, 101, 21))
        self.lineEdit.setStyleSheet("background-color: rgba(255, 255, 255, 100);")
        self.lineEdit.setObjectName("lineEdit")
        
    def _setup_table(self, Form):
        """设置表格"""
        self.tableWidget = QtWidgets.QTableWidget(Form)
        self.tableWidget.setGeometry(QtCore.QRect(90, 230, 1351, 591))
        self.tableWidget.setStyleSheet("background-color: rgba(255, 255, 255, 10);")
        self.tableWidget.setObjectName("tableWidget")
        self.tableWidget.setColumnCount(0)
        self.tableWidget.setRowCount(0)
        
    def _setup_browser(self, Form):
        """设置浏览器控件"""
        self.brower1 = QtWebEngineWidgets.QWebEngineView(Form)
        self.brower1.setGeometry(QtCore.QRect(0, 0, 0, 0))
        self.brower1.setObjectName("brower1")
        
    def _get_button_style(self):
        """获取按钮样式"""
        return """QPushButton{
            border-radius:15px;
            padding:2px 4px;
            background-color: rgba(0,191,255,200);
            color:white;
            min-width:20px;
            min-height:20px;
            font:bold 14px;
        }"""

    def retranslateUi(self, Form):
        """设置UI文本"""
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "情感分析系统"))
        self.pushButton_3.setText(_translate("Form", "数据爬取"))
        self.pushButton_4.setText(_translate("Form", "数据分析"))
        # 其他文本设置...
        
    def get_data(self):
        """获取数据"""
        global get_data, being_data
        keyword = str(self.lineEdit.text())
        num = str(self.lineEdit_2.text())
        
        if keyword and num:
            self.spider_thread = spider_Thread(keyword, num)
            self.spider_thread.spi_data.connect(self.additem)
            self.spider_thread.start()
            get_data = True
        else:
            QMessageBox.warning(None, "警告", "请输入关键词和数量！")
            
    def additem(self, list1):
        """添加数据到表格"""
        global being_data
        twicefive_data, current = list1[0], list1[1]
        being_data = len(twicefive_data)
        
        # 设置表格
        self.tableWidget.setRowCount(being_data)
        self.tableWidget.setColumnCount(7)
        self.tableWidget.setHorizontalHeaderLabels(
            ["用户", "IP", "话题", "内容", "点赞", "评论", "转发"]
        )
        
        # 填充数据
        for i, row_data in enumerate(twicefive_data):
            for j, cell_data in enumerate(row_data):
                item = QTableWidgetItem(str(cell_data))
                self.tableWidget.setItem(i, j, item)
                
    def openFile(self):
        """打开文件"""
        global filepath
        filepath, _ = QFileDialog.getOpenFileName(
            None, "选择文件", "", "CSV files (*.csv)"
        )
        if filepath:
            self.lineEdit_3.setText(filepath)
            
    def preserve_data(self):
        """保存数据"""
        if filepath:
            try:
                shutil.copy(filepath, 'data/data.csv')
                QMessageBox.information(None, "成功", "数据保存成功！")
            except Exception as e:
                QMessageBox.warning(None, "错误", f"保存失败：{str(e)}")
        else:
            QMessageBox.warning(None, "警告", "请先选择文件！")
            
    def preserve_fig(self):
        """保存可视化图表"""
        try:
            save_path, _ = QFileDialog.getSaveFileName(
                None, "保存图表", "", "HTML files (*.html)"
            )
            if save_path:
                shutil.copy('output.html', save_path)
                QMessageBox.information(None, "成功", "图表保存成功！")
        except Exception as e:
            QMessageBox.warning(None, "错误", f"保存失败：{str(e)}")
            
    def view_charts(self):
        """查看图表"""
        global view
        option = self.comboBox.currentText()
        
        try:
            if option == "地区分布":
                line1('data/data.csv')
            elif option == "活跃用户":
                line2('data/data.csv')
            elif option == "词云图":
                tu('data/data.csv')
                
            # 显示图表
            if not view:
                self.brower1.setGeometry(QtCore.QRect(500, 300, 800, 500))
                view = True
                
            url = QUrl.fromLocalFile(
                QtCore.QDir.current().absoluteFilePath("output.html")
            )
            self.brower1.load(url)
            
        except Exception as e:
            QMessageBox.warning(None, "错误", f"生成图表失败：{str(e)}")
            
    def Form2(self):
        """切换到第二个界面"""
        self.ui = second_UI()
        self.ui.show()
        

class Main_UI(QMainWindow, First_Form):
    """主界面窗口类"""
    
    def __init__(self):
        super(Main_UI, self).__init__()
        self.setupUi(self)
        self._connect_signals()
        
    def _connect_signals(self):
        """连接信号槽"""
        self.pushButton.clicked.connect(self.get_data)
        self.pushButton_3.clicked.connect(self.get_data)
        self.pushButton_4.clicked.connect(self.Form2)
        self.pushButton_7.clicked.connect(self.openFile)
        self.pushButton_5.clicked.connect(self.preserve_data)
        self.pushButton_6.clicked.connect(self.preserve_fig)
        self.comboBox.currentTextChanged.connect(self.view_charts)
        
        # 下拉框选项
        self.comboBox.addItems(["地区分布", "活跃用户", "词云图"])
        

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    ui = Main_UI()
    ui.show()
    sys.exit(app.exec_())