import sys

from new_main import First_Form
from PyQt5.QtWidgets import QMainWindow, QApplication


class MyWindow(QMainWindow, First_Form):
    def __init__(self):
        super(MyWindow, self).__init__()
        self.setupUi(self)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    myWin = MyWindow()

    myWin.show()
    sys.exit(app.exec_())
