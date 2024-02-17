from PyQt5.QtWidgets import QApplication, QComboBox, QWidget, QVBoxLayout
import sys

class Example(QWidget):

    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):

        self.cb = QComboBox(self)
        self.cb.addItem('Option 1')
        self.cb.addItem('Option 2')
        self.cb.addItem('Option 3')
        self.cb.addItem('Option 4')
        self.cb.currentIndexChanged.connect(self.selectionchange)

        vbox = QVBoxLayout()
        vbox.addWidget(self.cb)

        self.setLayout(vbox)

        self.setGeometry(300, 300, 300, 200)
        self.setWindowTitle('QComboBox')
        self.show()

    def selectionchange(self, i):
        print('Selection changed to:', self.cb.currentText())

if __name__ == '__main__':

    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())
