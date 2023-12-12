import numpy as np
import matplotlib.pyplot as plt
import os

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
from PyQt5.QtCore import pyqtSignal, pyqtSlot
from pyqtgraph.dockarea import Dock, DockArea

from skimage import io
from skimage.feature import peak_local_max 
from tkinter import Tk, filedialog
from scipy import signal

import viewbox_tools


class Frontend(QtGui.QFrame):
    
    lineparametersSignal = pyqtSignal(int, int, int, bool, int, int)
    update_image_Signal =  pyqtSignal()
    
    live_LINE_Signal = pyqtSignal(bool)
    maximum_Signal = pyqtSignal(np.ndarray, np.ndarray, float, int)
     
    savespectrumSignal = pyqtSignal(list, int, int, int)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.setUpGUI()

    def setUpGUI(self): 
        

        self.import_image_button = QtGui.QPushButton('Import Image')
        self.import_image_button.setCheckable(True)
        self.import_image_button.clicked.connect(self.import_image)
        self.import_image_button.setStyleSheet(
                "QPushButton { background-color: yellow; }"
                "QPushButton:pressed { background-color: blue; }")

        self.lineprofile_button = QtGui.QPushButton('Find maximum of NPs with Vertical LINE')
        self.lineprofile_button.setCheckable(True)
        self.lineprofile_button.clicked.connect(self.create_line_profile)
        self.lineprofile_button.setStyleSheet(
                "QPushButton:pressed { background-color: blue; }")

        threshold_rel_Label = QtGui.QLabel('Threshold relatived:')
        self.threshold_rel_Edit = QtGui.QLineEdit('0.17')
        self.threshold_rel_Edit.textChanged.connect(self.get_peak_local_maximum)

        minimum_distance_Label = QtGui.QLabel('Minimum distance:')
        self.minimum_distance_Edit = QtGui.QLineEdit('15')
        self.minimum_distance_Edit.textChanged.connect(self.get_peak_local_maximum)
        
        self.maximum_button = QtGui.QPushButton('Index of maximum:')
        self.index_NP_Edit = QtGui.QLineEdit('0,1')
        
        total_NP_Label = QtGui.QLabel('Total NPs:')
        self.total_NP_Edit = QtGui.QLineEdit('2')
        
        empty_NP_Label = QtGui.QLabel('Position of empty NPs:')
        self.empty_NP_Edit = QtGui.QLineEdit('0')
        
        center_NP_Label = QtGui.QLabel('Position row of NPs (pixel):')
        self.center_NP_Edit = QtGui.QLineEdit('')
        
        name_NP_Label = QtGui.QLabel('Name NP_:')
        self.name_NP_Edit = QtGui.QLineEdit('')
        
        center_bkg_Label = QtGui.QLabel('Position row of bkg (pixel):')
        self.center_bkg_Edit = QtGui.QLineEdit('300')
        
        self.maximum_button.clicked.connect(self.get_peak_local_maximum_position)
        
        self.plot_NPs_button = QtGui.QPushButton('Plot spectrum NPs')
        self.plot_NPs_button.setCheckable(True)
        self.plot_NPs_button.clicked.connect(self.plot_NPs)

        self.save_spectrum_button = QtGui.QPushButton('Save spectrum NPs')
        self.save_spectrum_button.setCheckable(True)
        self.save_spectrum_button.clicked.connect(self.save_spectrum)

        #Line profile Horizontal:

        self.lineHorizontal_button = QtGui.QPushButton('View spectrum of LINE Horizontal')
        self.lineHorizontal_button.setCheckable(True)
        self.lineHorizontal_button.clicked.connect(self.create_line_Horizontal)
        self.lineHorizontal_button.setStyleSheet(
                       
                "QPushButton:pressed { background-color: blue; }")
          
        size_spot_Label = QtGui.QLabel('Size Bin:')
        self.size_spot_Edit = QtGui.QLineEdit('5')
        self.size_spot_Edit.textChanged.connect(self.lines_Horizontal)
        
        self.sl = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.sl.setMinimum( int( self.size_spot_Edit.text() ) )
        self.sl.setMaximum(1002 - int(self.size_spot_Edit.text()) )
        self.sl.setValue(400)
        self.sl.setTickPosition(QtGui.QSlider.TicksBelow)
        self.sl.setTickInterval(1)
        self.sl.valueChanged.connect(self.lines_Horizontal)
        
        center_row_Label = QtGui.QLabel('Center row (pixel):')
        self.center_row_Edit = QtGui.QLabel('0')
        self.center_row_Edit.setText(format(int(self.sl.value())))
        
        
        self.frame_sl = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.frame_sl.setMinimum(0)
        self.frame_sl.setMaximum(1000)
        self.frame_sl.setValue(0)
        self.frame_sl.setTickPosition(QtGui.QSlider.TicksBelow)
        self.frame_sl.setTickInterval(1)
        self.frame_sl.valueChanged.connect(self.lines_Horizontal)
        self.frame_sl.valueChanged.connect(self.update_frame)

        frame_Label = QtGui.QLabel('Frame:')
        self.frame_Edit = QtGui.QLabel('0')
        self.frame_Edit.setText(format(int(self.frame_sl.value())))
     
        self.normalized_lamp_option = QtGui.QCheckBox('Normalized with lamp')
        self.normalized_lamp_option.clicked.connect(self.normalized_lamp_check)
        self.normalized_lamp_option.setToolTip('Check is Normalized')
        
        window_smooth_Label = QtGui.QLabel('Window smooth of filter:')
        self.window_smooth_Edit = QtGui.QLineEdit('11')
        self.window_smooth_Edit.textChanged.connect(self.lines_Horizontal)

        self.analize_parameters = QtGui.QWidget()
        parameters_layout = QtGui.QGridLayout()
        self.analize_parameters.setLayout(parameters_layout)

        parameters_layout.addWidget(self.import_image_button,    0, 0)

        parameters_layout.addWidget(self.lineHorizontal_button,         2, 0)
        parameters_layout.addWidget(size_spot_Label,                    3, 0)
        parameters_layout.addWidget(self.size_spot_Edit,                3, 1)
        parameters_layout.addWidget(center_row_Label,                   4, 0)
        parameters_layout.addWidget(self.center_row_Edit,               4, 1)
        parameters_layout.addWidget(self.sl,                            5, 0)
        parameters_layout.addWidget(frame_Label,                        6, 0)
        parameters_layout.addWidget(self.frame_Edit,                    6, 1)
        parameters_layout.addWidget(self.frame_sl,                      7, 0)
        
        parameters_layout.addWidget(self.normalized_lamp_option,        8, 0)
        parameters_layout.addWidget(center_bkg_Label,                    9, 0)
        parameters_layout.addWidget(self.center_bkg_Edit,                9, 1)
        parameters_layout.addWidget(window_smooth_Label,                10, 0)
        parameters_layout.addWidget(self.window_smooth_Edit,            10, 1)

        parameters_layout.addWidget(self.lineprofile_button,            11, 0)
        parameters_layout.addWidget(threshold_rel_Label ,               12, 0)
        parameters_layout.addWidget(self.threshold_rel_Edit ,           12, 1)
        parameters_layout.addWidget(minimum_distance_Label ,            13, 0)
        parameters_layout.addWidget(self.minimum_distance_Edit ,        13, 1)

        parameters_layout.addWidget(self.maximum_button ,               14, 0)
        parameters_layout.addWidget(self.index_NP_Edit ,                15, 0)
        
        parameters_layout.addWidget(total_NP_Label,                     16, 0)
        parameters_layout.addWidget(self.total_NP_Edit,                 16, 1)
                
        parameters_layout.addWidget(empty_NP_Label,                     17, 0)
        parameters_layout.addWidget(self.empty_NP_Edit,                 17, 1)
        
        parameters_layout.addWidget(name_NP_Label,                       18, 0)
        parameters_layout.addWidget(self.name_NP_Edit,                   19, 0)
        
        parameters_layout.addWidget(center_NP_Label,                     20, 0)
        parameters_layout.addWidget(self.center_NP_Edit,                 21, 0)
        
        parameters_layout.addWidget(self.save_spectrum_button ,          22, 0)
       # parameters_layout.addWidget(self.plot_NPs_button ,               2, 0)
        
        ## Viewbox
        
        imageWidget = pg.GraphicsLayoutWidget()
        imageWidget.setAspectLocked(True)

        self.vb = imageWidget.addPlot()
        self.img= pg.ImageItem()
        self.vb.addItem(self.img)

        self.hist = pg.HistogramLUTItem(image=self.img)
        self.hist.gradient.loadPreset('grey')
# 'thermal', 'flame', 'yellowy', 'bipolar', 'spectrum',
# 'cyclic', 'greyclip', 'grey' # Solo son estos
        for tick in self.hist.gradient.ticks:
            tick.hide()
        imageWidget.addItem(self.hist, row=0, col=1)

        ## Viewbox sum

        image_sum_Widget = pg.GraphicsLayoutWidget()
        image_sum_Widget.setAspectLocked(True)
     
        self.vb_sum = image_sum_Widget.addPlot()
        self.img_sum = pg.ImageItem()
        self.vb_sum.addItem(self.img_sum)
        
        # set up histogram for the liveview image

        self.hist_sum = pg.HistogramLUTItem(image=self.img_sum)
        self.hist_sum.gradient.loadPreset('yellowy')
# 'thermal', 'flame', 'yellowy', 'bipolar', 'spectrum',
# 'cyclic', 'greyclip', 'grey' # Solo son estos
        for tick in self.hist_sum.gradient.ticks:
            tick.hide()
        image_sum_Widget.addItem(self.hist_sum, row=0, col=1)

        ## Viewbox live
        
        self.vb_live = QtGui.QWidget()
        self.vb_live.setLayout(QtGui.QHBoxLayout())
        self.img_live = pg.ImageView()
        self.vb_live.layout().addWidget(self.img_live)

        ## Mas cositas

        self.lineplotWidget = viewbox_tools.linePlotWidget_pixel()
        self.curve_line = self.lineplotWidget.linePlot.plot(open='y')
        self.curve_line_max = self.lineplotWidget.linePlot.plot(open='y')

        self.lineplot_spectrum_Widget = viewbox_tools.linePlotWidget_spectrum()
        self.curve_spectrum = self.lineplot_spectrum_Widget.linePlot.plot(open='y')
        
        self.frame = int(self.frame_sl.value())

        self.center_row  = int(self.sl.value())
        self.spot_size = int(self.size_spot_Edit.text())
        
        self.normalized_bool = False
        self.center_bkg = int(self.center_bkg_Edit.text())
        self.window_smooth = int(self.window_smooth_Edit.text())
        
        self.mouse_cursor_x = viewbox_tools.TwolinesHorizontal_fixed(self.vb, self.center_row, self.spot_size)
        
        self.lineplot_spectrum_NPs_Widget = viewbox_tools.linePlotWidget_spectrum()
        
        #docks

        hbox = QtGui.QHBoxLayout(self)
        dockArea = DockArea()
        hbox.addWidget(dockArea)
        self.setLayout(hbox)

        analize_dock = Dock('Analisis')
        analize_dock.addWidget(self.analize_parameters)
        dockArea.addDock(analize_dock)

        viewbox_live_dock = Dock('Viewbox Live', size = (500, 500))
        viewbox_live_dock.addWidget(self.vb_live)
        dockArea.addDock(viewbox_live_dock, 'right', analize_dock)

        viewbox_sum_dock = Dock('Viewbox sum',  size = (500, 500))
        viewbox_sum_dock.addWidget(image_sum_Widget)
        dockArea.addDock(viewbox_sum_dock, 'above', viewbox_live_dock)

        viewbox_dock = Dock('Viewbox', size = (500, 500))
        viewbox_dock.addWidget(imageWidget)
        dockArea.addDock(viewbox_dock, 'above', viewbox_sum_dock)

        line_profile_max_dock = Dock('Maximums of Vertical Line')
        line_profile_max_dock.addWidget(self.lineplotWidget)
        dockArea.addDock(line_profile_max_dock, 'right', viewbox_live_dock)
        
        line_spectrum_dock = Dock('Spectrum of Horizontal Line')
        line_spectrum_dock.addWidget(self.lineplot_spectrum_Widget)
        dockArea.addDock(line_spectrum_dock, 'bottom', line_profile_max_dock)
    
    def import_image(self):
        self.update_image_Signal.emit()
       
    @pyqtSlot(np.ndarray)
    def get_image(self, image):

        x = np.linspace(1., image.shape[0], image.shape[0])
        
        self.img_live.setImage(image, xvals=x, autoHistogramRange=True)
        
        image_sum = np.zeros((image.shape[0], int(image.shape[1]/16) , image.shape[2]))
          
        for i in range(image.shape[0]):
            
            imagen = image[i, :, :]
            image_sum[i, :, :] = sum_image(imagen, 16)
        
        self.frame_sl.setMaximum(int(image.shape[0])-1)

        self.img.setImage(image[self.frame, :, :])
        self.img_sum.setImage(image_sum[self.frame, :, :])
        
        self.image = image
        self.image_sum = image_sum
        
        
    def update_frame(self):

        self.frame_Edit.setText(format(int(self.frame_sl.value())))
        #self.frame = int(self.frame_Edit.text())
        self.frame = int(self.frame_sl.value())

        self.img.setImage(self.image[self.frame, :, :])
        self.img_sum.setImage(self.image_sum[self.frame, :, :])
        

    def create_line_Horizontal(self):
        
        self.center_row = int(self.sl.value())
        self.spot_size = int(self.size_spot_Edit.text())
        
        self.center_bkg = int(self.center_bkg_Edit.text())
        
        self.window_smooth = int(self.window_smooth_Edit.text())
        self.frame = int(self.frame_sl.value())

        if self.lineHorizontal_button.isChecked():

            self.mouse_cursor_x = viewbox_tools.TwolinesHorizontal_fixed(self.vb , self.center_row, self.spot_size)

            self.lineparametersSignal.emit(self.frame, self.center_row, self.spot_size, self.normalized_bool, self.center_bkg, self.window_smooth)
                    
            self.mouse_cursor_x.show()
            self.lineplot_spectrum_Widget.show()
       
        else:
            
            self.mouse_cursor_x.hide()
            self.lineplot_spectrum_Widget.hide()     
    
    def normalized_lamp_check(self):
        
        self.center_row = int(self.sl.value())
        self.spot_size = int(self.size_spot_Edit.text())
        
        self.center_bkg = int(self.center_bkg_Edit.text())
        
        self.window_smooth = int(self.window_smooth_Edit.text())
        self.frame = int(self.frame_sl.value())
        
        if self.normalized_lamp_option.isChecked():
            self.normalized_bool = True
            if self.window_smooth > 1:
                self.lineparametersSignal.emit(self.frame, self.center_row, self.spot_size, self.normalized_bool, self.center_bkg, self.window_smooth)
            else:
                print('Fijarse que window smooth sea mayor a 1 (orden del polinomio)')
        else:
            self.normalized_bool = False
            self.lineparametersSignal.emit(self.frame, self.center_row, self.spot_size, self.normalized_bool, self.center_bkg, self.window_smooth)

    def lines_Horizontal(self):

        self.center_row_Edit.setText(format(int(self.sl.value())))  

        if self.lineHorizontal_button.isChecked():
            
            self.frame = int(self.frame_sl.value())
            self.center_row = int(self.center_row_Edit.text())  
            self.spot_size = int(self.size_spot_Edit.text())
            
            self.center_bkg = int(self.center_bkg_Edit.text())  
            self.window_smooth = int(self.window_smooth_Edit.text())

            down_row = self.center_row - int((self.spot_size-1)/2)
            up_row = self.center_row + int((self.spot_size-1)/2)+1

            self.mouse_cursor_x.hLine_up.setPos(up_row) 
            self.mouse_cursor_x.hLine_down.setPos(down_row)

            self.lineparametersSignal.emit(self.frame, self.center_row, self.spot_size, self.normalized_bool, self.center_bkg, self.window_smooth)

        
    def create_line_profile(self):
        
        if self.lineprofile_button.isChecked():

            self.live_LINE_Signal.emit(True)

            self.lineROI = pg.LineSegmentROI([[16, 250], [16, 500]], pen='m')
            self.vb_sum.addItem(self.lineROI)
        
            self.lineplotWidget.show()
                            
        else:

            self.live_LINE_Signal.emit(False)
            self.lineplotWidget.hide()
            self.vb_sum.removeItem(self.lineROI)
            
    @pyqtSlot(np.ndarray)        
    def update_LINE(self, image_sum):
        
        x_o, y_o = self.lineROI.listPoints()[0]
        x_f, y_f = self.lineROI.listPoints()[1]
        
        imagen_sum = self.image_sum[self.frame, :, :]
            
        array_intensity = self.lineROI.getArrayRegion(imagen_sum, self.img_sum)

        xmin, ymin = self.lineROI.pos() + self.lineROI.listPoints()[0]
        xmax, ymax = self.lineROI.pos() + self.lineROI.listPoints()[1]
                
        array_pos_y = np.linspace(ymin,  ymax, len(array_intensity))
        
        self.curve_line.setData(array_pos_y, array_intensity,
                           pen=pg.mkPen('m', width=3))

        self.array_intensity = array_intensity
        self.array_pos_y = array_pos_y

        self.get_peak_local_maximum()
        
    def get_peak_local_maximum(self):
        
        threshold_rel = float(self.threshold_rel_Edit.text())
        mimimum_distance = int(self.minimum_distance_Edit.text())

        self.maximum_Signal.emit(self.array_pos_y, self.array_intensity, threshold_rel, mimimum_distance)
        
    def get_peak_local_maximum_index(self):

        index_NP = int(self.index_NP_Edit.text())

        posNP = int(self.index_max[index_NP])
        position_NP = int(self.y_pos_max[posNP])

        self.center_NP_Edit.setText(format(position_NP))

    def get_peak_local_maximum_position(self):

        x = str(self.index_NP_Edit.text())
        values_index_NP = transform_str(x)

        center_NP_list = []
        
        total_NP = int(self.total_NP_Edit.text())
        
        name_NP_list = np.arange(0,total_NP ,1) + 1
        
        empty = str(self.empty_NP_Edit.text())
        values_empty_NP = transform_str(empty)
        
        if values_empty_NP[0] != 0:
            
            i = 0
            for i in range(len(values_empty_NP)):
                
                name_NP_list = np.where(name_NP_list == values_empty_NP[i], 0, name_NP_list)
            
            name_NP_list = list(np.sort(name_NP_list)[len(values_empty_NP):])
            
        else:
            
            name_NP_list = list(np.arange(0,total_NP ,1) + 1)
        
        
        self.name_NP_Edit.setText(format(str(name_NP_list)[1:-1]))
        
        if len(values_index_NP) > len(self.index_max):

            print('No existe esa cantidad de maximos.')

        else:

            for index_NP in values_index_NP: 

                if index_NP < len(self.index_max):

                    posNP = int(self.index_max[index_NP])
                    position_NP = int(self.y_pos_max[posNP])
                    center_NP_list.append(position_NP)
                else:

                    print('Alguno de esos maximos no existe.')

            self.center_NP_Edit.setText(format(str(center_NP_list)[1:-1]))
            
    def plot_NPs(self):

        x = str(self.index_NP_Edit.text())
        values_index_NP = transform_str(x)

        if len(values_index_NP) > len(self.index_max):

            print('plot_NPs: No existe esa cantidad de maximos.')

        else:

            if self.plot_NPs_button.isChecked():

                self.lineplot_spectrum_NPs_Widget.show()

                for index_NP in values_index_NP: 

                    if index_NP < len(self.index_max):

                        posNP = int(self.index_max[index_NP])
                        position_NP = int(self.y_pos_max[posNP])

                        self.center_row = position_NP 

                        self.lineparametersSignal.emit(self.frame, self.center_row, self.spot_size, self.normalized_bool, self.center_bkg, self.window_smooth)
                        self.mouse_cursor_NPs = viewbox_tools.TwolinesHorizontal_fixed(self.vb , position_NP, self.spot_size)
                        self.mouse_cursor_NPs.show()

                    else:
                        print('plot_NPs: Alguno de esos maximos no existe.')

            else:

                for index_NP in values_index_NP: 

                    #self.lineplot_spectrum_NPs_Widget.hide()
                    self.curve_spectrum_NPs.clear()
                    self.mouse_cursor_NPs.hide() 

    def save_spectrum(self):

        name_col = int(self.frame)
        name_NPs = self.name_NP_Edit.text()
        center_NPs = self.center_NP_Edit.text()

        info_NPs = [name_col, name_NPs, center_NPs]
        
        spot_size = int(self.size_spot_Edit.text())
        center_bkg = int(self.center_bkg_Edit.text())
        window_smooth = int(self.window_smooth_Edit.text())

        self.savespectrumSignal.emit(info_NPs, spot_size, center_bkg, window_smooth)
        
                
    @pyqtSlot(np.ndarray, np.ndarray, np.ndarray)        
    def update_LINE_maximum(self, index_max, y_pos_max, profile_max):
        
        self.index_max = index_max
        self.y_pos_max = y_pos_max
        
        self.curve_line_max.setData(y_pos_max, profile_max, pen = None, symbol = 'o') 
        

    @pyqtSlot(np.ndarray, np.ndarray)
    def spectrum_line_show(self, array_wavelength, array_profile):

        if self.lineHorizontal_button.isChecked():

            self.curve_spectrum.setData(array_wavelength, array_profile,
                           pen=pg.mkPen('g', width=3)) 

        if self.plot_NPs_button.isChecked():

            self.curve_spectrum_NPs = self.lineplot_spectrum_NPs_Widget.linePlot.plot(open='y')
            self.curve_spectrum_NPs.setData(array_wavelength, array_profile,
                           pen=pg.mkPen('m', width=3))


    def closeEvent(self, *args, **kwargs):
        
        super().closeEvent(*args, **kwargs)
        
    def make_connection(self, backend):
        
        backend.imageSignal.connect(self.get_image)
        
        backend.imageSignal_LINE.connect(self.update_LINE)
        backend.peak_local_maximum_Signal.connect(self.update_LINE_maximum)
        backend.line_spectrumSignal.connect(self.spectrum_line_show)
        
        
class Backend(QtCore.QObject):

    imageSignal = pyqtSignal(np.ndarray)
    
    imageSignal_LINE = pyqtSignal(np.ndarray)
    peak_local_maximum_Signal = pyqtSignal(np.ndarray, np.ndarray, np.ndarray)
    
    bkg_measure_Signal = pyqtSignal(float)
    line_spectrumSignal = pyqtSignal(np.ndarray, np.ndarray)
    

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.viewTimer_LINE = QtCore.QTimer()
        self.viewTimer_LINE.timeout.connect(self.update_view_LINE)   
 
        self.center_row = 400
        self.spot_size =  5
        self.normalized_lamp_bool = False
        self.frame = 0

        self.lamp = lamp # ver como cargarlo
        self.window_smooth = 11
        self.center_bkg = 300

    def update_image(self):

        # Remove annoying empty window
        root = Tk()
        root.withdraw()

        #Select image from file
       # self.f = filedialog.askopenfilename(filetypes=(("", "*.tiff"), ("", "*.tif")))
       # Import selected image
       # image = io.imread(self.f)
        #self.image = np.transpose(image)
        #self.image = data_image
        
        direction = filedialog.askdirectory()
        self.file_path = direction
        
        listfiles =[]
        data_calibration = 0
        for files in os.listdir(direction):
            if files.endswith(".tiff") :
                listfiles.append(files)
            if files.endswith(".txt"):
                data_calibration = files
                
        listfiles = sorted(listfiles)
        print(listfiles, data_calibration)

        first_image = np.transpose(io.imread(direction + "/"+ listfiles[0]))

        stack = np.zeros((len(listfiles),first_image.shape[0],first_image.shape[1]))
        
        for n in range(0,len(listfiles)):
            stack[n,:,:]= np.transpose(io.imread(direction + "/"+ listfiles[n]))
        
        #self.image = stack[:,1:,:]  #solo porq guarde mal StepandGlue
        self.image = stack
        
        self.calibration = np.array(np.loadtxt(str(direction + "/"+ data_calibration)))
        
        print('Image shape', self.image.shape, 'Calibration len', len(self.calibration))
        
        self.imageSignal.emit(self.image)
        self.line_spectrum_parameters(self.frame, self.center_row, self.spot_size, self.normalized_lamp_bool, self.center_bkg, self.window_smooth)
    
        image_sum = np.zeros((self.image.shape[0], int(self.image.shape[1]/16) , self.image.shape[2]))
        
        for i in range(self.image.shape[0]):
            
            imagen = self.image[i, :, :]
            image_sum[i, :, :] = sum_image(imagen, 16)
            
        self.image_sum = image_sum
            
        print(direction)


    @pyqtSlot(bool)
    def liveview_LINE(self, livebool):

        time_refresh = 0.5
        
        if livebool: 
            self.viewTimer_LINE.start(time_refresh*10**3)
        else:
            self.viewTimer_LINE.stop()

    def update_view_LINE(self):

        self.imageSignal_LINE.emit(self.image_sum)
        
    @pyqtSlot(np.ndarray, np.ndarray, float, int)    
    def peak_local_maximum_LINE(self, y_position_LINE, profile_LINE, threshold, minimum_distance):
        
        index_max = peak_local_max(profile_LINE,  min_distance = minimum_distance, threshold_rel = threshold, indices = True)
        #min_distance = 1, threshold_rel = 0.9, threshold_abs = 4, 
        
        profile_max = np.zeros(len(profile_LINE))
        
        #y_pos_max = y_position_LINE[index_max]
        profile_max[index_max] = profile_LINE[index_max]

        self.peak_local_maximum_Signal.emit(index_max, y_position_LINE, profile_max)
        

    @pyqtSlot(int, int, int, bool, int, int)    
    def line_spectrum_parameters(self, frame, center_row, spot_size, normalized_lamp_bool, center_bkg, window_smooth):
        
        self.frame = frame
        self.center_row = center_row
        self.spot_size = spot_size
        
        self.center_bkg = center_bkg
        self.window_smooth = window_smooth
        self.normalized_lamp_bool = normalized_lamp_bool

        if self.frame in range(self.image.shape[0]):
            
            imagen = self.image[frame, :, :]
                
            if not self.normalized_lamp_bool:

                wavelength, spectrum_NP = self.line_spectrum(self.calibration, imagen)
                self.line_spectrumSignal.emit(wavelength, spectrum_NP)

            else:

                wavelength, spectrum_NP_normalized = self.line_spectrum_normalized_lamp(self.calibration, imagen, self.lamp, self.center_row, self.window_smooth)
                self.line_spectrumSignal.emit(wavelength, spectrum_NP_normalized)

    def line_spectrum(self, calibration, image):
        
        down_row = self.center_row - int((self.spot_size-1)/2)
        up_row = self.center_row + int((self.spot_size-1)/2) + 1  
        roi_rows = range(down_row, up_row)
        
        spectrum = np.round(np.mean(image[:,roi_rows], axis=1),2)
        wavelength = calibration 

        return wavelength, spectrum
    
    def line_spectrum_bkg(self, calibration, image):
        
        down_row = self.center_bkg - int((self.spot_size-1)/2)
        up_row = self.center_bkg + int((self.spot_size-1)/2) + 1  
        roi_rows = range(down_row, up_row)
        
        spectrum = np.round(np.mean(image[:,roi_rows], axis=1),2)
        wavelength = calibration 

        return wavelength, spectrum
    
    def line_spectrum_normalized_lamp(self, calibration, image, lamp, center_row, window_smooth):
        
        wavelength, spectrum_NP = self.line_spectrum(calibration, image)
        wavelengt, spectrum_bkg = self.line_spectrum_bkg(calibration, image)
        
        spectrum_NP = [spectrum_all for _,spectrum_all in sorted(zip(wavelength,spectrum_NP))]
        spectrum_bkg = [spectrum_bkg for _,spectrum_bkg in sorted(zip(wavelength,spectrum_bkg))]

        wavelength = np.sort(wavelength)
        spectrum_NP = np.array(spectrum_NP)
        spectrum_bkg = np.array(spectrum_bkg)
        
       # desired_range = np.where((wavelength >= 400) & (wavelength <=1000))  
       # wavelength = wavelength[desired_range]
       # spectrum_NP = spectrum_NP[desired_range]
       # spectrum_bkg = spectrum_bkg[desired_range]
        
        smooth_spectrum_NP =  signal.savgol_filter(spectrum_NP, window_smooth, 0, mode = 'mirror')
        
        smooth_spectrum_bkg =  signal.savgol_filter(spectrum_bkg, window_smooth, 0, mode = 'mirror')
        smooth_spectrum_bkg =  signal.savgol_filter(spectrum_bkg, window_smooth, 0, mode = 'mirror')
        smooth_spectrum_bkg =  signal.savgol_filter(spectrum_bkg, window_smooth, 0, mode = 'mirror')
             
        smooth_spectrum_lamp =  signal.savgol_filter(lamp[1], window_smooth, 0, mode = 'mirror')
        
        spectrum_lamp = self.interpolated_lamp(wavelength, [lamp[0], smooth_spectrum_lamp])
        normalized_spectrum_lamp = spectrum_lamp/max(spectrum_lamp)

        spectrum_NP_normalized = (smooth_spectrum_NP-smooth_spectrum_bkg)/normalized_spectrum_lamp

        return wavelength, spectrum_NP_normalized
    
    def interpolated_lamp(self, wavelength, lamp):

        lower_lambda = wavelength[0]
        upper_lambda = wavelength[-1]
        step = len(wavelength)
        
        wavelength_lamp = lamp[0]
        spectrum_lamp = lamp[1]
            
        wavelength_new = np.linspace(lower_lambda, upper_lambda, step)

        desired_range = np.where((wavelength_lamp>=lower_lambda) & (wavelength_lamp<=upper_lambda))
        
        wavelength_lamp = wavelength_lamp[desired_range]
        spectrum_lamp = spectrum_lamp[desired_range]

        new_lamp_spectrum = np.interp(wavelength_new, wavelength_lamp, spectrum_lamp)

        return new_lamp_spectrum

    @pyqtSlot(list, int, int, int)
    def spectrum_NPs(self, info_NPs, spot_size, center_bkg, window_smooth):

        frame = info_NPs[0]

        folder = self.create_folder(self.file_path, frame, normalized = False)
        folder_normalized = self.create_folder(self.file_path, frame, normalized = True)

        name_NPs = transform_str(info_NPs[1])
        center_NPs = transform_str(info_NPs[2])
        
        print(name_NPs, center_NPs)

        self.spot_size = spot_size
        
        self.center_bkg = center_bkg
        self.window_smooth = window_smooth
        
        imagen = self.image[frame, :, :]
        
        i = 0
        
        for i in range(len(center_NPs)):

            self.center_row = int(center_NPs[i])

            NP = int(name_NPs[i])
            wavelength, spectrum_NP = self.line_spectrum(self.calibration, imagen)
            self.save_spectrum(wavelength, spectrum_NP, NP, folder)
            
            wavelength, spectrum_bkg = self.line_spectrum_bkg(self.calibration, imagen)
            self.save_spectrum_bkg(wavelength, spectrum_bkg, center_bkg, folder)

            wavelength, spectrum_NP_normalized = self.line_spectrum_normalized_lamp(self.calibration, imagen, self.lamp, self.center_bkg, self.window_smooth)
            self.save_spectrum(wavelength, spectrum_NP_normalized, NP, folder_normalized)

            print('Save col', self.frame)

    def save_spectrum(self, axe_x, axe_y, NP, folder):

        file_path = folder
        file_name = "NP_%2d.txt"%(NP)

        fullname = os.path.join(file_path, file_name)
        np.savetxt(fullname, np.transpose([axe_x, axe_y]))
        
    def save_spectrum_bkg(self, axe_x, axe_y, row, folder):

        file_path = folder
        file_name = "background_row_%2d.txt"%(row)

        fullname = os.path.join(file_path, file_name)
        np.savetxt(fullname, np.transpose([axe_x, axe_y]))
        
    def create_folder(self, file, number, normalized):

        file_path = file
        
        if not normalized:
        
            name_folder = "frame_%2d"%(number)
            folder = os.path.join(file_path, name_folder)
            
        else:
            
            name_folder = "normalized_frame_%2d"%(number)
            folder = os.path.join(file_path, name_folder)
            
        os.makedirs(folder)

        return folder

    def make_connection(self, frontend):

        frontend.lineparametersSignal.connect(self.line_spectrum_parameters)
        frontend.update_image_Signal.connect(self.update_image)
        frontend.live_LINE_Signal.connect(self.liveview_LINE)
        frontend.maximum_Signal.connect(self.peak_local_maximum_LINE)
        frontend.savespectrumSignal.connect(self.spectrum_NPs)

#%%   extra functions     

def transform_str(str_values):

        values = list(str(str_values).split(","))

        data = []

        for i in range(len(values)):
            data.append(int(values[i]))

        return data

def sum_image(image, N):

        columns_image = image.shape[0]
        first = int(columns_image/N)
        image_sum = np.zeros((first, image.shape[1]))

        i = 1
        j = int(columns_image/N)

        while i < N + 1:

            image_sum = image_sum + image[range(j*(i-1), j*i), :]
            i = i + 1

        return image_sum

if __name__ == '__main__':
    
    lampara = np.loadtxt('lamp_IR_unpol.asc', dtype='float')
    
    wave_lampara = np.array(lampara[:, 0])
    image_lampara = np.array(lampara[:, 1:])
    
    rows = range(400, 601)
    espectro_lampara = np.round(np.mean(image_lampara[:,rows], axis=1),2)
    
    smooth_espectro_lampara  =  signal.savgol_filter(espectro_lampara, 21, 0, mode = 'mirror') 
    
    plt.plot(wave_lampara, espectro_lampara)
    plt.plot(wave_lampara, smooth_espectro_lampara)
    plt.show()
    
    lamp = [wave_lampara, espectro_lampara]

    app = QtGui.QApplication([])

    gui = Frontend()   
    worker = Backend()

    worker.make_connection(gui)
    gui.make_connection(worker)

    cameraThread = QtCore.QThread()
    worker.moveToThread(cameraThread)
    worker.viewTimer_LINE.moveToThread(cameraThread)
    cameraThread.start()
    
    gui.showMaximized()
    gui.show()
    app.exec_()