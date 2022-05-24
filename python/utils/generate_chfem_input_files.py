import os
import sys
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets 

# This function was adapted from (https://github.com/Entscheider/SeamEater/blob/master/gui/QtTool.py)
# Project: SeamEater; Author: Entscheider; File: QtTool.py; GNU General Public License v3.0 
# Original function name: qimage2numpy(qimg)
# We consider just 8 bits images and convert to single depth:
def convertQImageToNumpy(_qimg):
    h = _qimg.height()
    w = _qimg.width()
    ow = _qimg.bytesPerLine() * 8 // _qimg.depth()
    d = 0
    if _qimg.format() in (QtGui.QImage.Format_ARGB32_Premultiplied,
                          QtGui.QImage.Format_ARGB32,
                          QtGui.QImage.Format_RGB32):
        d = 4 # formats: 6, 5, 4.
    elif _qimg.format() in (QtGui.QImage.Format_Indexed8,
                            QtGui.QImage.Format_Grayscale8):
        d = 1 # formats: 3, 24.
    else:
        raise ValueError(".ERROR: Unsupported QImage format!")
    buf = _qimg.bits().asstring(_qimg.byteCount())
    res = np.frombuffer(buf, 'uint8')
    res = res.reshape((h,ow,d)).copy()
    if w != ow:
        res = res[:,:w] 
    if d >= 3:
        res = res[:,:,0].copy()
    else:
        res = res[:,:,0] 
    return res 

def loadImageData(_filepath):
    m_image = QtGui.QImage(_filepath)

    # We perform these conversions in order to deal with just 8 bits images:
    # convert Mono format to Indexed8
    if m_image.depth() == 1:
        m_image = m_image.convertToFormat(QtGui.QImage.Format_Indexed8)

    # convert Grayscale16 format to Grayscale8
    if not m_image.format() == QtGui.QImage.Format_Grayscale8:
        m_image = m_image.convertToFormat(QtGui.QImage.Format_Grayscale8)
    
    return convertQImageToNumpy(m_image)

def exportImage(_filename, _imgfilelist, _resolution):
    filename = _filename
    if filename[-4:] != '.raw':
        filename = filename + '.raw'

    # Save image data in RAW format
    m_data = None
    materials = {}
    with open(filename, "bw") as file_raw:
        for filepath in _imgfilelist:
            m_data = loadImageData(filepath)

            mat_i, cmat_i = np.unique(m_data,return_counts=True)
            for i in range(len(mat_i)):
                if mat_i[i] in materials:  
                    materials[mat_i[i]] += cmat_i[i]
                else:
                    materials[mat_i[i]] = cmat_i[i]

            # Save image data in binary format
            m_data.tofile(file_raw)

    materials = dict(sorted(materials.items(), key=lambda x: x[0]))
    dimensions = np.array([m_data.shape[1], m_data.shape[0], len(_imgfilelist)], dtype=int)
    vol = m_data.shape[1] * m_data.shape[0] * len(_imgfilelist)
    mat = np.array(list(materials.keys()))  
    cmat = np.array(list(materials.values()))   
    # mat = np.vstack((mat, np.zeros((mat.shape[0]), dtype=int))).T
    cmat = cmat*100.0/vol    
    
    nfdata = {}
    nfdata["type_of_analysis"] = 2
    nfdata["type_of_solver"] = 0
    nfdata["type_of_rhs"] = 0
    nfdata["voxel_size"] = _resolution
    nfdata["solver_tolerance"] = 1.0e-6
    nfdata["number_of_iterations"] = 10000
    nfdata["number_of_load_steps"] = 1
    nfdata["maximum_NR_iterations"] = 50
    nfdata["NR_tolerance"] = 1e-06
    nfdata["image_dimensions"] = dimensions.tolist()          
    nfdata["refinement"] = 1
    nfdata["number_of_materials"] = mat.shape[0]
    nfdata["properties_of_materials"] = mat.tolist()
    nfdata["volume_fraction"] = list(np.around(cmat,2))
    nfdata["data_type"] = "uint8"

    # Save image data in NF format
    with open(filename[0:len(filename)-4] + ".nf",'w') as file_nf:
        sText = ''
        for key, value in nfdata.items():
            sText += '%' + str(key) + '\n'+ str(value) + '\n\n'
        sText = sText.replace('], ','\n')
        sText = sText.replace('[','')
        sText = sText.replace(']','')
        sText = sText.replace(',','')
        file_nf.write(sText)
    
    return 

def main():
    resolution = 1.0
    cwd = os.getcwd()
    db_dir = os.path.join(cwd, r'revs', r'periodic')
    raw_dir = r'raw_files'
    destin_dir = os.path.join(cwd, raw_dir)    
    sim_dirs = [sim_dir for sim_dir in sorted(os.listdir(db_dir)) if os.path.isdir(os.path.join(db_dir, sim_dir))]
    print('images:', sim_dirs)
    print('destination:', destin_dir)

    for sim_dir in sim_dirs:
        print('- ', sim_dir)
        imgfiles = []
        for file in os.listdir(os.path.join(db_dir, sim_dir)):
            imgfiles.append(os.path.join(db_dir, sim_dir, file))
        imgfiles = sorted(imgfiles)
        print(len(imgfiles), 'files')
        exportImage(os.path.join(destin_dir, sim_dir), imgfiles, resolution)

    return

if __name__ == '__main__':
    main()