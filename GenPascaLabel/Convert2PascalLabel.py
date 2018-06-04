
# -*- coding: utf-8 -*-

from PIL import Image
from xml.etree import ElementTree as et
import os

wd = os.getcwd()
classname = "yaoxiang"
DirDatasetRoot = "train_dataset"
print("current working directry should be one level above the pos")
print(wd)

def EncodeClassToIdx(classname) : 
    if classname == "yaoxiang" :
        featureids = 0
    return featureids

def AnnotateXML(fidx,imw,imh,xmin,ymin,xmax,ymax):
  #load xml template
  #replace the certain value
    imgfname = "yaoxiang-" + str(fidx) + ".jpg" 
    annfname = "yaoxiang-" + str(fidx) + ".xml"
    fullpathname = "./image/" + imgfname

    tree = et.parse("pascal_anno_template.xml")
    root = tree.getroot()
    print("root is ",root)
    print("......parasing template xml")

    #annotating the correct XML file
    tree.find('filename').text           = imgfname 
    tree.find('path').text               = fullpathname
    tree.find('size/width').text         = str(imw)
    tree.find('size/height').text        = str(imh)
    tree.find('object/name').text        = classname 
    tree.find('object/bndbox/xmin').text = str(xmin)
    tree.find('object/bndbox/ymin').text = str(ymin)
    tree.find('object/bndbox/xmax').text = str(xmax) 
    tree.find('object/bndbox/ymax').text = str(ymax)

    tree.write(annfname)
    print("......generated ", annfname)
    return 0


def ExtractXY(RawFile) : 

    RawLabel = open(RawFile)

    #lines = RawLabel.read().split('\r\n')   #for ubuntu, use "\r\n" instead of "\n"
    lines = RawLabel.read().split('\n')   #for ubuntu, use "\r\n" instead of "\n"
  
    print("...extracing raw file from label file")

    ct = 0
    for line in lines:
        #print('lenth of line is: ')
        if(len(line) >= 2):
            ct = ct + 1
            print(line + "\n")
            elems = line.split(' ')
            print(elems)
            xmin = elems[0]
            xmax = elems[2]
            ymin = elems[1]
            ymax = elems[3]
            #
            #img_path = str('%s/images/%s/%s.JPEG'%(wd, cls, os.path.splitext(RawFile)[0]))
            #t = magic.from_file(img_path)
            #wh= re.search('(\d+) x (\d+)', t).groups()
            #im=Image.open(img_path)
            #imw= int(im.size[0])
            #imh= int(im.size[1])
            #w = int(xmax) - int(xmin)
            #h = int(ymax) - int(ymin)
            # print(xmin)
            #print(w, h)
            imw = 0
            imh = 0
    return (imw,imh, xmin,ymin,xmax,ymax)




def ProcessDataSet(RootDirOfDataSet):

     """
     Suppose the file directory is as follows:
     dataset
     |
     |
     |----Labels(input)---encode(classname)
     |----Images(input)
     |----Annotations(output)
     """
     labeldir      = RootDirOfDataSet +  "/Labels/"   + str(EncodeClassToIdx(classname)) 
     annotatoindir = RootDirOfDataSet + "_pascal_format/" + "/Annotations/  
     
     for AlableFile in os.listdir(labeldir) :
        FlableFilePath = labeldir + "/" + AlableFile

        imw,imh,xmin,ymin,xmax,ymax  = ExtractXY(FlableFilePath)
        print FlableFilePath

if __name__ == '__main__':
   imw,imh,xmin,ymin,xmax,ymax  = ExtractXY('label_rawout_01.txt')
   print("xmin",xmin)
   AnnotateXML(0,640,480,0,0,640,480) 
   ProcessDataSet(DirDatasetRoot)