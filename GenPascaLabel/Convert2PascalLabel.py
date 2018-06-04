
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

def AnnotateXML(fidx,imw,imh,xmin,ymin,xmax,ymax,imgfname,annfname):
  #load xml template
  #replace the certain value
    #annfname = EncodeClassToIdx(classname) + str(fidx) + ".xml"

    tree = et.parse("pascal_anno_template.xml")
    root = tree.getroot()
    #print("root is ",root)
    #print("......parasing template xml")

    #annotating the correct XML file
    tree.find('filename').text           = os.path.basename(imgfname) 
    tree.find('path').text               = imgfname 
    tree.find('size/width').text         = str(imw)
    tree.find('size/height').text        = str(imh)
    tree.find('object/name').text        = classname 
    tree.find('object/bndbox/xmin').text = str(xmin)
    tree.find('object/bndbox/ymin').text = str(ymin)
    tree.find('object/bndbox/xmax').text = str(xmax) 
    tree.find('object/bndbox/ymax').text = str(ymax)

    tree.write(annfname)
    print("......generated ", annfname)


def ExtractXY(RawFile) : 

    RawLabel = open(RawFile)

    #lines = RawLabel.read().split('\r\n')   #for ubuntu, use "\r\n" instead of "\n"
    lines = RawLabel.read().split('\n')   #for ubuntu, use "\r\n" instead of "\n"
  
    #print("...Extracting from label file",RawFile)

    ct = 0
    for line in lines:
        if(len(line) >= 2):
            ct = ct + 1
            elems = line.split(' ')
            xmin = elems[0]
            xmax = elems[2]
            ymin = elems[1]
            ymax = elems[3]
    return (xmin,ymin,xmax,ymax)


def ExtractImageSize(ImageFile) :

    #print("...Extracing image size from image file",ImageFile)

    im=Image.open(ImageFile)
    imw= int(im.size[0])
    imh= int(im.size[1])

    return (imw,imh)

def ProcessDataSet(RootDirOfDataSet):

     """
     Suppose the file directory is as follows:
     dataset
     |
     |
     |----Labels(input)---encode(classname)
     |----Images(input)---encode(classname)
     |----Annotations(output)
     """
     labeldir      = RootDirOfDataSet +  "/Labels/"   + str(EncodeClassToIdx(classname)) 
     annotationdir = RootDirOfDataSet + "_pascal_format/" + "Annotations/"  

     cmd = "mkdir -p " + annotationdir + "0"
     #print(cmd)
     os.system(cmd)
     
     cnt = 0
     
     for AlableFile in os.listdir(labeldir) :

        idx = os.path.splitext(AlableFile)[0]
        FlableFilePath = labeldir + "/" + AlableFile
        FImageFilePath = RootDirOfDataSet + "/Images/" + str(EncodeClassToIdx(classname)) + "/" + str(idx) + ".jpg"
        annfname = annotationdir + str(EncodeClassToIdx(classname)) + "/" + str(idx) + ".xml"
        #print(annfname)

        #Extract Raw Lable Data  
        xmin,ymin,xmax,ymax  = ExtractXY(FlableFilePath)
        imw,imh              = ExtractImageSize(FImageFilePath)
        AnnotateXML(idx,imw,imh,xmin,ymin,xmax,ymax,FImageFilePath,annfname)

        cnt += 1 

     print("Number of DataSet in image",cnt)

if __name__ == '__main__':

   ProcessDataSet(DirDatasetRoot)