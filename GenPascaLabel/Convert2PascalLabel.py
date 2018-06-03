
# -*- coding: utf-8 -*-

from PIL import Image
from xml.etree import ElementTree as et
import os

wd = os.getcwd()
classname = "yaoxiang"
print("current working directry should be one level above the pos")
print(wd)


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



"""
def ExtractXY(RawFile) ：

    RawLabel = open(RawFilw,r)

    lines = RawLabel.read().split('\r\n')   #for ubuntu, use "\r\n" instead of "\n"

    ct = 0
    for line in lines:
        #print('lenth of line is: ')
        #print(len(line))
        #print('\n')
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
            img_path = str('%s/images/%s/%s.JPEG'%(wd, cls, os.path.splitext(RawFile)[0]))
            #t = magic.from_file(img_path)
            #wh= re.search('(\d+) x (\d+)', t).groups()
            im=Image.open(img_path)
            imw= int(im.size[0])
            imh= int(im.size[1])
            #w = int(xmax) - int(xmin)
            #h = int(ymax) - int(ymin)
            # print(xmin)
            print(w, h)
    return (imw,imh, xmin,ymin,xmax,ymax)





def LabelDir():

"""
"""
"""
if __name__ == '__main__':
   AnnotateXML(0,640,480,0,0,640,480)