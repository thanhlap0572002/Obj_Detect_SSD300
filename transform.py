
from augmentation import Compose, ConvertFromInts, \
    ToAbsoluteCoords, PhotometricDistort, Expand, \
        RandomSampleCrop, RandomMirror, ToPercentCoords, Resize, SubtractMeans
from make_datapath import make_datapath_list
from lib import *
from extract_inform_annotation import Anno_xml

class DataTransform():
    def __init__(self, input_size, color_mean):
        self.data_transform = {
            "train": Compose([
                ConvertFromInts(),#convert image from int to float 32
                ToAbsoluteCoords(),#back anno về dạng ban đầu
                PhotometricDistort(), # thay đổi màu dùng hàm random
                Expand(color_mean),# kéo dãn ảnh 
                RandomSampleCrop(),# random cut img
                RandomMirror(),#xoay ảnh ngc lại kiểu gương
                ToPercentCoords(),# chuẩn hóa về dạng (0,1)
                Resize(input_size),
                SubtractMeans(color_mean) #trừ đi mean của BGR
            ]),
            "val": Compose([
                ConvertFromInts(),
                Resize(input_size),
                SubtractMeans(color_mean)
            ])
                
        }
    def __call__(self, img, phase, boxes, labels):
        return self.data_transform[phase](img, boxes, labels)
if __name__ == "__main__":
    
    #prepare train , val , annotations
    root_path = "D:\project_python_2\object_detection\data\VOCdevkit\VOC2012"
    train_img_list , train_annotation_list , val_img_list , val_annotation_list = make_datapath_list(root_path)
    classes = ["aeroplane", "bicycle", "bird",  "boat", "bottle", 
               "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant",
               "sheep", "sofa", "train", "tvmonitor"]
    #read img
    img_file_path = train_img_list[0]
    img = cv2.imread(img_file_path)
    height, width, channel = img.shape

    #annotation infomation
    trans_anno = Anno_xml(classes)
    
    anno_info_list = trans_anno(train_annotation_list[0], width, height)
    
    #plot original image
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()
    #prepare data transformations
    color_mean = (104, 117, 123)# gg search color mean for voc data
    input_size = 300
    transform = DataTransform(input_size, color_mean)
    #transform tranin img
    phase = "train"
    img_transfomred , boxes, labels = transform(img, phase, anno_info_list[:,:4],anno_info_list[:, 4])
    
    plt.imshow(cv2.cvtColor(img_transfomred, cv2.COLOR_BGR2RGB))
    plt.show()
    #transform val img
    phase = "val"
    img_transfomred , boxes, labels = transform(img, phase, anno_info_list[:,:4],anno_info_list[:, 4])
    plt.imshow(cv2.cvtColor(img_transfomred, cv2.COLOR_BGR2RGB))
    plt.show()