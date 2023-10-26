from lib import *

def make_datapath_list(root_path):
    # tạo đường link mẫu của ảnh và anno
    image_path_template = osp.join(root_path, "JPEGimages", "%s.jpg")
    annotation_path_template = osp.join(root_path, "Annotation", "%s.xml")
    # tạo path dẫn đến train và test data
    train_id_names = osp.join(root_path, "ImageSets/Main/train.txt")
    val_id_names = osp.join(root_path, "ImageSets/Main/val.txt")
    #tạo list chứa các link trên
    train_img_list = []
    train_annotation_list = []
    val_img_list = []
    val_annotation_list = []
    # đưa các link vào train list
    for line in open(train_id_names):
        # lấy id
        file_id= line.strip() # xóa kí tự xuống dòng, spaces 
        img_path = (image_path_template % file_id)
        # đưa cái file id vào cái image_path_template để tạo thành đuôi .jpg
        anno_path = (annotation_path_template % file_id)
        # annotation_path_template lấy ra như trên
        
        train_img_list.append(img_path)
        train_annotation_list.append(anno_path)
    # đưa các link vào val list
    for line in open(val_id_names):
        file_id= line.strip()
        img_path = (image_path_template % file_id)
        anno_path = (annotation_path_template % file_id)
        
        val_img_list.append(img_path)
        val_annotation_list.append(anno_path)

    return val_img_list, val_annotation_list , train_img_list, train_annotation_list

if __name__ == "__main__":
    root_path = "D:\project_python_2\object_detection\data\VOCdevkit\VOC2012"
    train_img_list , train_annotation_list , val_img_list , val_annotation_list = make_datapath_list(root_path)
    print(len(train_img_list))
    print(train_img_list[0])
    
    