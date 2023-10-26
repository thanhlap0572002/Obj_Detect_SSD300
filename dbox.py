from lib import *

cfg = {
    "num_classes" : 21, #VOC data include 20classes + 1 class background
    "input_size" : 300, #sdd 300
    "bbox_aspect_num" : [4, 6, 6, 6, 4, 4],# source1 -> source6  tỉ lệ khung hình các source  , trọng số cho các source của mô hình
    "feature_maps" : [38, 19, 10, 5, 3, 1], # đặc trưng ảnh kiểu 38x38
    "steps" : [8, 16, 32, 64, 100, 300], # SIZE OF  DBOX
    "min_size" : [30, 60, 111, 162, 213, 264], #size of dbox
    "max_size" : [60, 111, 162, 213, 264, 315],# size of dbox
    "aspect_ratios" : [[2] , [2,3] , [2,3] , [2,3] , [2] , [2]],
}
class DefBox():
    def __init__(self, cfg):    
        self.img_size = cfg["input_size"]
        self.feature_maps = cfg["feature_maps"]
        self.min_size = cfg["min_size"]
        self.max_size = cfg["max_size"]
        self.aspect_ratios = cfg["aspect_ratios"]
        self.steps = cfg["steps"]
    
    def create_defbox(self): 
        defbox_list = []
        
        for k, f in enumerate(self.feature_maps): # k là scale , f là feature 
            for i, j in itertools.product(range(f), repeat = 2):
                f_k = self.img_size / self.steps[k] #f_k là size của các box feature map : 300/8
                
                cx = (j+0.5)/f_k
                cy = (i+0.5)/f_k
                #small square box
                s_k = self.min_size[k] / self.img_size # 30/300
                defbox_list += [cx, cy, s_k, s_k]
                
                #big square box
                s_k_ = sqrt(s_k*(self.max_size[k] / self.img_size))
                defbox_list += [cx, cy, s_k_, s_k_]
                
                for ar in self.aspect_ratios[k]:
                    defbox_list += [cx, cy, s_k*sqrt(ar), s_k/sqrt(ar)]
                    defbox_list += [cx, cy, s_k/sqrt(ar), s_k*sqrt(ar)]
        output = torch.Tensor(defbox_list).view(-1, 4)
        output.clamp_(max=1, min=0)
        
        return output

if __name__ == '__main__':
    defbox = DefBox(cfg)
    dbox_list = defbox.create_defbox()
    # print(debox_list)
    print(pd.DataFrame(dbox_list.numpy()))
    
    