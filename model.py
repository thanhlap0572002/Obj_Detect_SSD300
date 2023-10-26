from lib import *
from l2_norm import L2Norm
from dbox import DefBox
def create_vgg():
    layers = []
    in_channels = 3
    
    cfgs = [64, 64, "M", 128, 128, "M",
            256, 256, 256, "MC", 512, 512, 512, "M",
            512, 512, 512]
    for cfg in cfgs :
        if cfg == "M":# kiểu max mặc định làm tròn xuống
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif cfg == "MC": # làm tròn lên
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, cfg, kernel_size = 3, padding = 1)
            
            layers += [conv2d, nn.ReLU(inplace= True)]
            in_channels = cfg
        
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    # padding in convolution network ?
    
    conv6 = nn.Conv2d(512, 1024, kernel_size=3,padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=3)

    layers += [pool5, conv6, nn.ReLU(inplace=True), conv7, nn.ReLU(inplace= True)]
    
    return nn.ModuleList(layers)

def create_extras():
    layers = []
    in_channel = 1024
    
    cfgs = [256, 512, 128, 256, 128, 256, 128, 256]
    
    layers += [nn.Conv2d(in_channel,cfgs[0],kernel_size=1), nn.ReLU(inplace= True)]
    layers += [nn.Conv2d(cfgs[0], cfgs[1], kernel_size=3, stride=2, padding=1), nn.ReLU(inplace= True)]
    layers += [nn.Conv2d(cfgs[1], cfgs[2], kernel_size=1), nn.ReLU(inplace= True)]
    layers += [nn.Conv2d(cfgs[2], cfgs[3], kernel_size=3, stride=2, padding=1), nn.ReLU(inplace= True)]
    layers += [nn.Conv2d(cfgs[3], cfgs[4], kernel_size=1), nn.ReLU(inplace= True)]
    layers += [nn.Conv2d(cfgs[4], cfgs[5], kernel_size=3, stride=2, padding=1), nn.ReLU(inplace= True)]
    layers += [nn.Conv2d(cfgs[5], cfgs[6], kernel_size=1), nn.ReLU(inplace= True)]
    layers += [nn.Conv2d(cfgs[6], cfgs[7], kernel_size=3, stride=2, padding=1), nn.ReLU(inplace= True)]

    return nn.ModuleList(layers)

def create_loc_conf(num_classes = 21, bbox_ratio_num = [4,6,6,6,4,4]):
    layers_loc = []
    layers_conf = []
    
    # source 1
    #loc
    layers_loc += [nn.Conv2d(512, bbox_ratio_num[0]*4, kernel_size=3, padding=1)]
    #conf
    layers_conf += [nn.Conv2d(512, bbox_ratio_num[0]*num_classes, kernel_size=3, padding=1)]
    
    # source 2
    #loc
    layers_loc += [nn.Conv2d(1024, bbox_ratio_num[1]*4, kernel_size=3, padding=1)]
    #conf
    layers_conf += [nn.Conv2d(1024, bbox_ratio_num[1]*num_classes, kernel_size=3, padding=1)]
    
    # source 3
    #loc
    layers_loc += [nn.Conv2d(512, bbox_ratio_num[2]*4, kernel_size=3, padding=1)]
    #conf
    layers_conf += [nn.Conv2d(512, bbox_ratio_num[2]*num_classes, kernel_size=3, padding=1)]
    
    # source 4
    #loc
    layers_loc += [nn.Conv2d(256, bbox_ratio_num[3]*4, kernel_size=3, padding=1)]
    #conf
    layers_conf += [nn.Conv2d(256, bbox_ratio_num[3]*num_classes, kernel_size=3, padding=1)]
    
    # source 5
    #loc
    layers_loc += [nn.Conv2d(256, bbox_ratio_num[4]*4, kernel_size=3, padding=1)]
    #conf
    layers_conf += [nn.Conv2d(256, bbox_ratio_num[4]*num_classes, kernel_size=3, padding=1)]
    
    # source 6
    #loc
    layers_loc += [nn.Conv2d(256, bbox_ratio_num[5]*4, kernel_size=3, padding=1)]
    #conf
    layers_conf += [nn.Conv2d(256, bbox_ratio_num[5]*num_classes, kernel_size=3, padding=1)]
    
    return nn.ModuleList(layers_loc) , nn.ModuleList(layers_conf)

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

class SSD(nn.Module):
    def __init__(self, phase, cfg):
        super(SSD, self).__init__()
        self.phase = phase
        self.num_classes = cfg["num_classes"]
        
        #create main modules
        self.vgg = create_vgg()
        self.extras = create_extras()
        self.loc, self.conf = create_loc_conf(cfg["num_classes"], cfg["bbox_aspect_num"])
        self.L2norm = L2Norm()
        
        #default box create
        dBox = DefBox(cfg)
        self.dbox_list = dBox.create_defbox()
        
        if phase == "inference":
            self.detect = Detect()
    
    def forward(self, x): # x là ảnh đầu vào
        sources = list()
        loc = list()
        conf = list()
        
        for k in range(23):
            x = self.vgg[k][x]
        #source1
        source1 = self.L2norm
        sources.append(source1)
        #source2
        for k in range(23, len(self.vgg)):
            x= self.vgg[k](x)
        sources.append(x)
        #source3->6
        for k, v in enumerate(self.extras): #for index, value in enumerate(sequence):
            x= nn.ReLU(v(x), inplace=True)
        if k % 2 == 1 :
            sources.append(x)
        
        for(x, l ,c) in zip(sources, self.loc, self.conf):
            #aspect_ratio_num = 4, 6
            #(batch_size, 4*aspect_ratio_num, featuremap_hight, featuremap_width)
            #->(batch_size, featuremap_hight, featuremap_width, 4*aspect_ratio_num,)
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1)for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1)for o in conf], 1)
        
        loc = loc.view(loc.size(0), -1, 4)
        conf = conf.view(conf.size(0), -1, 21)
        
        output = (loc, conf, self.dbox_list)
        
        if self.phase == "inference":
            return self.detect(output[0], output[1], output[2])
        else:
            return output
        
def decode(loc, defbox_list):
    """
    parameters :
    loc : [8732, 4] (deltaX, deltaY, deltaW, deltaH)
    defbox_list : [8732, 4](cx_d, cy_d, w_d, h_d)
    
    return :
    boxes[xmin, ymin, xmax, ymax]
    """
    # công thức : cx = cx_d(1+0.1delta_cX) 
    #             cy = cy_d(1+0.1delta_cy)
    #             w = w_d*exp(0.2delta_W)
    #             h = h_d*exp(0.2delta_H)
    # a4 kẹp vở đen
    boxes = torch.cat((
        defbox_list[:, :2] + 0.1*loc[:, :2] * defbox_list[:, :2],
        defbox_list[:, 2:] * torch.exp(loc[:, 2:]*0.2)), dim = 1)
    
    boxes[:, :2] -= boxes[:, 2:] / 2 # tính xmin ymin
    boxes[:, 2:] +=boxes[:, :2] # xmax , ymax

    return boxes
#non-maximum-supression
def nms(boxes, scores, overlap=0.45, top_k = 200):
    """
    boxes : [num_boxes, 4]-(xmin, ymin, xmax, ymax)
    scores : [num_boxes] # độ tự tin
    
    """
    count = 0
    keep = scores.new(scores.size(0)).zero_().long()
    
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    #area of boxes - diện tích boxes
    area = torch.mul(x2-x1, y2-y1) # w*h
    
    tmp_x1 = boxes.new()
    tmp_y1 = boxes.new()
    tmp_x2 = boxes.new()
    tmp_y2 = boxes.new() 
    tmp_w = boxes.new()
    tmp_h = boxes.new()

    value, idx = scores.sort(0)
    idx = idx[-top_k:]#id của top 200 box có độ tự tin cao nhất
    
    while idx.numel() > 0 :
        i = idx[-1:] # id của box có độ tự tin cao nhất
        
        keep[count] = i
        count += 1
        
        if idx.size(0) == 1:
            break
        idx = idx[-1:] #id của box ngoại trừ box có độ tự tin cao nhât
        
        #infor box 
        torch.index_select(x1, 0, idx, out=tmp_x1) # lấy ra x1
        torch.index_select(x2, 0, idx, out=tmp_x2) # lấy ra x2
        torch.index_select(y1, 0, idx, out=tmp_y1) # lấy ra y1
        torch.index_select(y2, 0, idx, out=tmp_y2) # lấy ra y2

        tmp_x1 = torch.clamp(tmp_x1, min=x1[i]) # =x1[i] if tmp_x1 < x1[i]
        tmp_y1 = torch.clamp(tmp_y1, min=y1[i])
        tmp_x2 = torch.clamp(tmp_x2, max=x2[i]) # =x2[i] if tmp_x2 > x2[i]
        tmp_y2 = torch.clamp(tmp_y2, max=y2[i])
        # giảm số chiều của w, h về giống x1, y1, y2,x2
        tmp_w.resize_as_(tmp_x2)
        tmp_h.resize_as_(tmp_y2)
        
        tmp_w = tmp_x2 - tmp_x1
        tmp_h = tmp_y2 - tmp_y1
        
        tmp_w = torch.clamp(tmp_w, min=0.0)
        tmp_h = torch.clamp(tmp_h, min=0.0)
        
        #area overlap
        inter = tmp_h * tmp_w
        
        others_area = torch.index_select(area, 0, idx) # diện tích của mỗi bbox
        
        area_union = area[i] + others_area - inter
        
        iou = inter / area_union
        
        idx = idx[iou.le(overlap)]
        
    return keep, count

class Detect(Function):
    def __init__(self, conf_thresh = 0.01, top_k = 200, nms_thresh = 0.45):
        self.softmax = nn.Softmax(dim=-1)
        self.conf_thresh = conf_thresh
        self.top_k = top_k
        self.nms_thresh = nms_thresh
        
        
    def forward(self, loc_data, conf_data, dbox_list):
        num_batch = loc_data.size(0) # batch size
        num_dbox = loc_data.size(1) # 8732
        num_classes = conf_data.size(2) # 21
        
        conf_data = self.softmax(conf_data) 
        #(batch_num, num_dbox, num_classes) -> (batch_num, num_classes, num_dbox)
        conf_preds = conf_data.transpose(2, 1)
        
        output = torch.zeros(num_batch, num_classes, self.top_k, 5)
        
        #xử lý từng ảnh trong 1 bacth
        for i in range(num_batch):
            #tính bbox từ offset inf  và default box
            decode_box = decode(loc_data, dbox_list)
            # copy conf scores của ảnh thứ i
            conf_scores = conf_preds[i].clone()
            
            for cl in range(1, num_classes):
                c_mask = conf_preds[cl].gt(self.conf_thresh)# lấy ra những confidence > 0.01
                scores = conf_preds[cl][c_mask]
                
                if scores.nelements() == 0:
                    continue
                #đưa chiều về giống chiều của dcode box để tính toán
                l_mask = c_mask.unsqueeze(1).expand_as(decode_box) #(8732, 4)
                    
                boxes = decode_box[l_mask].view(-1, 4)
                
                ids, count = nms(boxes, scores, self.nms_thresh, self.top_k)
                
                output[i, cl, count] = torch.cat((scores[ids[:count]].unsqueeze(1), boxes[ids[:count]]),1)

        return output


if __name__ == '__main__':
    # vgg = create_vgg()
    # extras = create_extras()
    # loc, conf = create_loc_conf()
    
    # print(vgg)
    # print(extras)
    # print(loc)
    # print(conf)
    
    ssd = SSD(phase="train", cfg=cfg)
    print(ssd)