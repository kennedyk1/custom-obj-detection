import os
import cv2
import glob
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

def load_bbox(filepath,img_size):
    classes = []
    bboxes = []
    try:
        with open(filepath,'r') as f:
            tmp = f.readlines()
        tmp = tmp.split('\n')
        for i in tmp:
            classes.append(i.split(' ')[0])
            xc, yc, w, h = i.split(' ')[1:]
            xmin = int((xc - w/2)*img_size)
            xmax = int((xc + w/2)*img_size)
            ymin = int((yc - h/2)*img_size)
            ymax = int((yc + h/2)*img_size)
            bboxes.append([xmin,ymin,xmax,ymax]) 
    except:
        pass
    return {'cls':classes,'bboxes':bboxes}

#CLASS TO LOAD DATASET
class LoadDataset(Dataset):
    def __init__(self, dir_path, classes, img_size=640, transforms=None):
        self.transforms = transforms
        self.dir_path = dir_path
        self.img_size = img_size
        self.classes = classes
        
        #GET ALL THE IMAGE PATHS IN SORTED ORDER
        #self.image_paths = glob.glob(f"{self.dir_path}/*.jpg")
        image_paths = []
        a = self.dir_path

        files = os.listdir(self.dir_path)
        for file in files:
            if '.jpg' in file:
                image_paths.append(os.path.join(self.dir_path,file))
        self.image_paths = image_paths
        
        self.all_images = [image_path.split('/')[-1] for image_path in self.image_paths]
        #a = [image_path.split('/')[-1] for image_path in self.image_paths]
        #print(a)
        self.all_images = sorted(self.all_images)
        
    def __getitem__(self, idx):
        #CAPTURE IMAGE AND PATH
        image_name = self.all_images[idx]
        image_path = os.path.join(self.dir_path, image_name)
        print(image_path)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image_resized = cv2.resize(image, (self.width, self.height))
        image_resized /= 255.0
        
        #GET CORRESPONDING FILE ANNOTATION
        annot_filename = image_name[:-4] + '.txt'
        annot_file_path = os.path.join(self.dir_path, annot_filename)
        
        data = load_bbox(annot_file_path,self.img_size)
        bboxes = data['bboxes']
        labels = data['cls']
        
        #CONVERT ALL TO TENSORS
        bboxes = torch.as_tensor(bboxes,dtype=torch.float32)
        area = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
        labels = torch.as_tensor(labels,dtype=torch.int64)
        
        #GET IMAGE SIZE
        image_width = image.shape[1]
        image_height = image.shape[0]
        
        target = {}
        target['boxes'] = bboxes
        target['labels'] = labels
        image_id = torch.tensor([idx])
        
        #APPLY THE IMAGE TRANSFORMS
        if self.transforms:
            sample = self.transforms(image = image_resized,
                                     bboxes = target['boxes'],
                                     labels = labels)
            image_resized = sample['image']
            target['boxes'] = torch.Tensor(sample['bboxes'])
            
        return image_resized, target

    def __len__(self):
        return len(self.all_images)
    
train_dataset = LoadDataset('dataset/train',['Arduino_Nano', 'ESP8266', 'Heltec_ESP32_Lora', 'Raspberry_Pi_3'],None)
print(train_dataset.__len__())
CLASSES = ['Arduino_Nano', 'ESP8266', 'Heltec_ESP32_Lora', 'Raspberry_Pi_3']

def visualize_sample(image, target):
    box = target['boxes'][0]
    label = CLASSES[target['labels']]
    cv2.rectangle(
        image, 
        (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
        (0, 255, 0), 2
    )
    cv2.putText(
        image, label, (int(box[0]), int(box[1]-5)), 
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
    )
    cv2.imshow('Image', image)
    cv2.waitKey(0)
    
NUM_SAMPLES_TO_VISUALIZE = 5
for i in range(NUM_SAMPLES_TO_VISUALIZE):
    image, target = train_dataset[i]
    print(image)
    print(target)
    visualize_sample(image, target)