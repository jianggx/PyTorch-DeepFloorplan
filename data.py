from importmod import *
import pandas as pd
import random
import os
from skimage.transform import rotate
from torch.utils.data import Dataset, DataLoader

class MyRotationTransform:
    def __init__(self,angles=[0,90,-90.180]):
        self.angles = angles
    def _r(self,x,angle):
        return rotate(x,angle,preserve_range=True)
    def __call__(self,x,y,z,g):
        angle = random.choice(self.angles)
        return self._r(x,angle),self._r(y,angle),self._r(z,angle),self._r(g,angle)

class r3dDataset(Dataset):
    def __init__(self,csv_file='r3d.csv',size=512,transform=None):
        self.df = pd.read_csv(csv_file)
        #self.df2 = pd.read_csv('r3d2.csv')
        self.size = size
        self.transform = transform
        self.rotation = MyRotationTransform()
    def __len__(self):
        return self.df.shape[0]#+self.df2.shape[0]
    def _getset(self,idx): 
        target = self.df #if idx < self.df.shape[0] else self.df2
        idx = idx if idx < self.df.shape[0] else idx-self.df.shape[0]
        image = np.fromstring(target.loc[idx]['image'][1:-1],
                dtype=np.uint8,sep=', ').reshape(self.size,self.size,3)
        boundary = np.fromstring(target.loc[idx]['boundary'][1:-1],
                dtype=np.uint8,sep=', ').reshape(self.size,self.size)
        room = np.fromstring(target.loc[idx]['room'][1:-1],
                dtype=np.uint8,sep=', ').reshape(self.size,self.size)
        door = np.fromstring(target.loc[idx]['door'][1:-1],
                dtype=np.uint8,sep=', ').reshape(self.size,self.size)
        return image,boundary,room,door
    def __getitem__(self,idx):
        image,boundary,room,door = self._getset(idx)
        #image,boundary,room,door = self.rotation(image,boundary,room,door)
        if self.transform:
            image = self.transform(image.astype(np.float32)/255.0)
            boundary = self.transform(F.one_hot(
                torch.LongTensor(boundary),3).numpy())
            room = self.transform(F.one_hot(
                torch.LongTensor(room),9).numpy())
            door = self.transform(door)
        return image,boundary,room,door

class FolderDataset(Dataset):
    def __init__(self,folder,size=512,transform=None):
        self.folder = folder
        self.size = size
        self.transform = transform
        self.rotation = MyRotationTransform()

        self.file_item_list = []
        self.add_files(folder)
        self.cache_for_item = {}

    def add_files(self, folder):
        for file in os.listdir(folder):
            file_path = os.path.join(folder, file)
            if not os.path.exists(file_path):
                continue
            
            if not os.path.isfile(file_path):
                continue

            if file.endswith('_room.png'):
                base_path = file[:-9]
                img_path = f"{folder}/{base_path}.png"
                room_path = f"{folder}/{base_path}_room.png"
                door_path = f"{folder}/{base_path}_door.png"
                boundary_path = f"{folder}/{base_path}_boundary.png"
                if not os.path.exists(img_path) or not os.path.exists(room_path) or not os.path.exists(door_path) or not os.path.exists(boundary_path):
                    print(f'ERROR: file miss for {base_path}')
                self.file_item_list.append({'img_path':img_path, 'room_path':room_path, 'door_path':door_path, 'boundary_path':boundary_path})

    def __len__(self):
        return len(self.file_item_list)
    
    def _get_file_item_for_path(self, file_path):
        if file_path in self.cache_for_item:
            return self.cache_for_item[file_path]
        return None

    def _getset(self,idx): 
        file_item = self.file_item_list[idx]

        file_path = file_item['img_path']
        image = self._get_file_item_for_path(file_path)
        if image is None:
            image_raw = cv2.imread(file_path)
            image = cv2.resize(image_raw,(self.size,self.size))
            self.cache_for_item[file_path] = image

        file_path = file_item['boundary_path']
        boundary = self._get_file_item_for_path(file_path)
        if boundary is None:
            image_raw = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image_raw,(self.size,self.size), interpolation=cv2.INTER_NEAREST)
            self.cache_for_item[file_path] = boundary

        file_path = file_item['room_path']
        room = self._get_file_item_for_path(file_path)
        if room is None:
            image_raw = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image_raw,(self.size,self.size), cv2.INTER_NEAREST)
            self.cache_for_item[file_path] = room

        file_path = file_item['door_path']
        door = self._get_file_item_for_path(file_path)
        if door is None:
            image_raw = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image_raw,(self.size,self.size), cv2.INTER_NEAREST)
            self.cache_for_item[file_path] = door

        return image,boundary,room,door
    
    def __getitem__(self,idx):
        image,boundary,room,door = self._getset(idx)
        #image,boundary,room,door = self.rotation(image,boundary,room,door)
        if self.transform:
            image = self.transform(image.astype(np.float32)/255.0)
            boundary = self.transform(F.one_hot(
                torch.LongTensor(boundary),3).numpy())
            room = self.transform(F.one_hot(
                torch.LongTensor(room),9).numpy())
            door = self.transform(door)
        return image,boundary,room,door
    
if __name__ == "__main__":

    import matplotlib.pyplot as plt
    DFPdataset = r3dDataset()
    image,boundary,room,door = DFPdataset[200]

    plt.subplot(2,2,1); plt.imshow(image)
    plt.subplot(2,2,2); plt.imshow(boundary)
    plt.subplot(2,2,3); plt.imshow(room)
    plt.subplot(2,2,4); plt.imshow(door)
    plt.show()

    breakpoint()
    
    gc.collect()

