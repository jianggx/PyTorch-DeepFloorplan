from importmod import *
import pandas as pd
import random
import os
from skimage.transform import rotate
from torch.utils.data import Dataset, DataLoader
import PIL
import tqdm

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
    def __init__(self,base_folder,size=512,transform=None, is_test=False):
        self.base_folder = base_folder
        self.size = size
        self.transform = transform
        self.rotation = MyRotationTransform()

        self.file_item_list = []

        if is_test:
            self.add_files(os.path.join(base_folder,'test'))
        else:
            
            self.add_files(os.path.join(base_folder,'train'))
            self.add_files(os.path.join(base_folder,'val'))


    def add_files(self, folder):
        print(f'Loading {folder}...')
        for file in tqdm.tqdm(os.listdir(folder)):
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
                self.file_item_list.append({'image':self._get_numpystr_of_file(img_path, 'RGB'), 
                                            'room':self._get_numpystr_of_file(room_path, 'L'), 
                                            'door':self._get_numpystr_of_file(door_path, 'L'), 
                                            'boundary':self._get_numpystr_of_file(boundary_path, 'L')})

    def __len__(self):
        return len(self.file_item_list)
    
    def _get_numpystr_of_file(self, file_path, mode):
        img = PIL.Image.open(file_path)
        if img.mode != mode:
            img = img.convert(mode)
        img = img.resize((self.size, self.size))
        npstr = np.array(img).tostring()
        return npstr


    def _getset(self,idx): 
        file_item = self.file_item_list[idx]

        #file_path = file_item['img_path']
        image = np.fromstring(file_item['image'], dtype=np.uint8).reshape(self.size, self.size, 3)

        #file_path = file_item['boundary_path']
        boundary = np.fromstring(file_item['boundary'], dtype=np.uint8).reshape(self.size, self.size)

        #file_path = file_item['room_path']
        room = np.fromstring(file_item['room'], dtype=np.uint8).reshape(self.size, self.size)

        #file_path = file_item['door_path']
        door = np.fromstring(file_item['door'], dtype=np.uint8).reshape(self.size, self.size)

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

