from torch.utils.data import Dataset
from torchvision.transforms import v2
import cv2
import numpy as np
import random
from utility import font2id



class CharDataset(Dataset):
    def __init__(self, data_db, transform=None):
        self.data = data_db['data']
        self.names = list(self.data.keys())

        self.is_train = 'font' in list(self.data[self.names[0]].attrs.keys())
        txts = [self.data[im].attrs['txt'] for im in self.names]

        self.char_positions = []
        self.char_position_in_image = []
        for i, t in enumerate(txts):
            pos = 0
            for j, w in enumerate(t):
                for k, c in enumerate(w):
                    self.char_positions.append((i, j, k))
                    self.char_position_in_image.append(pos)
                    pos += 1

        if transform is None:
            self.transform = v2.ToImage()
        else:
            self.transform = v2.Compose([v2.ToImage(), transform])

    def crop_char(self, image, char_bb):
        new_bb = np.array([[0, 0], [128, 0], [128, 128], [0, 128]])
        homography_matrix = cv2.findHomography(char_bb.T, new_bb)
        aligned_image = cv2.warpPerspective(image, homography_matrix[0],
                                            (image.shape[1], image.shape[0]),
                                            flags=cv2.INTER_LINEAR)
        return aligned_image[0: 128, 0:  128, :]

    def __len__(self):
        return len(self.char_positions)

    def __getitem__(self, idx):
        img_idx, w_idx, c_idx = self.char_positions[idx]

        im = self.names[img_idx]
        image = self.data[im][:]
        font = self.data[im].attrs['font'] if self.is_train else None
        txt = self.data[im].attrs['txt']
        charBB = self.data[im].attrs['charBB']

        c = txt[w_idx].decode('UTF-8')[c_idx]
        c_pos = self.char_position_in_image[idx]
        c_bb = charBB[..., c_pos]
        crop_img = self.crop_char(image, c_bb)

        if self.transform:
            crop_img = self.transform(crop_img)

        if self.is_train:
            c_bb = c_bb.T
            for i in range(len(c_bb)):
                vert1 = c_bb[i]
                vert2 = c_bb[(i + 1) % len(c_bb)]
                length = np.sqrt(np.sum((vert1 - vert2) ** 2))

                if length == 0:
                    return self.__getitem__(random.randint(0, len(self.char_positions)))
            return crop_img, c, font2id[font[c_pos]], img_idx, w_idx
        else:
            to_eval = True
            c_bb = c_bb.T
            for i in range(len(c_bb)):
                vert1 = c_bb[i]
                vert2 = c_bb[(i + 1) % len(c_bb)]
                length = np.sqrt(np.sum((vert1 - vert2) ** 2))

                if length == 0:
                    to_eval = False
                    break

            return crop_img, c, img_idx, w_idx, to_eval