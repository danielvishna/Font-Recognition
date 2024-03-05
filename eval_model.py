import h5py
import os
from torchvision.transforms import v2
import torch
from torch.utils.data import DataLoader
import csv
from collections import Counter
from tqdm import tqdm
from utility import font2id
from CharDataset import CharDataset
from CharClassifier import CharClassifier


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
    data_dir = ""
    model = CharClassifier(len(font2id)).to(device)
    model.load_state_dict(torch.load('bast_state.pkl', map_location=torch.device('cpu')))
    transform = v2.Compose([v2.ToDtype(torch.float32), v2.Normalize(
        (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    test_file_loc = os.path.join(data_dir, 'test.h5')
    test_db = h5py.File(test_file_loc, 'r')
    test_dataset = CharDataset(test_db, transform=transform)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    num_classes = len(font2id)
    model = model.to(device)
    model.eval()

    true_labels = []
    predicted_probs = []

    correct = 0
    total = 0
    pre_img_ind, pre_word_ind = 0, 0
    index = []
    pred_ls = []
    out_pro = torch.zeros(num_classes)

    with open(data_dir + 'max_pred_word.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['ind', 'font'])
        with torch.no_grad():
            for i, data in enumerate(tqdm(test_dataset), 0):
                img = data[0]
                c = data[1]
                img_ind = data[2]
                word_ind = data[3]
                to_eval = data[4]

                if img_ind != pre_img_ind or pre_word_ind != word_ind:
                    count_pred = Counter(pred_ls)
                    max_pred = max(pred_ls, key=lambda pred: [count_pred[pred],
                                                              out_pro[pred]])
                    for idx in index:
                        writer.writerow([idx, max_pred])
                    index = []
                    pred_ls = []
                    out_pro = torch.zeros(num_classes)
                    pre_img_ind = img_ind
                    pre_word_ind = word_ind
                if to_eval:
                    img = img.float()
                    img = img.unsqueeze(0).to(device)
                    outputs = model(img)
                    _, predicted = torch.max(outputs.data, 1)
                    predicted = int(predicted)
                    pred_ls.append(predicted)
                    out_pro += outputs[0].to('cpu')
                index.append(i)

        data = Counter(pred_ls)
        max_pred = max(pred_ls, key=lambda pred: [data[pred], out_pro[pred]])
        for idx in index:
            writer.writerow([idx, max_pred])
