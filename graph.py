from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, roc_curve, auc, average_precision_score, precision_recall_curve
import numpy as np
import torch
from collections import Counter
from utility import id2font, font2id

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_classes = len(id2font)


def plot_confusion_matrix(model, val_dataset):
    model = model.to(device)
    predicted_labels = []
    true_labels = []
    model.eval()
    pre_img_ind, pre_word_ind = 0, 0
    pred_ls = []
    out_pro = torch.zeros(num_classes)
    with torch.no_grad():
        for inputs, chars, labels, img_ind, word_ind in val_dataset:
            if img_ind != pre_img_ind or pre_word_ind != word_ind:
                count_pred = Counter(pred_ls)
                max_pred = max(pred_ls, key=lambda i: [count_pred[i],
                                                       out_pro[i]])
                predicted_labels.extend([max_pred] * len(pred_ls))
                pred_ls = []
                out_pro = torch.zeros(num_classes)
                pre_img_ind = img_ind
                pre_word_ind = word_ind

            inputs = inputs.to(device)
            img = inputs.float()
            img = img.unsqueeze(0).to(device)
            outputs = model(img)
            _, predicted = torch.max(outputs.data, 1)
            predicted = int(predicted)
            pred_ls.append(predicted)
            out_pro += outputs[0].to('cpu')
            true_labels.append(labels)

    count_pred = Counter(pred_ls)
    max_pred = max(pred_ls, key=lambda i: [count_pred[i], out_pro[i]])
    predicted_labels.extend([max_pred] * len(pred_ls))
    cm = confusion_matrix(true_labels, predicted_labels)

    plt.figure(figsize=(10, 10))
    plt.imshow(cm, cmap='Blues')
    plt.colorbar()
    plt.xticks(np.arange(len(font2id)), list(font2id.keys()), rotation=45)
    plt.yticks(np.arange(len(font2id)), list(font2id.keys()))
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig("Confusion Matrix.png")
    plt.show()


def plot_precision_roc(model, val_dataloader):
    model.eval()

    true_labels = []
    predicted_probs = []

    correct = 0
    total = 0

    with torch.no_grad():
        for (img, c, font, _, __) in val_dataloader:
            img = img.unsqueeze(0).to(device)[0]
            font = font.to(device)
            outputs = model(img)
            _, predicted = torch.max(outputs.data, 1)

            total += font.size(0)
            correct += int((predicted == font).sum())

            true_labels.extend(font.cpu().numpy())
            predicted_probs.extend(torch.softmax(outputs, dim=1).cpu().numpy())

    predicted_probs = np.array(predicted_probs)

    # Convert true labels to one-hot encoding
    true_labels_onehot = torch.eye(num_classes)[true_labels].numpy()

    # Calculate ROC curve and AUC for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    precisions = dict()
    recalls = dict()
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(true_labels_onehot[:, i], predicted_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        precisions[i], recalls[i], _ = precision_recall_curve(true_labels_onehot[:, i],
                                                              predicted_probs[:, i])

    # Plot ROC curve for each class
    plt.figure(figsize=(10, 8))
    for i in range(num_classes):
        plt.plot(fpr[i], tpr[i], lw=2,
                 label=f'ROC curve (class {id2font[i]}) (area = {round(roc_auc[i], 4)})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve for each class')
    plt.savefig('ROC.png')

    plt.legend()
    plt.show()

    # Generate Precision-Recall curve
    precision, recall, _ = precision_recall_curve(true_labels_onehot.ravel(), np.array(predicted_probs).ravel())
    average_precision = average_precision_score(true_labels_onehot, np.array(predicted_probs), average='weighted')

    # Plot Precision-Recall curve
    plt.figure(figsize=(8, 8))
    plt.step(recall, precision, color='b', where='post',
             label=f'Precision-Recall curve (AP = {round(average_precision, 4)})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.savefig('precision recall class curve.png')
    plt.legend()
    plt.show()

    precision_per_class = precision_score(true_labels, np.argmax(predicted_probs, axis=1), average=None)

    plt.figure(figsize=(10, 8))
    for i in range(num_classes):
        plt.plot(recalls[i], precisions[i], lw=2, label=f'{id2font[i]} class (AP={round(precision_per_class[i], 4)})')

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="best")
    plt.title("Precision-Recall Curve for each class")
    plt.savefig('precision recall class curve.png')
    plt.show()

    # Calculate precision for each class
    precision_per_class = precision_score(true_labels, np.argmax(predicted_probs, axis=1), average=None)

    # Print precision for each class
    for i, precision in enumerate(precision_per_class):
        print(f'Precision for class {i}: {round(precision, 4)}')

    # Calculate overall accuracy
    accuracy = correct / total
    print(f'Accuracy on the test set: {accuracy}')
