from sklearn import metrics
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from analysis import ChunkVisualiser


class ClassificationVisualiser:

    def plot_confusion_matrix(self, labels, test_labels, le):
        confusion_matrix = metrics.confusion_matrix(test_labels, labels).astype(np.float32)
        confusion_matrix = confusion_matrix / confusion_matrix.sum(axis=1)[:, np.newaxis]
        class_names = le.classes_
        df_cm = pd.DataFrame(confusion_matrix, index=class_names, columns=class_names)
        plt.figure(figsize=(10, 7))
        sns.heatmap(df_cm, annot=True)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.show(block=False)

    def visualise_incorrect_classifications(self, pred_labels, test_labels, le, chunks, frames, translated_chunks, videos):
        visualiser = ChunkVisualiser(chunks, frames, translated_chunks)
        unique_labels = set(pred_labels)
        for pred_label in unique_labels:
            for true_label in unique_labels:
                if pred_label == -1 or true_label == -1 or pred_label == true_label:
                    continue

                pred_class_member_mask = (pred_labels == pred_label)
                true_class_member_mask = (test_labels == true_label)
                node = np.where(pred_class_member_mask & true_class_member_mask)[0]
                name = "P {}, T {}".format(le.classes_[pred_label], le.classes_[true_label])

                if len(node) != 0:
                    repeat = 'y'
                    while repeat == 'y':
                        visualiser.draw_node(videos, name, node)
                        repeat = input("again? (y/n)")
