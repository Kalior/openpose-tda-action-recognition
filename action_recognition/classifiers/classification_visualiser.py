from sklearn import metrics
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging

from ..analysis import ChunkVisualiser


class ClassificationVisualiser:
    """Visualisation helper for the classifiers.

    Contains functionality for plotting a confusion matrix
    from predicted and true labels, and a method for
    visualising the incorrect classifications.
    """

    def plot_confusion_matrix(self, labels, test_labels, class_names, title):
        """Plots a confusion matrix for the given labels.

        Parameters
        ----------
        labels : array-like
            Predicted labels from the classifier.
        test_labels : array-like
            Ground truth labels.
        class_names : array-like
            The names of the classes in the data.
        title : str
            Title of the plot and name of the file.

        """
        confusion_matrix = metrics.confusion_matrix(test_labels, labels).astype(np.float32)
        confusion_matrix = confusion_matrix / confusion_matrix.sum(axis=1)[:, np.newaxis]
        df_cm = pd.DataFrame(confusion_matrix, index=class_names, columns=class_names)
        plt.figure(figsize=(10, 7))
        sns.heatmap(df_cm, annot=True)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.title(title)
        plt.tight_layout()
        plt.savefig(title + '.png', bbox_inches='tight')
        plt.show(block=False)

    def visualise_incorrect_classifications(self, pred_labels, test_labels, chunks, frames, translated_chunks, videos):
        """Displays videos of the incorrect classifications.

        Can help with identifying features that can be added to the
        classification or bad data etc.

        Parameters
        ----------
        pred_labels : array-like
            Predicted labels from the classifier.
        test_labels : array-like
            Ground truth labels.
        chunks : array-like
            The corresponding chunks for the labels.
        frames : array-like
            The frame numbers for the chunks.
        videos : array-like
            The paths to the corresponding videos of the chunks.

        """
        visualiser = ChunkVisualiser(chunks, frames, translated_chunks)
        unique_labels = set(pred_labels)
        for pred_label in unique_labels:
            for true_label in unique_labels:
                if pred_label == -1 or true_label == -1 or pred_label == true_label:
                    continue

                pred_class_member_mask = (pred_labels == pred_label)
                true_class_member_mask = (test_labels == true_label)
                node = np.where(pred_class_member_mask & true_class_member_mask)[0]
                name = "P {}, T {}".format(pred_label, true_label)

                logging.debug("\n".join("{}-{} {}, {}".format(
                    f[0], f[-1], true_label, video)
                    for f, video in zip(frames[node], videos[node])))

                if len(node) != 0:
                    repeat = 'y'
                    while repeat == 'y':
                        visualiser.draw_node(videos, name, node)
                        repeat = input("again? (y/n)")
