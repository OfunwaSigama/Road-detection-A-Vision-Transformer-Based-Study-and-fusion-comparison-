import numpy as np


class RoadMetrics:
    def __init__(self):
        self.reset()

    def reset(self):
        self.tp = self.fp = self.fn = self.tn = 0.0

    def update(self, pred, target):
        p = pred.cpu().numpy().flatten()
        t = target.cpu().numpy().flatten()
        rp, rt = (p == 1), (t == 1)
        self.tp += np.sum(rp & rt)
        self.fp += np.sum(rp & ~rt)
        self.fn += np.sum(~rp & rt)
        self.tn += np.sum(~rp & ~rt)

    def compute_all(self):
        eps = 1e-10
        road_iou = self.tp / (self.tp + self.fp + self.fn + eps)
        background_iou = self.tn / (self.tn + self.fp + self.fn + eps)
        mean_iou = (road_iou + background_iou) / 2
        precision = self.tp / (self.tp + self.fp + eps)
        recall = self.tp / (self.tp + self.fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        accuracy = (self.tp + self.tn) / (self.tp + self.fp + self.fn + self.tn + eps)
        fpr = self.fp / (self.fp + self.tn + eps)
        fnr = self.fn / (self.fn + self.tp + eps)

        return {
            'road_iou': float(road_iou),
            'background_iou': float(background_iou),
            'mean_iou': float(mean_iou),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'accuracy': float(accuracy),
            'fpr': float(fpr),
            'fnr': float(fnr)
        }
