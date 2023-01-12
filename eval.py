import numpy as np
import torch
from torchmetrics import AUROC

def hit(gt_item, pred_items):
    """ whether gt_item is in pred_items or not """
    if gt_item in pred_items:
        return 1
    return 0


def ndcg(gt_item, pred_items):
	if gt_item in pred_items:
		index = pred_items.index(gt_item)
		return np.reciprocal(np.log2(index+2))
	return 0


def metrics(model, test_loader, top_k):
    """ Calculate Metrics (HR, NDCG, AUROC) """
    HR, NDCG = [], []
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    for user, item, label in test_loader:
        user = user.to(device)
        item = item.to(device)

        
        score = model(user, item)
        _, indices = torch.topk(score, top_k)
        recommends = []
        for idx in indices:
            recommends.append(item[idx].cpu().numpy())
        
        gt_item = item[0].item()

        auroc = AUROC(task='binary')
        predict_auroc = auroc(recommends,gt_item)
        HR.append(hit(gt_item, recommends))
        NDCG.append(ndcg(gt_item, recommends))

    return np.mean(HR), np.mean(NDCG), predict_auroc