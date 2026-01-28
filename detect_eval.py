import argparse
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np
import torch
from sklearn.metrics import roc_curve
import numpy as np

from scipy.stats import rankdata
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import numpy as np

from sklearn.preprocessing import StandardScaler

parser = argparse.ArgumentParser(description="diffusion memorization detection eval")
parser.add_argument("--path", default='./det_outputs/ablation3/sd2_mem_gen1_modex,c|x_seed51.pt', type=str)
parser.add_argument("--npath", default='./det_outputs/ablation3/sd2_nmem_gen1_modex,c|x_seed51.pt', type=str)
parser.add_argument("--path_cosine", default='./det_outputs/ablation3/sd2_mem_cosine_gen1_modex,c|x_seed51.pt', type=str)
parser.add_argument("--npath_cosine", default='./det_outputs/ablation3/sd2_nmem_cosine_gen1_modex,c|x_seed51.pt', type=str)
parser.add_argument("--sd_ver", default=2, type=int)
args = parser.parse_args()

def compute_tpr_at_thresholds(labels, scores):
    fpr, tpr, thresholds = roc_curve(labels, scores)

    tpr_lst = []
    thre_fpr = [0.01, 0.03]  
    for thres in thre_fpr:
        target_fpr = thres
        closest_fpr_index = np.argmin(np.abs(fpr - target_fpr))
        tpr_at_target_fpr = tpr[closest_fpr_index]
        tpr_lst.append(tpr_at_target_fpr)

    return tpr_lst

#TODO: manually insert path for comparison
#SD2
if args.sd_ver == 1:
    gamma_1= 1.0 ###norm
    gamma_2= 2.0 ###cosine
elif args.sd_ver == 2:
    gamma_1= 1.0 ###norm
    gamma_2= 0.1 ###cosine
else:
    gamma_1= 2 ###norm
    gamma_2= 1 ###cosine

#ablation
#gamma_2=0.0

#SD1
#gamma_1=1.0
#gamma_2=2.0

mem_data, nmem_data = torch.load(args.path, weights_only=True),\
                        torch.load(args.npath, weights_only=True)
mem_cosine, nmem_cosine = torch.load(args.path_cosine, weights_only=True),\
                           torch.load(args.npath_cosine, weights_only=True)

mem_data, nmem_data = mem_data.mean(dim=1).float().numpy(), nmem_data.mean(dim=1).float().numpy()
mem_cosine, nmem_cosine = mem_cosine.mean(dim=1).float().numpy(), nmem_cosine.mean(dim=1).float().numpy()


for i in range(len(mem_data)):
    mem_data[i] = mem_data[i]
for i in range(len(nmem_data)):
    nmem_data[i] = nmem_data[i]
scores, labels = np.concatenate([mem_cosine, nmem_cosine]), np.concatenate([np.ones(mem_data.shape[0]), np.zeros(nmem_data.shape[0])])
auroc = roc_auc_score(labels, scores)

if auroc < 0.5:
    #change the labels if lower than 0.5
    auroc = auroc if auroc > 0.5 else 1 - auroc
    labels = 1 - labels

tpr_lst = compute_tpr_at_thresholds(labels, scores)
#print(f'AUC: {auroc:.3f} | TPR@1%FPR: {tpr_lst[0]:.3f} | TPR@3%FPR: {tpr_lst[1]:.3f}')

cosine_sims = []
text_noise_norm = []
combined_score=[]
memorised_bool=[]
lambda_max=[]
uncond_noise_norm = []
hessian = []
svd_cn = []
for i in range(len(mem_data)):
    hessian.append(mem_data[i])
    cosine_sims.append(mem_cosine[i])
    combined_score.append((gamma_1*mem_data[i] + gamma_2*mem_cosine[i]))
    memorised_bool.append(1)
for i in range(len(nmem_data)):
    memorised_bool.append(0)
    hessian.append(nmem_data[i])
    cosine_sims.append(nmem_cosine[i])
    combined_score.append((gamma_1*nmem_data[i] + gamma_2*nmem_cosine[i]))

auc = roc_auc_score(memorised_bool, combined_score)
tpr_lst = compute_tpr_at_thresholds(memorised_bool, combined_score)

print(f'AUC: {auc:.3f} | TPR@1%FPR: {tpr_lst[0]:.3f} | TPR@3%FPR: {tpr_lst[1]:.3f}')