import os, random
import mlflow as mf, numpy as np, torch as th
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment


def enforce_reproducibility(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.backends.cudnn.benchmark = True
    th.backends.cudnn.deterministic = True

def count_params(parameters):
    parameters = filter(lambda p: p.requires_grad, parameters)
    parameters = sum([np.prod(p.size()) for p in parameters]) 
    print('no. trainable parameters: {:.3f}M'.format(parameters / 1_000_000))
    return parameters

def minimize_average_distance(tensor_a, tensor_b, device='cuda'):
  tensor_a = tensor_a.detach().cpu().numpy()
  tensor_b = tensor_b.detach().cpu().numpy()
  output = []
  for c in range(tensor_a.shape[0]):
    a,b = tensor_a[c,:,:], tensor_b[c,:,:]
    distances = cdist(a, b)
    row_ind, col_ind = linear_sum_assignment(distances)
    col_ind = th.as_tensor(col_ind)
    output.append(col_ind)
  return output
