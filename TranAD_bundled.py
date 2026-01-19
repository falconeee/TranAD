import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, ndcg_score
from pprint import pprint
import math
from torch.autograd import Variable
from matplotlib.backends.backend_pdf import PdfPages
import warnings

warnings.filterwarnings('ignore')

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x: x

# ==========================================
# CONFIGURATION
# ==========================================
class Config:
    def __init__(self):
        self.model = 'TranAD'
        self.test = False
        self.retrain = False
        self.less = False
        self.bs = 128  # Batch size
        self.lr = 0.001
        self.window_size = 10
        self.dataset = 'generic' # Placeholder

args = Config()

# ==========================================
# UTILS
# ==========================================
class color:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    RED = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def smooth(y, box_pts=1):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def plotter(name, y_true, y_pred, ascore, labels):
    if 'TranAD' in name: y_true = torch.roll(y_true, 1, 0)
    os.makedirs(os.path.join('plots', name), exist_ok=True)
    pdf = PdfPages(f'plots/{name}/output.pdf')
    for dim in range(y_true.shape[1]):
        y_t, y_p, l, a_s = y_true[:, dim], y_pred[:, dim], labels[:, dim], ascore[:, dim]
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        ax1.set_ylabel('Value')
        ax1.set_title(f'Dimension = {dim}')
        ax1.plot(smooth(y_t), linewidth=0.2, label='True')
        ax1.plot(smooth(y_p), '-', alpha=0.6, linewidth=0.3, label='Predicted')
        ax3 = ax1.twinx()
        ax3.plot(l, '--', linewidth=0.3, alpha=0.5)
        ax3.fill_between(np.arange(l.shape[0]), l, color='blue', alpha=0.3)
        if dim == 0: ax1.legend(ncol=2, bbox_to_anchor=(0.6, 1.02))
        ax2.plot(smooth(a_s), linewidth=0.2, color='g')
        ax2.set_xlabel('Timestamp')
        ax2.set_ylabel('Anomaly Score')
        pdf.savefig(fig)
        plt.close()
    pdf.close()

# ==========================================
# SPOT / POT
# ==========================================
class SPOT:
    def __init__(self, q=1e-4):
        self.proba = q
        self.extreme_quantile = None
        self.data = None
        self.init_data = None
        self.init_threshold = None
        self.peaks = None
        self.n = 0
        self.Nt = 0

    def fit(self, init_data, data):
        if isinstance(data, list): self.data = np.array(data)
        elif isinstance(data, np.ndarray): self.data = data
        elif isinstance(data, pd.Series): self.data = data.values
        else: return
        if isinstance(init_data, list): self.init_data = np.array(init_data)
        elif isinstance(init_data, np.ndarray): self.init_data = init_data
        elif isinstance(init_data, pd.Series): self.init_data = init_data.values
        elif isinstance(init_data, int):
            self.init_data = self.data[:init_data]
            self.data = self.data[init_data:]
        elif isinstance(init_data, float) and (init_data < 1) and (init_data > 0):
            r = int(init_data * data.size)
            self.init_data = self.data[:r]
            self.data = self.data[r:]
        else: return

    def add(self, data):
        if isinstance(data, list): data = np.array(data)
        elif isinstance(data, np.ndarray): data = data
        elif isinstance(data, pd.Series): data = data.values
        else: return
        self.data = np.append(self.data, data)

    def initialize(self, level=0.98, min_extrema=False, verbose=True):
        if min_extrema:
            self.init_data = -self.init_data
            self.data = -self.data
            level = 1 - level
        level = level - math.floor(level)
        n_init = self.init_data.size
        S = np.sort(self.init_data)
        self.init_threshold = S[int(level * n_init)]
        self.peaks = self.init_data[self.init_data > self.init_threshold] - self.init_threshold
        self.Nt = self.peaks.size
        self.n = n_init
        if verbose:
            print('Initial threshold : %s' % self.init_threshold)
            print('Number of peaks : %s' % self.Nt)
            print('Grimshaw maximum log-likelihood estimation ... ', end='')
        g, s, l = self._grimshaw()
        self.extreme_quantile = self._quantile(g, s)
        if verbose:
            print('[done]')
            print('\t' + chr(0x03B3) + ' = ' + str(g))
            print('\t' + chr(0x03C3) + ' = ' + str(s))
            print('\tL = ' + str(l))
            print('Extreme quantile (probability = %s): %s' % (self.proba, self.extreme_quantile))

    def _rootsFinder(fun, jac, bounds, npoints, method):
        from scipy.optimize import minimize
        if method == 'regular':
            step = (bounds[1] - bounds[0]) / (npoints + 1)
            if step == 0: bounds, step = (0, 1e-4), 1e-5
            X0 = np.arange(bounds[0] + step, bounds[1], step)
        elif method == 'random':
            X0 = np.random.uniform(bounds[0], bounds[1], npoints)
        def objFun(X, f, jac):
            g = 0
            j = np.zeros(X.shape)
            i = 0
            for x in X:
                fx = f(x)
                g = g + fx ** 2
                j[i] = 2 * fx * jac(x)
                i = i + 1
            return g, j
        opt = minimize(lambda X: objFun(X, fun, jac), X0,
                       method='L-BFGS-B',
                       jac=True, bounds=[bounds] * len(X0))
        X = opt.x
        np.round(X, decimals=5)
        return np.unique(X)

    def _log_likelihood(Y, gamma, sigma):
        n = Y.size
        if gamma != 0:
            tau = gamma / sigma
            L = -n * math.log(sigma) - (1 + (1 / gamma)) * (np.log(1 + tau * Y)).sum()
        else:
            L = n * (1 + math.log(Y.mean()))
        return L

    def _grimshaw(self, epsilon=1e-8, n_points=10):
        def u(s): return 1 + np.log(s).mean()
        def v(s): return np.mean(1 / s)
        def w(Y, t):
            s = 1 + t * Y
            us = u(s); vs = v(s)
            return us * vs - 1
        def jac_w(Y, t):
            s = 1 + t * Y
            us = u(s); vs = v(s)
            jac_us = (1 / t) * (1 - vs)
            jac_vs = (1 / t) * (-vs + np.mean(1 / s ** 2))
            return us * jac_vs + vs * jac_us
        Ym = self.peaks.min()
        YM = self.peaks.max()
        Ymean = self.peaks.mean()
        a = -1 / YM
        if abs(a) < 2 * epsilon: epsilon = abs(a) / n_points
        a = a + epsilon
        b = 2 * (Ymean - Ym) / (Ymean * Ym)
        c = 2 * (Ymean - Ym) / (Ym ** 2)
        left_zeros = SPOT._rootsFinder(lambda t: w(self.peaks, t),
                                       lambda t: jac_w(self.peaks, t),
                                       (a + epsilon, -epsilon),
                                       n_points, 'regular')
        right_zeros = SPOT._rootsFinder(lambda t: w(self.peaks, t),
                                        lambda t: jac_w(self.peaks, t),
                                        (b, c),
                                        n_points, 'regular')
        zeros = np.concatenate((left_zeros, right_zeros))
        gamma_best = 0
        sigma_best = Ymean
        ll_best = SPOT._log_likelihood(self.peaks, gamma_best, sigma_best)
        for z in zeros:
            gamma = u(1 + z * self.peaks) - 1
            sigma = gamma / z
            ll = SPOT._log_likelihood(self.peaks, gamma, sigma)
            if ll > ll_best:
                gamma_best = gamma
                sigma_best = sigma
                ll_best = ll
        return gamma_best, sigma_best, ll_best

    def _quantile(self, gamma, sigma):
        r = self.n * self.proba / self.Nt
        if gamma != 0: return self.init_threshold + (sigma / gamma) * (pow(r, -gamma) - 1)
        else: return self.init_threshold - sigma * math.log(r)

    def run(self, with_alarm=True, dynamic=True):
        if self.n > self.init_data.size:
            print('Warning : the algorithm seems to have already been run, you should initialize before running again')
            return {}
        th = []
        alarm = []
        for i in range(self.data.size):
            if not dynamic:
                if self.data[i] > self.init_threshold and with_alarm:
                    self.extreme_quantile = self.init_threshold
                    alarm.append(i)
            else:
                if self.data[i] > self.extreme_quantile:
                    if with_alarm: alarm.append(i)
                    else:
                        self.peaks = np.append(self.peaks, self.data[i] - self.init_threshold)
                        self.Nt += 1
                        self.n += 1
                        g, s, l = self._grimshaw()
                        self.extreme_quantile = self._quantile(g, s)
                elif self.data[i] > self.init_threshold:
                    self.peaks = np.append(self.peaks, self.data[i] - self.init_threshold)
                    self.Nt += 1
                    self.n += 1
                    g, s, l = self._grimshaw()
                    self.extreme_quantile = self._quantile(g, s)
                else:
                    self.n += 1
            th.append(self.extreme_quantile)
        return {'thresholds': th, 'alarms': alarm}

def calc_point2point(predict, actual):
    TP = np.sum(predict * actual)
    TN = np.sum((1 - predict) * (1 - actual))
    FP = np.sum(predict * (1 - actual))
    FN = np.sum((1 - predict) * actual)
    precision = TP / (TP + FP + 0.00001)
    recall = TP / (TP + FN + 0.00001)
    f1 = 2 * precision * recall / (precision + recall + 0.00001)
    try: roc_auc = roc_auc_score(actual, predict)
    except: roc_auc = 0
    return f1, precision, recall, TP, TN, FP, FN, roc_auc

def adjust_predicts(score, label, threshold=None, pred=None, calc_latency=False):
    if len(score) != len(label): raise ValueError("score and label must have the same length")
    score = np.asarray(score)
    label = np.asarray(label)
    latency = 0
    if pred is None: predict = score > threshold
    else: predict = pred
    actual = label > 0.1
    anomaly_state = False
    anomaly_count = 0
    for i in range(len(score)):
        if actual[i] and predict[i] and not anomaly_state:
                anomaly_state = True
                anomaly_count += 1
                for j in range(i, 0, -1):
                    if not actual[j]: break
                    else:
                        if not predict[j]:
                            predict[j] = True
                            latency += 1
        elif not actual[i]: anomaly_state = False
        if anomaly_state: predict[i] = True
    if calc_latency: return predict, latency / (anomaly_count + 1e-4)
    else: return predict

def pot_eval(init_score, score, label, q=1e-5, level=0.02):
    # Default POT parameters
    lms = level
    while True:
        try:
            s = SPOT(q)
            s.fit(init_score, score)
            s.initialize(level=lms, min_extrema=False, verbose=False)
        except: lms = lms * 0.999
        else: break
    ret = s.run(dynamic=False)
    pot_th = np.mean(ret['thresholds'])
    pred, p_latency = adjust_predicts(score, label, pot_th, calc_latency=True)
    p_t = calc_point2point(pred, label)
    return {
        'f1': p_t[0],
        'precision': p_t[1],
        'recall': p_t[2],
        'TP': p_t[3],
        'TN': p_t[4],
        'FP': p_t[5],
        'FN': p_t[6],
        'ROC/AUC': p_t[7],
        'threshold': pot_th,
    }, np.array(pred)

def hit_att(ascore, labels, ps = [100, 150]):
    res = {}
    for p in ps:
        hit_score = []
        for i in range(ascore.shape[0]):
            a, l = ascore[i], labels[i]
            a, l = np.argsort(a).tolist()[::-1], set(np.where(l == 1)[0])
            if l:
                size = round(p * len(l) / 100)
                a_p = set(a[:size])
                intersect = a_p.intersection(l)
                hit = len(intersect) / len(l)
                hit_score.append(hit)
        res[f'Hit@{p}%'] = np.mean(hit_score)
    return res

def ndcg(ascore, labels, ps = [100, 150]):
    res = {}
    for p in ps:
        ndcg_scores = []
        for i in range(ascore.shape[0]):
            a, l = ascore[i], labels[i]
            labs = list(np.where(l == 1)[0])
            if labs:
                k_p = round(p * len(labs) / 100)
                try: hit = ndcg_score(l.reshape(1, -1), a.reshape(1, -1), k = k_p)
                except Exception as e: return {}
                ndcg_scores.append(hit)
        res[f'NDCG@{p}%'] = np.mean(ndcg_scores)
    return res

# ==========================================
# DL MODELS PARTS
# ==========================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model).float() * (-math.log(10000.0) / d_model))
        pe += torch.sin(position * div_term)
        pe += torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x, pos=0):
        x = x + self.pe[pos:pos+x.size(0), :]
        return self.dropout(x)

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=16, dropout=0):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU(True)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout1(src2)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        return src

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=16, dropout=0):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU(True)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        tgt2 = self.self_attn(tgt, tgt, tgt)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.multihead_attn(tgt, memory, memory)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

# ==========================================
# TranAD MODEl
# ==========================================
torch.manual_seed(1)

class TranAD(nn.Module):
    def __init__(self, feats):
        super(TranAD, self).__init__()
        self.name = 'TranAD'
        self.lr = args.lr
        self.batch = args.bs
        self.n_feats = feats
        self.n_window = args.window_size
        self.n = self.n_feats * self.n_window
        self.pos_encoder = PositionalEncoding(2 * feats, 0.1, self.n_window)
        encoder_layers = TransformerEncoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, 1)
        decoder_layers1 = TransformerDecoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
        self.transformer_decoder1 = nn.TransformerDecoder(decoder_layers1, 1)
        decoder_layers2 = TransformerDecoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
        self.transformer_decoder2 = nn.TransformerDecoder(decoder_layers2, 1)
        self.fcn = nn.Sequential(nn.Linear(2 * feats, feats), nn.Sigmoid())

    def encode(self, src, c, tgt):
        src = torch.cat((src, c), dim=2)
        src = src * math.sqrt(self.n_feats)
        src = self.pos_encoder(src)
        memory = self.transformer_encoder(src)
        tgt = tgt.repeat(1, 1, 2)
        return tgt, memory

    def forward(self, src, tgt):
        c = torch.zeros_like(src)
        x1 = self.fcn(self.transformer_decoder1(*self.encode(src, c, tgt)))
        c = (x1 - src) ** 2
        x2 = self.fcn(self.transformer_decoder2(*self.encode(src, c, tgt)))
        return x1, x2

# ==========================================
# TRAINING & LOGIC
# ==========================================
def convert_to_windows(data, model):
    windows = []; w_size = model.n_window
    for i, g in enumerate(data): 
        if i >= w_size: w = data[i-w_size:i]
        else: w = torch.cat([data[0].repeat(w_size-i, 1), data[0:i]])
        windows.append(w)
    return torch.stack(windows)

def preprocess_df(df_train, df_test):
    """
    Standardize/Normalize (0-1 min-max-syle or other) 
    and convert to PyTorch tensors.
    Assumes df_train and df_test are mostly numeric.
    """
    # 1. Convert to numpy
    train = df_train.values
    test = df_test.values

    # 2. Normalize (using min-max of train for both)
    # Similar to original: (x - min) / (max - min + epsilon)
    min_val = np.min(train, axis=0)
    max_val = np.max(train, axis=0)
    
    train = (train - min_val) / (max_val - min_val + 1e-4)
    test = (test - min_val) / (max_val - min_val + 1e-4)

    # 3. Handle NaNs/Infs just in case
    train = np.nan_to_num(train)
    test = np.nan_to_num(test)

    # 4. Make tensors
    train = torch.DoubleTensor(train)
    test = torch.DoubleTensor(test)
    
    return train, test

def backprop(epoch, model, data, dataO, optimizer, scheduler, training = True):
    l = nn.MSELoss(reduction = 'none')
    data_x = torch.DoubleTensor(data); dataset = TensorDataset(data_x, data_x)
    bs = model.batch if training else len(data)
    dataloader = DataLoader(dataset, batch_size = bs)
    n = epoch + 1; w_size = model.n_window
    l1s, l2s = [], []
    feats = dataO.shape[1]
    
    if training:
        for d, _ in dataloader:
            local_bs = d.shape[0]
            window = d.permute(1, 0, 2)
            elem = window[-1, :, :].view(1, local_bs, feats)
            z = model(window, elem)
            l1 = l(z, elem) if not isinstance(z, tuple) else (1 / n) * l(z[0], elem) + (1 - 1/n) * l(z[1], elem)
            if isinstance(z, tuple): z = z[1]
            l1s.append(torch.mean(l1).item())
            loss = torch.mean(l1)
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
        scheduler.step()
        tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)}')
        return np.mean(l1s), optimizer.param_groups[0]['lr']
    else:
        for d, _ in dataloader:
            window = d.permute(1, 0, 2)
            elem = window[-1, :, :].view(1, bs, feats)
            z = model(window, elem)
            if isinstance(z, tuple): z = z[1]
        loss = l(z, elem)[0]
        return loss.detach().numpy(), z.detach().numpy()[0]

def train_model(model, train_loader, test_loader, labels=None):
    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters() , lr=model.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.9)
    epoch = -1
    accuracy_list = []
    
    ## Prepare data
    trainD, testD = next(iter(train_loader)), next(iter(test_loader))
    trainO, testO = trainD, testD
    
    # Windows
    trainD, testD = convert_to_windows(trainD, model), convert_to_windows(testD, model)
    
    # Training
    print(f'{color.HEADER}Training {model.name}{color.ENDC}')
    num_epochs = 5
    start = time()
    for e in tqdm(list(range(epoch+1, epoch+num_epochs+1))):
        lossT, lr = backprop(e, model, trainD, trainO, optimizer, scheduler, training=True)
        accuracy_list.append((lossT, lr))
    print(color.BOLD+'Training time: '+"{:10.4f}".format(time()-start)+' s'+color.ENDC)
    
    # Testing
    torch.zero_grad = True
    model.eval()
    print(f'{color.HEADER}Testing {model.name}{color.ENDC}')
    loss, y_pred = backprop(0, model, testD, testO, optimizer, scheduler, training=False)
    
    return loss, y_pred, accuracy_list

def run_pipeline(df_train, df_test, df_labels=None, labels_col=None):
    """
    Main entry point for generic dataframes.
    df_train: pandas DataFrame for training (numeric columns)
    df_test: pandas DataFrame for testing (numeric columns)
    df_labels: (Optional) pandas DataFrame with labels for test data.
    labels_col: (Optional) name of the label column if df_labels is None and df_test contains labels.
    """
    
    # Separate labels if needed
    if labels_col and labels_col in df_test.columns:
        labels = df_test[labels_col].values
        df_test = df_test.drop(columns=[labels_col])
    elif df_labels is not None:
        labels = df_labels.values
        # Ensure labels are binary (0/1) for anomaly detection
        if len(labels.shape) == 1:
            labels = labels.reshape(-1, 1)
    else:
        # Dummy labels if none provided (all zeros) for coding compatibility
        labels = np.zeros((df_test.shape[0], df_test.shape[1]))
    
    # Preprocess
    train_tensor, test_tensor = preprocess_df(df_train, df_test)
    
    # Create DataLoaders
    # TranAD expects full batch mostly in original code, but we can standardise
    train_loader = DataLoader(train_tensor, batch_size=train_tensor.shape[0])
    test_loader = DataLoader(test_tensor, batch_size=test_tensor.shape[0])
    
    # Initialize Model
    dims = train_tensor.shape[1]
    model = TranAD(dims).double()
    
    # Train & Test
    loss, y_pred, accuracy_list = train_model(model, train_loader, test_loader)
    
    # Evaluation (POT/SPOT)
    print("Running Evaluation (POT)...")
    # Need anomaly scores from Training set for POT initialization
    # We can run backprop(training=False) on Train data
    optimizer = torch.optim.AdamW(model.parameters() , lr=model.lr, weight_decay=1e-5) # Mock for func signature
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.9)
    trainD = next(iter(train_loader)) # Original data
    trainD_windows = convert_to_windows(trainD, model)
    lossT, _ = backprop(0, model, trainD_windows, trainD, optimizer, scheduler, training=False)
    
    results = {}
    preds = []
    
    # For each dimension
    for i in range(loss.shape[1]):
        lt, l, ls = lossT[:, i], loss[:, i], labels[:, i] if labels.shape[1] > 1 else labels[:, 0]
        # Using a default risk q and level for now
        result, pred = pot_eval(lt, l, ls) 
        preds.append(pred)
        # print(f'Dimension {i}:', result)
        
    preds = np.stack(preds, axis=1)
    
    # Final anomaly score (Aggregate)
    lossTfinal, lossFinal = np.mean(lossT, axis=1), np.mean(loss, axis=1)
    labelsFinal = (np.sum(labels, axis=1) >= 1) + 0
    final_result, final_pred = pot_eval(lossTfinal, lossFinal, labelsFinal)
    
    print("\nFinal Results (Aggregate):")
    pprint(final_result)
    
    # Plotting
    plotter(f'{args.model}_generic', torch.DoubleTensor(test_tensor.numpy()), y_pred, loss, labels)
    
    return final_result, preds

if __name__ == '__main__':
    print("TranAD script loaded. Call run_pipeline(df_train, df_test) to execute.")
