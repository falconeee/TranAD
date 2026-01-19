"""
TranAD: Deep Transformer Networks for Anomaly Detection in Multivariate Time Series Data
Refatorado para execução em Jupyter Notebook
"""

import pickle
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.nn import TransformerEncoder, TransformerDecoder
from tqdm import tqdm
from time import time
from pprint import pprint
import math
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import roc_auc_score, ndcg_score
from scipy.optimize import minimize

# ============================================================================
# CONFIGURAÇÕES E CONSTANTES
# ============================================================================

class Config:
    """Configurações do modelo TranAD"""
    # Pastas de dados
    output_folder = 'processed'
    data_folder = 'data'
    
    # Parâmetros de threshold por dataset
    lm_d = {
        'SMD': [(0.99995, 1.04), (0.99995, 1.06)],
        'synthetic': [(0.999, 1), (0.999, 1)],
        'SWaT': [(0.993, 1), (0.993, 1)],
        'UCR': [(0.993, 1), (0.99935, 1)],
        'NAB': [(0.991, 1), (0.99, 1)],
        'SMAP': [(0.98, 1), (0.98, 1)],
        'MSL': [(0.97, 1), (0.999, 1.04)],
        'WADI': [(0.99, 1), (0.999, 1)],
        'MSDS': [(0.91, 1), (0.9, 1.04)],
        'MBA': [(0.87, 1), (0.93, 1.04)],
    }
    
    # Learning rates por dataset
    lr_d = {
        'SMD': 0.0001,
        'synthetic': 0.0001,
        'SWaT': 0.008,
        'SMAP': 0.001,
        'MSL': 0.002,
        'WADI': 0.0001,
        'MSDS': 0.001,
        'UCR': 0.006,
        'NAB': 0.009,
        'MBA': 0.001,
    }
    
    def __init__(self, dataset='SMAP', model='TranAD'):
        self.dataset = dataset
        self.model = model
        self.lm = self.lm_d.get(dataset, [(0.98, 1), (0.98, 1)])[1 if 'TranAD' in model else 0]
        self.lr = self.lr_d.get(dataset, 0.001)
        self.less = False
        self.test = False
        self.retrain = True

class Color:
    """Cores para output no terminal"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    RED = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# ============================================================================
# COMPONENTES DO MODELO
# ============================================================================

class PositionalEncoding(nn.Module):
    """Encoding posicional para transformers"""
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
    """Camada de encoder do transformer"""
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
    """Camada de decoder do transformer"""
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

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, 
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        tgt2 = self.self_attn(tgt, tgt, tgt)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.multihead_attn(tgt, memory, memory)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

# ============================================================================
# MODELO TRANAD
# ============================================================================

class TranAD(nn.Module):
    """Modelo TranAD completo com Self-Conditioning e Adversarial"""
    def __init__(self, feats, lr=0.001):
        super(TranAD, self).__init__()
        self.name = 'TranAD'
        self.lr = lr
        self.batch = 128
        self.n_feats = feats
        self.n_window = 10
        self.n = self.n_feats * self.n_window
        self.pos_encoder = PositionalEncoding(2 * feats, 0.1, self.n_window)
        encoder_layers = TransformerEncoderLayer(d_model=2 * feats, nhead=feats, 
                                                  dim_feedforward=16, dropout=0.1)
        self.transformer_encoder = TransformerEncoder(encoder_layers, 1)
        decoder_layers1 = TransformerDecoderLayer(d_model=2 * feats, nhead=feats, 
                                                  dim_feedforward=16, dropout=0.1)
        self.transformer_decoder1 = TransformerDecoder(decoder_layers1, 1)
        decoder_layers2 = TransformerDecoderLayer(d_model=2 * feats, nhead=feats, 
                                                  dim_feedforward=16, dropout=0.1)
        self.transformer_decoder2 = TransformerDecoder(decoder_layers2, 1)
        self.fcn = nn.Sequential(nn.Linear(2 * feats, feats), nn.Sigmoid())

    def encode(self, src, c, tgt):
        """Encoding com condicionamento"""
        src = torch.cat((src, c), dim=2)
        src = src * math.sqrt(self.n_feats)
        src = self.pos_encoder(src)
        memory = self.transformer_encoder(src)
        tgt = tgt.repeat(1, 1, 2)
        return tgt, memory

    def forward(self, src, tgt):
        """Forward pass em duas fases"""
        # Fase 1 - Sem scores de anomalia
        c = torch.zeros_like(src)
        x1 = self.fcn(self.transformer_decoder1(*self.encode(src, c, tgt)))
        # Fase 2 - Com scores de anomalia
        c = (x1 - src) ** 2
        x2 = self.fcn(self.transformer_decoder2(*self.encode(src, c, tgt)))
        return x1, x2

# ============================================================================
# UTILITÁRIOS DE DADOS
# ============================================================================

def convert_to_windows(data, model):
    """Converte dados em janelas para o modelo"""
    windows = []
    w_size = model.n_window
    for i, g in enumerate(data):
        if i >= w_size:
            w = data[i-w_size:i]
        else:
            w = torch.cat([data[0].repeat(w_size-i, 1), data[0:i]])
        windows.append(w)
    return torch.stack(windows)

def load_dataset(dataset, config):
    """Carrega dataset processado"""
    folder = os.path.join(config.output_folder, dataset)
    if not os.path.exists(folder):
        raise Exception('Processed Data not found.')
    
    loader = []
    for file in ['train', 'test', 'labels']:
        if dataset == 'SMD': file = 'machine-1-1_' + file
        if dataset == 'SMAP': file = 'P-1_' + file
        if dataset == 'MSL': file = 'C-1_' + file
        if dataset == 'UCR': file = '136_' + file
        if dataset == 'NAB': file = 'ec2_request_latency_system_failure_' + file
        loader.append(np.load(os.path.join(folder, f'{file}.npy')))
    
    if config.less:
        loader[0] = cut_array(0.2, loader[0])
    
    train_loader = DataLoader(loader[0], batch_size=loader[0].shape[0])
    test_loader = DataLoader(loader[1], batch_size=loader[1].shape[0])
    labels = loader[2]
    return train_loader, test_loader, labels

def cut_array(percentage, arr):
    """Corta array para porcentagem especificada"""
    print(f'{Color.BOLD}Slicing dataset to {int(percentage*100)}%{Color.ENDC}')
    mid = round(arr.shape[0] / 2)
    window = round(arr.shape[0] * percentage * 0.5)
    return arr[mid - window : mid + window, :]

# ============================================================================
# TREINAMENTO E TESTE
# ============================================================================

def backprop(epoch, model, data, dataO, optimizer, scheduler, training=True):
    """Backpropagation para TranAD"""
    l = nn.MSELoss(reduction='none')
    feats = dataO.shape[1]
    
    if 'TranAD' in model.name:
        data_x = torch.DoubleTensor(data)
        dataset = TensorDataset(data_x, data_x)
        bs = model.batch if training else len(data)
        dataloader = DataLoader(dataset, batch_size=bs)
        n = epoch + 1
        w_size = model.n_window
        l1s, l2s = [], []
        
        if training:
            for d, _ in dataloader:
                local_bs = d.shape[0]
                window = d.permute(1, 0, 2)
                elem = window[-1, :, :].view(1, local_bs, feats)
                z = model(window, elem)
                l1 = l(z, elem) if not isinstance(z, tuple) else (1 / n) * l(z[0], elem) + (1 - 1/n) * l(z[1], elem)
                if isinstance(z, tuple):
                    z = z[1]
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
                if isinstance(z, tuple):
                    z = z[1]
            loss = l(z, elem)[0]
            return loss.detach().numpy(), z.detach().numpy()[0]
    else:
        raise NotImplementedError(f"Model {model.name} not implemented in this refactored version")

def save_model(model, optimizer, scheduler, epoch, accuracy_list, config):
    """Salva modelo treinado"""
    folder = f'checkpoints/{config.model}_{config.dataset}/'
    os.makedirs(folder, exist_ok=True)
    file_path = f'{folder}/model.ckpt'
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'accuracy_list': accuracy_list
    }, file_path)

def load_model(modelname, dims, config):
    """Carrega modelo (cria novo se não existir)"""
    model = TranAD(dims, lr=config.lr).double()
    optimizer = torch.optim.AdamW(model.parameters(), lr=model.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.9)
    fname = f'checkpoints/{config.model}_{config.dataset}/model.ckpt'
    
    if os.path.exists(fname) and (not config.retrain or config.test):
        print(f"{Color.GREEN}Loading pre-trained model: {model.name}{Color.ENDC}")
        checkpoint = torch.load(fname)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch = checkpoint['epoch']
        accuracy_list = checkpoint['accuracy_list']
    else:
        print(f"{Color.GREEN}Creating new model: {model.name}{Color.ENDC}")
        epoch = -1
        accuracy_list = []
    return model, optimizer, scheduler, epoch, accuracy_list

# ============================================================================
# AVALIAÇÃO (POT - Peaks Over Threshold)
# ============================================================================

class SPOT:
    """Streaming Peaks-Over-Threshold para detecção de anomalias"""
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
        if isinstance(data, list):
            self.data = np.array(data)
        elif isinstance(data, np.ndarray):
            self.data = data
        else:
            self.data = np.array(data)
        
        if isinstance(init_data, list):
            self.init_data = np.array(init_data)
        elif isinstance(init_data, np.ndarray):
            self.init_data = init_data
        else:
            self.init_data = np.array(init_data)

    def initialize(self, level=0.98, min_extrema=False, verbose=False):
        from math import floor
        if min_extrema:
            self.init_data = -self.init_data
            self.data = -self.data
            level = 1 - level
        level = level - floor(level)
        n_init = self.init_data.size
        S = np.sort(self.init_data)
        self.init_threshold = S[int(level * n_init)]
        self.peaks = self.init_data[self.init_data > self.init_threshold] - self.init_threshold
        self.Nt = self.peaks.size
        self.n = n_init
        g, s, l = self._grimshaw()
        self.extreme_quantile = self._quantile(g, s)
        return

    def _grimshaw(self, epsilon=1e-8, n_points=10):
        def u(s):
            return 1 + np.log(s).mean()
        def v(s):
            return np.mean(1 / s)
        def w(Y, t):
            s = 1 + t * Y
            us = u(s)
            vs = v(s)
            return us * vs - 1
        def jac_w(Y, t):
            s = 1 + t * Y
            us = u(s)
            vs = v(s)
            jac_us = (1 / t) * (1 - vs)
            jac_vs = (1 / t) * (-vs + np.mean(1 / s ** 2))
            return us * jac_vs + vs * jac_us

        Ym = self.peaks.min()
        YM = self.peaks.max()
        Ymean = self.peaks.mean()
        a = -1 / YM
        if abs(a) < 2 * epsilon:
            epsilon = abs(a) / n_points
        a = a + epsilon
        b = 2 * (Ymean - Ym) / (Ymean * Ym)
        c = 2 * (Ymean - Ym) / (Ym ** 2)

        left_zeros = self._rootsFinder(lambda t: w(self.peaks, t),
                                       lambda t: jac_w(self.peaks, t),
                                       (a + epsilon, -epsilon),
                                       n_points, 'regular')
        right_zeros = self._rootsFinder(lambda t: w(self.peaks, t),
                                        lambda t: jac_w(self.peaks, t),
                                        (b, c),
                                        n_points, 'regular')
        zeros = np.concatenate((left_zeros, right_zeros))
        gamma_best = 0
        sigma_best = Ymean
        ll_best = self._log_likelihood(self.peaks, gamma_best, sigma_best)
        for z in zeros:
            gamma = u(1 + z * self.peaks) - 1
            sigma = gamma / z
            ll = self._log_likelihood(self.peaks, gamma, sigma)
            if ll > ll_best:
                gamma_best = gamma
                sigma_best = sigma
                ll_best = ll
        return gamma_best, sigma_best, ll_best

    def _rootsFinder(self, fun, jac, bounds, npoints, method):
        if method == 'regular':
            step = (bounds[1] - bounds[0]) / (npoints + 1)
            if step == 0:
                bounds, step = (0, 1e-4), 1e-5
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

    def _log_likelihood(self, Y, gamma, sigma):
        from math import log
        n = Y.size
        if gamma != 0:
            tau = gamma / sigma
            L = -n * log(sigma) - (1 + (1 / gamma)) * (np.log(1 + tau * Y)).sum()
        else:
            L = n * (1 + log(Y.mean()))
        return L

    def _quantile(self, gamma, sigma):
        from math import log, pow
        r = self.n * self.proba / self.Nt
        if gamma != 0:
            return self.init_threshold + (sigma / gamma) * (pow(r, -gamma) - 1)
        else:
            return self.init_threshold - sigma * log(r)

    def run(self, dynamic=False):
        if self.n > self.init_data.size:
            return {}
        th = []
        alarm = []
        for i in range(self.data.size):
            if not dynamic:
                if self.data[i] > self.init_threshold:
                    self.extreme_quantile = self.init_threshold
                    alarm.append(i)
            else:
                if self.data[i] > self.extreme_quantile:
                    alarm.append(i)
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
    """Calcula métricas ponto a ponto"""
    TP = np.sum(predict * actual)
    TN = np.sum((1 - predict) * (1 - actual))
    FP = np.sum(predict * (1 - actual))
    FN = np.sum((1 - predict) * actual)
    precision = TP / (TP + FP + 0.00001)
    recall = TP / (TP + FN + 0.00001)
    f1 = 2 * precision * recall / (precision + recall + 0.00001)
    try:
        roc_auc = roc_auc_score(actual, predict)
    except:
        roc_auc = 0
    return f1, precision, recall, TP, TN, FP, FN, roc_auc

def adjust_predicts(score, label, threshold=None, pred=None, calc_latency=False):
    """Ajusta predições usando threshold"""
    if len(score) != len(label):
        raise ValueError("score and label must have the same length")
    score = np.asarray(score)
    label = np.asarray(label)
    latency = 0
    if pred is None:
        predict = score > threshold
    else:
        predict = pred
    actual = label > 0.1
    anomaly_state = False
    anomaly_count = 0
    for i in range(len(score)):
        if actual[i] and predict[i] and not anomaly_state:
            anomaly_state = True
            anomaly_count += 1
            for j in range(i, 0, -1):
                if not actual[j]:
                    break
                else:
                    if not predict[j]:
                        predict[j] = True
                        latency += 1
        elif not actual[i]:
            anomaly_state = False
        if anomaly_state:
            predict[i] = True
    if calc_latency:
        return predict, latency / (anomaly_count + 1e-4)
    else:
        return predict

def pot_eval(init_score, score, label, q=1e-5, level=0.02, lm=(0.98, 1)):
    """Avaliação usando POT"""
    lms = lm[0]
    while True:
        try:
            s = SPOT(q)
            s.fit(init_score, score)
            s.initialize(level=lms, min_extrema=False, verbose=False)
        except:
            lms = lms * 0.999
        else:
            break
    ret = s.run(dynamic=False)
    pot_th = np.mean(ret['thresholds']) * lm[1]
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

# ============================================================================
# DIAGNÓSTICO
# ============================================================================

def hit_att(ascore, labels, ps=[100, 150]):
    """Hit@K para diagnóstico"""
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

def ndcg(ascore, labels, ps=[100, 150]):
    """NDCG@K para diagnóstico"""
    res = {}
    for p in ps:
        ndcg_scores = []
        for i in range(ascore.shape[0]):
            a, l = ascore[i], labels[i]
            labs = list(np.where(l == 1)[0])
            if labs:
                k_p = round(p * len(labs) / 100)
                try:
                    hit = ndcg_score(l.reshape(1, -1), a.reshape(1, -1), k=k_p)
                except Exception as e:
                    return {}
                ndcg_scores.append(hit)
        res[f'NDCG@{p}%'] = np.mean(ndcg_scores)
    return res

# ============================================================================
# PLOTAGEM
# ============================================================================

def smooth(y, box_pts=1):
    """Suaviza sinal"""
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def plot_accuracies(accuracy_list, folder):
    """Plota acurácias de treinamento"""
    os.makedirs(f'plots/{folder}/', exist_ok=True)
    trainAcc = [i[0] for i in accuracy_list]
    lrs = [i[1] for i in accuracy_list]
    plt.xlabel('Epochs')
    plt.ylabel('Average Training Loss')
    plt.plot(range(len(trainAcc)), trainAcc, label='Average Training Loss', 
             linewidth=1, linestyle='-', marker='.')
    plt.twinx()
    plt.plot(range(len(lrs)), lrs, label='Learning Rate', color='r', 
             linewidth=1, linestyle='--', marker='.')
    plt.savefig(f'plots/{folder}/training-graph.pdf')
    plt.clf()

def plotter(name, y_true, y_pred, ascore, labels):
    """Plota resultados de detecção"""
    if 'TranAD' in name:
        y_true = torch.roll(torch.tensor(y_true), 1, 0).numpy()
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
        if dim == 0:
            ax1.legend(ncol=2, bbox_to_anchor=(0.6, 1.02))
        ax2.plot(smooth(a_s), linewidth=0.2, color='g')
        ax2.set_xlabel('Timestamp')
        ax2.set_ylabel('Anomaly Score')
        pdf.savefig(fig)
        plt.close()
    pdf.close()

# ============================================================================
# FUNÇÃO PRINCIPAL
# ============================================================================

def run_tranad(dataset='SMAP', model='TranAD', num_epochs=5, less=False, 
                test=False, retrain=True, plot=True):
    """
    Executa o pipeline completo do TranAD
    
    Parâmetros:
    -----------
    dataset : str
        Nome do dataset ('SMAP', 'MSL', 'SMD', etc.)
    model : str
        Nome do modelo (default: 'TranAD')
    num_epochs : int
        Número de épocas de treinamento
    less : bool
        Usar apenas 20% dos dados de treinamento
    test : bool
        Apenas testar (sem treinar)
    retrain : bool
        Retreinar mesmo se modelo existir
    plot : bool
        Gerar plots
    
    Retorna:
    --------
    dict : Resultados da avaliação
    """
    # Configuração
    config = Config(dataset=dataset, model=model)
    config.less = less
    config.test = test
    config.retrain = retrain
    
    # Carregar dados
    print(f'{Color.HEADER}Loading dataset: {dataset}{Color.ENDC}')
    train_loader, test_loader, labels = load_dataset(dataset, config)
    
    # Carregar modelo
    model, optimizer, scheduler, epoch, accuracy_list = load_model(model, labels.shape[1], config)
    
    # Preparar dados
    trainD, testD = next(iter(train_loader)), next(iter(test_loader))
    trainO, testO = trainD, testD
    trainD = convert_to_windows(trainD, model)
    testD = convert_to_windows(testD, model)
    
    # Treinamento
    if not test:
        print(f'{Color.HEADER}Training {model.name} on {dataset}{Color.ENDC}')
        e = epoch + 1
        start = time()
        for e in tqdm(list(range(epoch+1, epoch+num_epochs+1))):
            lossT, lr = backprop(e, model, trainD, trainO, optimizer, scheduler)
            accuracy_list.append((lossT, lr))
        print(Color.BOLD + 'Training time: ' + "{:10.4f}".format(time()-start) + ' s' + Color.ENDC)
        save_model(model, optimizer, scheduler, e, accuracy_list, config)
        if plot:
            plot_accuracies(accuracy_list, f'{model.name}_{dataset}')
    
    # Teste
    model.eval()
    print(f'{Color.HEADER}Testing {model.name} on {dataset}{Color.ENDC}')
    with torch.no_grad():
        loss, y_pred = backprop(0, model, testD, testO, optimizer, scheduler, training=False)
    
    # Plotar curvas
    if not test and plot:
        if 'TranAD' in model.name:
            testO = torch.roll(torch.tensor(testO), 1, 0).numpy()
        plotter(f'{model.name}_{dataset}', testO, y_pred, loss, labels)
    
    # Avaliação
    df = pd.DataFrame()
    preds = []
    lossT, _ = backprop(0, model, trainD, trainO, optimizer, scheduler, training=False)
    for i in range(loss.shape[1]):
        lt, l, ls = lossT[:, i], loss[:, i], labels[:, i]
        result, pred = pot_eval(lt, l, ls, lm=config.lm)
        preds.append(pred)
        df = pd.concat([df, pd.DataFrame([result])], ignore_index=True)
    
    lossTfinal, lossFinal = np.mean(lossT, axis=1), np.mean(loss, axis=1)
    labelsFinal = (np.sum(labels, axis=1) >= 1) + 0
    result, _ = pot_eval(lossTfinal, lossFinal, labelsFinal, lm=config.lm)
    result.update(hit_att(loss, labels))
    result.update(ndcg(loss, labels))
    
    print(df)
    pprint(result)
    
    return result

# ============================================================================
# EXEMPLO DE USO PARA NOTEBOOK
# ============================================================================

if __name__ == '__main__':
    # Exemplo de uso
    result = run_tranad(dataset='SMAP', num_epochs=5, retrain=True)
    print("\nResultados finais:")
    print(result)
