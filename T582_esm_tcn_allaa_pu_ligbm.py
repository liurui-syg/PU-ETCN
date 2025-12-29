
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestClassifier
from torch.nn.functional import pad
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from tabulate import tabulate
from sklearn.utils import resample
from torchinfo import summary
from sklearn.model_selection import KFold, StratifiedKFold, ShuffleSplit
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, matthews_corrcoef, recall_score, \
    precision_score, f1_score, confusion_matrix, roc_curve
import logging
import time
import warnings
import random
from Bio import SeqIO
import lightgbm as lgb
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split


warnings.filterwarnings("ignore")
logging.basicConfig(filename='Log/T582_esm(150M)_tcn_allaa_floss_pu_ligbm.log', level=logging.INFO, format='%(asctime)s - %(message)s')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
logging.info(f"Using device: {device}")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

args = {'batch_size': 16, # 32
        'epochs': 200,
        'lr': 0.0001,  # 0.00001
        'dropout': 0.2,
        'embedding_dim': 640,
        'num_channels': 256, # 128, # [256, 128, 64], [128, 64],
        'max_length': 1024,
        'patience': 10}
print(f"******** {args} ********")
logging.info(f"******** {args} ********")


esm_model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t30_150M_UR50D")
esm_model = esm_model.to(device)
batch_converter = alphabet.get_batch_converter()


# Step 1: Read sequences and labels
def load_sequences_from_fasta(fasta_file):
    sequences = []
    labels = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        seq = str(record.seq)
        sequences.append(seq.upper())
        labels.append([1 if c.isupper() else 0 for c in seq])
    return sequences, labels


# Step 2: Prepare the dataset
class ProteinDataset(Dataset):
    def __init__(self, sequences, labels, batch_converter, model, max_length=1024, fold_type="train"):
        self.sequences = sequences
        self.labels = labels
        self.batch_converter = batch_converter
        self.model = model
        self.max_length = max_length
        self.ignore_label = -1
        self.fold_type = fold_type  # "train"或"val"，控制是否应用PU

        self.encoded_sequences = self._encode_sequences()
        self.processed_features, self.processed_labels = self._process_with_pu()

    def _encode_sequences(self):
        encoded_sequences = []
        for idx, sequence in enumerate(self.sequences):
            if len(sequence) > self.max_length:
                sequence = sequence[:self.max_length]
            batch_labels, batch_strs, batch_tokens = self.batch_converter([(f"protein_{idx}", sequence.upper())])
            batch_tokens = batch_tokens.to(device)
            with torch.no_grad():
                results = self.model(batch_tokens, repr_layers=[30], return_contacts=False)
                encoded_sequence = results["representations"][30].squeeze(0)
                # print(f"Original: {len(sequence)}, Encoded: {encoded_sequence.shape[0]}")
                if encoded_sequence.shape[0] > self.max_length:
                    encoded_sequence = encoded_sequence[:self.max_length]
            encoded_sequences.append(encoded_sequence)
        return encoded_sequences

    def _process_with_pu(self):
        """处理标签并应用PU-Learning筛选可信负例"""
        all_features = []
        all_labels = []

        for seq_idx in range(len(self.sequences)):
            encoding = self.encoded_sequences[seq_idx]
            raw_label = self.labels[seq_idx]
            valid_len = min(len(raw_label), encoding.shape[0], self.max_length)
            device = encoding.device
            feat_dtype = encoding.dtype

            unified_label = torch.full((self.max_length,), self.ignore_label, dtype=torch.float, device=device)
            unified_feature = torch.zeros((self.max_length, encoding.shape[1]), dtype=feat_dtype, device=device)
            unified_feature[:valid_len] = encoding[:valid_len]

            positive_indices = [i for i in range(valid_len) if raw_label[i] == 1]
            unlabeled_indices = [i for i in range(valid_len) if raw_label[i] == 0]

            # PU-Learning-lightgbm
            if self.fold_type == "train" and len(positive_indices) > 0 and len(unlabeled_indices) > 0:
                positive_features = encoding[positive_indices].cpu().numpy()
                unlabeled_features = encoding[unlabeled_indices].cpu().numpy()

                # 构造训练集：正例 + 所有未标注（假设为负类）
                X_train = np.vstack([positive_features, unlabeled_features])
                y_train = np.array([1] * len(positive_features) + [0] * len(unlabeled_features))

                # # 1
                # clf = lgb.LGBMClassifier(
                #     n_estimators=200,  # 多给一些迭代次数，配合 early_stopping 自动截断
                #     learning_rate=0.05,  # 稍微小一点，提升稳定性
                #     num_leaves=7,  # 限制叶子数，避免过拟合
                #     max_depth=3,  # 限制树深度，避免过拟合
                #     min_data_in_leaf=2,  # 小样本下，允许叶子里样本数更少
                #     min_child_samples=2,  # 等价于 min_data_in_leaf，确保树能分裂
                #     reg_alpha=0.1,  # L1 正则，防止过拟合
                #     reg_lambda=0.1,  # L2 正则，防止过拟合
                #     subsample=0.8,  # 行采样，增强泛化
                #     colsample_bytree=0.8,  # 列采样，避免高维过拟合
                #     class_weight='balanced',  # 自动调整类别权重
                #     objective='binary',  # 二分类
                #     random_state=42,
                #     n_jobs=-1
                # )
                # 2
                clf = lgb.LGBMClassifier(
                    n_estimators=1000,
                    learning_rate=0.05,
                    num_leaves=127,
                    max_depth=-1,
                    min_data_in_leaf=20,
                    min_child_samples=20,
                    reg_alpha=0.1,
                    reg_lambda=0.1,
                    subsample=0.8,
                    subsample_freq=1,
                    colsample_bytree=0.8,
                    feature_fraction=0.8,
                    class_weight='balanced',
                    objective='binary',
                    random_state=42,
                    n_jobs=-1
                )
                # 3
                # clf = lgb.LGBMClassifier(
                #     n_estimators=2000,
                #     learning_rate=0.01,
                #     num_leaves=255,
                #     max_depth=-1,
                #     min_data_in_leaf=2,
                #     min_child_samples=10,
                #     reg_alpha=0.1,
                #     reg_lambda=0.1,
                #     subsample=0.8,
                #     subsample_freq=1,
                #     colsample_bytree=0.8,
                #     feature_fraction=0.8,
                #     class_weight='balanced',
                #     objective='binary',
                #     random_state=42,
                #     n_jobs=-1
                # )

                # 如果样本足够，拆分验证集做 early stopping
                if len(y_train) >= 10 and len(np.unique(y_train)) > 1:
                    try:
                        # stratify 保证正负样本分布一致
                        X_tr, X_val, y_tr, y_val = train_test_split(
                            X_train, y_train,
                            test_size=0.2,
                            stratify=y_train,
                            random_state=42
                        )
                    except ValueError:
                        # 如果 stratify 失败（比如全是正例或全是负例），退化为普通 split
                        X_tr, X_val, y_tr, y_val = train_test_split(
                            X_train, y_train,
                            test_size=0.2,
                            random_state=42
                        )

                    # 检查验证集是否包含两个类别
                    if len(np.unique(y_val)) < 2 or len(np.unique(y_tr)) < 2:
                        clf.fit(X_train, y_train)
                    else:
                        clf.fit(
                            X_tr, y_tr,
                            eval_set=[(X_val, y_val)],
                            eval_metric='binary_logloss',
                            callbacks=[lgb.early_stopping(stopping_rounds=10)]
                        )

                else:
                    clf.fit(X_train, y_train)

                # 预测所有未标注样本的正类概率
                probs = clf.predict_proba(unlabeled_features)[:, 1]

                # 选出 top_k 最低概率作为可信负例
                top_k = int(2 * len(positive_indices))
                top_k = min(top_k, len(unlabeled_indices))
                ranked_idxs = np.argsort(probs)[:top_k]
                trusted_negative_indices = [unlabeled_indices[i] for i in ranked_idxs]

                # 设置标签
                unified_label[positive_indices] = 1.0
                unified_label[trusted_negative_indices] = 0.0
            else:
                # 验证集/测试集直接使用原始标签
                unified_label[:valid_len] = torch.tensor(raw_label[:valid_len], dtype=torch.float)

            # 5. 确保特征长度严格等于max_length（补零已在初始化时完成）
            assert unified_feature.shape == (self.max_length, encoding.shape[1]), "The feature length is not uniform!!"
            assert unified_label.shape == (self.max_length,), "The label length is not uniform!!"

            all_features.append(unified_feature)
            all_labels.append(unified_label)

        return all_features, all_labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        feature = self.encoded_sequences[idx]
        label = self.processed_labels[idx]

        if feature.shape[0] < self.max_length:
            padding = torch.zeros((self.max_length - feature.shape[0], feature.shape[1]), device=feature.device)
            feature = torch.cat((feature, padding), dim=0)

        feature = feature.transpose(0, 1)
        return {'encoded_sequence': feature, 'labels': label}

print("******************** "
      "clf = lgb.LGBMClassifier(n_estimators=1000,"
      "learning_rate=0.05,"
      "num_leaves=127,"
      "max_depth=-1,"
      "min_data_in_leaf=2,"
      "min_child_samples=2,"
      "reg_alpha=0.1,"
      "reg_lambda=0.1,"
      "subsample=0.8,"
      "colsample_bytree=0.8,"
      "class_weight='balanced',"
      "objective='binary',"
      "random_state=42,"
      "n_jobs=-1) ********************")
logging.info("******************** "
      "clf = lgb.LGBMClassifier(n_estimators=1000,"
      "learning_rate=0.05,"
      "num_leaves=127,"
      "max_depth=-1,"
      "min_data_in_leaf=2,"
      "min_child_samples=2,"
      "reg_alpha=0.1,"
      "reg_lambda=0.1,"
      "subsample=0.8,"
      "colsample_bytree=0.8,"
      "class_weight='balanced',"
      "objective='binary',"
      "random_state=42,"
      "n_jobs=-1) ********************")
print("************************ PU-Learning--lightgbm top_k=int(2 * len(positive_indices)) ************************")
logging.info("************************ PU-Learning--lightgbm top_k=int(2 * len(positive_indices)) ************************")


# Step 3: TCN Model
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=args['dropout']):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        # (Batch, input_channel, padding + seq_len + padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.relu1, self.dropout1,
                                 self.conv2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        # x: size of (Batch, input_channel, seq_len)
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        if out.shape[2] != res.shape[2]:
            out = F.interpolate(out, size=res.shape[2], mode='linear', align_corners=False)
        return self.relu(out + res)

class TCNModel(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=args['dropout']):
        super(TCNModel, self).__init__()
        layers = []
        in_channels = num_inputs

        # 增加TCN层，通过 dilation 扩大感受野（覆盖全长序列）
        for i in range(4):  # 4层TCN，感受野 = 1 + 2*(kernel_size-1)*(2^4 - 1)
            dilation_size = 2 ** i
            out_channels = num_channels * (2 ** i)  # 通道数随层数增加
            layers += [TemporalBlock(
                in_channels,
                out_channels,
                kernel_size,
                stride=1,
                dilation=dilation_size,
                padding=(kernel_size - 1) * dilation_size,
                dropout=dropout
            )]
            in_channels = out_channels

        self.network = nn.Sequential(*layers)

        # 输出层：为每个残基位置预测一个标签（1通道）
        self.classifier = nn.Conv1d(in_channels, 1, kernel_size=1)

    def forward(self, x):
        # x: [batch, 640, max_length]
        tcn_out = self.network(x)  # [batch, 最终通道数, max_length]
        logits = self.classifier(tcn_out).squeeze(1)  # [batch, max_length]（每个残基的原始预测）

        output = torch.sigmoid(logits)
        attn_weights = None

        return output, attn_weights

print("**************************** TCN layers = 4, dilation_size = 2 ** i ****************************")
logging.info("**************************** TCN layers = 4, dilation_size = 2 ** i ****************************")


# Step 4: Train the model with early stopping
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs, patience, model_path):
    best_val_loss = float('inf')
    best_val = 0.0
    patience_counter = 0

    headers = ["Epoch", "Training Loss", "ACC", "AUC-ROC", "AUC-PR", "MCC", "Recall", "Specificity", "Precision",
               "F1-score", "BAC", "AUC10%",
               "Validation Loss", "ACC", "AUC-ROC", "AUC-PR", "MCC", "Recall", "Specificity", "Precision", "F1-score",
               "BAC", "AUC10%", "Learning Rate"]
    print(tabulate([], headers=headers, tablefmt="plain"))
    logging.info(tabulate([], headers=headers, tablefmt="plain"))

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        all_outputs, all_labels = [], []

        # 获取当前学习率
        current_lr = optimizer.param_groups[0]['lr']

        for batch in train_loader:
            optimizer.zero_grad()

            batch_embeddings = batch['encoded_sequence'].to(device)
            batch_labels = batch['labels'].to(device)

            model_output, _ = model(batch_embeddings)

            mask = (batch_labels != -1)
            model_output = model_output[mask]
            batch_labels = batch_labels[mask].float()

            loss = criterion(model_output, batch_labels.float())
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            all_outputs.append(model_output.detach().cpu().numpy())
            all_labels.append(batch_labels.cpu().numpy())

        # 每个epoch结束后更新学习率
        scheduler.step()

        # Calculate average training loss
        avg_train_loss = epoch_loss / len(train_loader)
        all_outputs, all_labels = np.concatenate(all_outputs), np.concatenate(all_labels)
        train_metrics = compute_metrics(all_labels, all_outputs)

        # Validate the model
        val_loss, val_metrics = evaluate_model(model, val_loader, criterion)
        metrics_to_print = [
            epoch + 1,
            round(avg_train_loss, 4),
            round(train_metrics["accuracy"], 4),
            round(train_metrics["auc_roc"], 4),
            round(train_metrics["auc_pr"], 4),
            round(train_metrics["mcc"], 4),
            round(train_metrics["recall"], 4),
            round(train_metrics["specificity"], 4),
            round(train_metrics["precision"], 4),
            round(train_metrics["f1"], 4),
            round(train_metrics["bac"], 4),
            round(train_metrics["auc10%"], 4),
            round(val_loss, 4),
            round(val_metrics["accuracy"], 4),
            round(val_metrics["auc_roc"], 4),
            round(val_metrics["auc_pr"], 4),
            round(val_metrics["mcc"], 4),
            round(val_metrics["recall"], 4),
            round(val_metrics["specificity"], 4),
            round(val_metrics["precision"], 4),
            round(val_metrics["f1"], 4),
            round(val_metrics["bac"], 4),
            round(val_metrics["auc10%"], 4),
            round(current_lr, 6)
        ]

        print(tabulate([metrics_to_print], tablefmt="plain"))
        logging.info(tabulate([metrics_to_print], tablefmt="plain"))

        # Early stopping and model saving logic
        if val_loss < best_val_loss or val_metrics['auc_pr'] > best_val:
            best_val_loss = min(val_loss, best_val_loss)
            best_val = max(val_metrics['auc_pr'], best_val)
            patience_counter = 0
            torch.save(model.state_dict(), model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("***** Early stopping triggered. *****")
                logging.info("***** Early stopping triggered. *****")
                break

print("**************************** Early stopping: loss + auc_pr ****************************")
logging.info("**************************** Early stopping: loss + auc_pr ****************************")


# Step 5: Evaluate the model
def evaluate_model(model, data_loader, criterion=None):
    model.eval()
    all_outputs, all_labels, val_loss = [], [], 0.0

    with torch.no_grad():
        for batch in data_loader:
            encoded_sequences = batch['encoded_sequence'].to(device)
            labels = batch['labels'].to(device)

            model_output, _ = model(encoded_sequences)
            if criterion:

                mask = (labels != -1)
                model_output = model_output[mask]
                labels = labels[mask].float()

                loss = criterion(model_output, labels)
                val_loss += loss.item()

            all_outputs.append(model_output.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    all_outputs, all_labels = np.concatenate(all_outputs, axis=0), np.concatenate(all_labels, axis=0)
    metrics = compute_metrics(all_labels, all_outputs)
    return (val_loss / len(data_loader) if criterion else None), metrics


# Compute metrics
def compute_metrics(labels, outputs, ignore_label=-1):
    valid_mask = (labels != ignore_label)
    labels = labels[valid_mask]
    outputs = outputs[valid_mask]

    if np.isnan(outputs).any():
        print("Warning: Valid outputs contain NaN!")
        nan_count = np.isnan(outputs).sum()
        print(f"NaN count: {nan_count}, Total valid samples: {len(outputs)}")

    labels = labels.flatten()
    outputs = outputs.flatten()
    predicted = (outputs > 0.5).astype(float)

    precision, recall, _ = precision_recall_curve(labels, outputs)
    tn, fp, fn, tp = confusion_matrix(labels, predicted).ravel()

    # --- AUC10% ---
    fpr, tpr, _ = roc_curve(labels, outputs)
    max_fpr = 0.1
    mask = fpr <= max_fpr
    fpr_filtered = np.append(fpr[mask], max_fpr)
    tpr_filtered = np.append(tpr[mask], np.interp(max_fpr, fpr, tpr))
    auc10 = auc(fpr_filtered, tpr_filtered) / max_fpr

    # --- BAC(Balanced Accuracy) ---
    sensitivity = recall_score(labels, predicted)
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
    bac = (sensitivity + specificity) / 2

    return {
        'accuracy': (predicted == labels).mean(),
        'auc_roc': roc_auc_score(labels, outputs),
        'auc_pr': auc(recall, precision),
        'mcc': matthews_corrcoef(labels, predicted),
        'recall': recall_score(labels, predicted),
        'specificity': tn / (tn + fp),
        'precision': precision_score(labels, predicted),
        'f1': f1_score(labels, predicted),
        'bac': round(bac, 4),
        'auc10%': round(auc10, 4)
    }


# Step 6: Test model
def test_model_ensemble(test_file, batch_converter, esm_model, batch_size, model_paths):
    sequences, labels = load_sequences_from_fasta(test_file)
    test_set = ProteinDataset(sequences, labels, batch_converter, esm_model, fold_type="val")
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    ensemble_outputs = []
    all_labels = []
    for batch in test_loader:
        # all_labels.extend(batch['labels'].numpy())
        all_labels.extend(batch['labels'].cpu().numpy())

    for model_path in model_paths:
        model = TCNModel(num_inputs=args['embedding_dim'], num_channels=args['num_channels'])
        model.load_state_dict(torch.load(model_path))
        model.to(device)
        model.eval()

        outputs = []
        with torch.no_grad():
            for batch in test_loader:
                encoded_sequences = batch['encoded_sequence'].to(device)
                model_output = model(encoded_sequences)
                batch_outputs, weights = model_output
                outputs.append(batch_outputs.cpu().numpy())
        ensemble_outputs.append(np.concatenate(outputs))

    final_outputs = np.mean(ensemble_outputs, axis=0)
    metrics = compute_metrics(np.array(all_labels), final_outputs)
    metrics = {key: round(value, 3) for key, value in metrics.items()}
    print("Ensemble Test Metrics:")
    print(tabulate(metrics.items(), headers=["Metric", "Value"], tablefmt="plain"))
    logging.info("Ensemble Test Metrics:")
    logging.info(tabulate(metrics.items(), headers=["Metric", "Value"], tablefmt="plain"))


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.5, smooth=1e-4, reduction='mean'):
        """
        参数说明：
        - alpha: 类别权重因子（[0,1]），平衡正负样本，默认0.5（均衡）
        - gamma: 难易样本调节因子（≥0），默认2.0（原论文最优值）
        - smooth: 数值稳定性因子，避免log(0)或分母为0，默认1e-6
        - reduction: 损失聚合方式，可选'mean'/'sum'/'none'，默认'mean'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
        self.reduction = reduction
        # 初始化alpha为可训练参数（通过sigmoid映射到[0,1]）
        # self.alpha_param = nn.Parameter(torch.tensor(0.0))  # 初始值对应sigmoid(0)=0.5
        # self.alpha_param = nn.Parameter(torch.tensor(2.1972))

    def forward(self, inputs, targets):
        inputs = torch.clamp(inputs, min=self.smooth, max=1 - self.smooth)
        p_t = inputs * targets + (1 - inputs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = -alpha_t * torch.pow(torch.clamp(1 - p_t, min=self.smooth), self.gamma) * torch.log(p_t + self.smooth)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss  # 返回与inputs形状一致的逐元素损失
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}, must be 'mean'/'sum'/'none'")

print("******************* FocalLoss : alpha=0.9 gamma=2.0, smooth=1e-6, reduction='mean' *********************")
logging.info("******************* FocalLoss : alpha=0.9 gamma=2.0, smooth=1e-6, reduction='mean' *********************")


# Step 7: 5-Fold Cross-Validation with model loading for best model evaluation
def cross_validate_model(file, batch_converter, esm_model, epochs, patience):
    sequences, labels = load_sequences_from_fasta(file)

    total_residues = 0
    count_1 = 0
    count_0 = 0

    for seq_labels in labels:
        for label in seq_labels:
            total_residues += 1
            if label == 1:
                count_1 += 1
            elif label == 0:
                count_0 += 1

    print(f"Total: {total_residues}; epitope: {count_1}; non-epitope: {count_0}; ratio: 1:{count_0/count_1}")
    logging.info(f"Total: {total_residues}; epitope: {count_1}; non-epitope: {count_0}; ratio: 1:{count_0/count_1}")

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    all_results = []

    model = TCNModel(num_inputs=args['embedding_dim'], num_channels=args['num_channels'])
    model.to(device)
    summary(model)
    print(model)
    logging.info(model)

    for fold, (train_idx, val_idx) in enumerate(kf.split(sequences)):
        print(f"Fold {fold + 1}/5")
        logging.info(f"Fold {fold + 1}/5")

        train_sequences = [sequences[i] for i in train_idx]
        val_sequences = [sequences[i] for i in val_idx]
        train_labels = [labels[i] for i in train_idx]
        val_labels = [labels[i] for i in val_idx]

        train_dataset = ProteinDataset(train_sequences, train_labels, batch_converter, esm_model, fold_type="train")
        val_dataset = ProteinDataset(val_sequences, val_labels, batch_converter, esm_model, fold_type="val")
        train_loader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args['batch_size'], shuffle=False)

        # Model initialization
        model = TCNModel(num_inputs=args['embedding_dim'], num_channels=args['num_channels'])
        model.to(device)

        criterion = FocalLoss(alpha=0.9, gamma=2.0)  # alpha可根据类别比例调整
        # criterion = FocalLoss(gamma=2.0)  # 仅指定gamma，alpha由内部参数控制
        criterion.to(device)
        # criterion = nn.BCEWithLogitsLoss()

        # 定义优化器和学习率调度器
        # optimizer = optim.Adam(model.parameters(), lr=args['lr'])
        optimizer = optim.Adam(list(model.parameters()) + list(criterion.parameters()), lr=args['lr'])

        # 指数衰减 - 每个epoch乘以0.95
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        print("******************* ExponentialLR : gamma=0.95 *********************")
        logging.info("******************* ExponentialLR : gamma=0.95 *********************")

        model_path = f'Best/fold_{fold + 1}_full_(150m)_T2_tcn_all_lr_pu_ligbm.pth'
        train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs, patience, model_path)

        model.load_state_dict(torch.load(model_path))
        val_loss, val_results = evaluate_model(model, val_loader, criterion)
        val_results = {key: round(value, 3) for key, value in val_results.items()}
        all_results.append(val_results)
        print(f"Validation Results for Fold {fold + 1}: {val_results}")
        logging.info(f"Validation Results for Fold {fold + 1}: {val_results}")

    avg_results = {key: round(np.mean([result[key] for result in all_results]), 3) for key in all_results[0].keys()}
    print("\nAverage Results across all folds:")
    print(tabulate(avg_results.items(), headers=['Metric', 'Average Value'], tablefmt="plain"))
    logging.info("\nAverage Results across all folds:")
    logging.info(tabulate(avg_results.items(), headers=['Metric', 'Average Value'], tablefmt="plain"))


if __name__ == "__main__":
    start_time = time.time()

    train_file = "BP3_582.fasta"
    test_file1 = "BP3_15.fasta"
    test_file2 = "Homo.fasta"
    test_file3 = "Mus.fasta"
    # train_file = "200.fasta"
    # test_file1 = "45.fasta"

    cross_validate_model(train_file, batch_converter, esm_model, args['epochs'], args['patience'])

    model_paths = [f'Best/fold_{i + 1}_full_(150m)_T2_tcn_all_lr_pu_ligbm.pth' for i in range(5)]
    test_model_ensemble(test_file1, batch_converter, esm_model, args['batch_size'], model_paths)
    test_model_ensemble(test_file2, batch_converter, esm_model, args['batch_size'], model_paths)
    test_model_ensemble(test_file3, batch_converter, esm_model, args['batch_size'], model_paths)

    elapsed_time = time.time() - start_time
    print(f"Total time: {elapsed_time // 3600:.0f}h {(elapsed_time % 3600) // 60:.0f}m {elapsed_time % 60:.2f}s")
    logging.info(f"Total time: {elapsed_time // 3600:.0f}h {(elapsed_time % 3600) // 60:.0f}m {elapsed_time % 60:.2f}s")

