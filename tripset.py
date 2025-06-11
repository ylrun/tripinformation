import faulthandler; faulthandler.enable()

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.inspection import permutation_importance
import xgboost as xgb

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader

print("STEP 1: Imports complete")

# 1. Load the CSV file
print("STEP 2: About to read CSV")
tf = 'trippub.csv'
df = pd.read_csv(tf)
print(f"STEP 2: CSV loaded, shape={df.shape}")

# 2. Remove rows with missing VEHTYPE and encode target
print("STEP 3: Dropping NA on VEHTYPE")
df = df.dropna(subset=['VEHTYPE'])
print(f"STEP 3: After dropna, shape={df.shape}")

le_target = LabelEncoder()
df['target'] = le_target.fit_transform(df['VEHTYPE'])
num_classes = len(le_target.classes_)
print(f"STEP 3: Label encoding done, num_classes={num_classes}")

# 3. Encode the first 10 destination codes as a sequence
print("STEP 4: Encoding sequence columns")
max_segments = 10
seq_cols = [f'ONTD_P{i}' for i in range(1, max_segments+1)]
seq_raw = df[seq_cols].fillna('0').astype(str).values
le_seq = LabelEncoder()
le_seq.fit(seq_raw.flatten())
X_seq = le_seq.transform(seq_raw.flatten()).reshape(-1, max_segments)
print("STEP 4: Sequence encoding done")

# 4. Select and scale an expanded set of 18 numeric features
print("STEP 5: Scaling numeric features")
num_cols = [
    'VMT_MILE','TRPMILES','HHSIZE','TRVLCMIN','HHVEHCNT',
    'NUMTRANS','GASPRICE','TDTRPNUM','TRWAITTM',
    'TRPMILAD','DWELTIME','TDWKND','PUBTRANS','PSGR_FLG',
    'NUMONTRP','TRPTRANS','TRPHHVEH','TRPHHACC'
]
X_num = df[num_cols].fillna(0).values.astype(float)
scaler = StandardScaler()
X_num = scaler.fit_transform(X_num)
y = df['target'].values
print("STEP 5: Numeric scaling done")

# 5. Split into train/test
print("STEP 6: Performing train_test_split")
X_seq_train, X_seq_test, X_num_train, X_num_test, y_train, y_test = train_test_split(
    X_seq, X_num, y, test_size=0.2, stratify=y, random_state=42
)
print("STEP 6: Split done")

# 6. Compute class weights for imbalance
class_counts = np.bincount(y_train)
class_weights = 1.0 / (class_counts + 1e-6)
class_weights = class_weights / class_weights.sum() * num_classes
weight_tensor = torch.tensor(class_weights, dtype=torch.float32)

# 7. Build DataLoaders
print("STEP 7: Building DataLoaders")
class TripChainDataset(Dataset):
    def __init__(self, seq, num, labels):
        self.seq = torch.tensor(seq, dtype=torch.long)
        self.num = torch.tensor(num, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return self.seq[idx], self.num[idx], self.labels[idx]

train_loader = DataLoader(
    TripChainDataset(X_seq_train, X_num_train, y_train),
    batch_size=64, shuffle=True
)
test_loader = DataLoader(
    TripChainDataset(X_seq_test,  X_num_test,  y_test),
    batch_size=64
)
print("STEP 7: DataLoaders ready")

# 8. Define a richer Transformer + deeper numeric MLP
print("STEP 8a: Instantiating RichTransformerClassifier")
class RichTransformerClassifier(nn.Module):
    def __init__(self,
                 num_codes,
                 embed_dim=48,
                 seq_len=10,
                 num_numeric=18,
                 n_heads=4,
                 ff_dim=512,
                 n_layers=4,
                 n_classes=11):
        super().__init__()
        # sequence branch
        self.embedding     = nn.Embedding(num_codes, embed_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, embed_dim))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            batch_first=True,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        # numeric branch
        self.num_mlp = nn.Sequential(
            nn.Linear(num_numeric, ff_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(ff_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # fusion & classifier
        self.norm = nn.LayerNorm(embed_dim * 2)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 2, ff_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(ff_dim, n_classes)
        )

    def forward(self, seq, num):
        # sequence path
        x = self.embedding(seq) + self.pos_embedding    # (B, L, D)
        x = self.transformer(x).mean(dim=1)             # (B, D)
        # numeric path
        n = self.num_mlp(num)                           # (B, D)
        # fuse + predict
        combined = torch.cat([x, n], dim=1)             # (B, 2D)
        combined = self.norm(combined)
        return self.classifier(combined)                # (B, C)

print("STEP 8b: Moving model to device")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = RichTransformerClassifier(
    num_codes=len(le_seq.classes_),
    embed_dim=48,
    seq_len=max_segments,
    num_numeric=len(num_cols),
    n_heads=4,
    ff_dim=512,
    n_layers=4,
    n_classes=num_classes
).to(device)
print(f"STEP 8c: Model is on {device}")

# 8.1 Weighted loss
print("STEP 8d: About to create weighted loss")
criterion = nn.CrossEntropyLoss(weight=weight_tensor.to(device))
print("STEP 8e: Criterion created")

# 8.2 Optimizer & scheduler
print("STEP 8f: About to create optimizer (AdamW)")
optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-2)
scheduler = CosineAnnealingLR(optimizer, T_max=30)
print("STEP 8g: Optimizer and scheduler created")

# 9. Training & evaluation
def train_epoch(loader):
    model.train()
    total_loss = 0
    for seq, num, labels in loader:
        seq, num, labels = seq.to(device), num.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(seq, num)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * seq.size(0)
    scheduler.step()
    return total_loss / len(loader.dataset)

def evaluate(loader):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for seq, num, labels in loader:
            seq, num = seq.to(device), num.to(device)
            logits = model(seq, num)
            preds.append(logits.argmax(dim=1).cpu().numpy())
            targets.append(labels.numpy())
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    mode_names = [str(c) for c in le_target.classes_]
    print(classification_report(targets, preds, target_names=mode_names, zero_division=0))

# 10. Train updated Transformer
print("STEP 9: Entering training loop")
for epoch in range(1, 31):
    loss = train_epoch(train_loader)
    print(f"Epoch {epoch}: loss={loss:.4f}")

print("=== Updated Transformer Evaluation ===")
evaluate(test_loader)

# 11. Train & evaluate XGBoost baseline
print("STEP 10: Running XGBoost baseline")
xgb_clf = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    eval_metric='mlogloss'
)
xgb_clf.fit(X_num_train, y_train)
y_pred = xgb_clf.predict(X_num_test)

mode_names = [str(c) for c in le_target.classes_]
print("=== XGBoost Evaluation ===")
print(classification_report(
    y_test,
    y_pred,
    target_names=mode_names,
    zero_division=0
))

# 12. Permutation Importance for XGBoost
print("STEP 11: Computing permutation importance")
perm_result = permutation_importance(
    xgb_clf, X_num_test, y_test,
    n_repeats=10, random_state=42, scoring="accuracy"
)
imp_df = (
    pd.DataFrame({
        "feature": num_cols,
        "importance": perm_result.importances_mean
    })
    .sort_values("importance", ascending=False)
)
print("Top 10 important numeric features:")
print(imp_df.head(10))
