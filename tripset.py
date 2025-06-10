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

# 2. Remove rows with missing target and encode VEHTYPE
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

# 4. Select and scale ONLY the Top-9 numeric features
print("STEP 5: Scaling numeric features")
num_cols = [
    'VMT_MILE','TRPMILES','HHSIZE','TRVLCMIN','HHVEHCNT',
    'NUMTRANS','GASPRICE','TDTRPNUM','TRWAITTM'
]
X_num = df[num_cols].fillna(0).values.astype(float)
scaler = StandardScaler()
X_num = scaler.fit_transform(X_num)
y = df['target'].values
print("STEP 5: Numeric scaling done")

# 5. Split into training and test sets
print("STEP 6: Performing train_test_split")
X_seq_train, X_seq_test, X_num_train, X_num_test, y_train, y_test = train_test_split(
    X_seq, X_num, y, test_size=0.2, stratify=y, random_state=42
)
print("STEP 6: Split done")

# 6. Compute class weights to address imbalance
class_counts = np.bincount(y_train)
class_weights = 1.0 / (class_counts + 1e-6)
class_weights = class_weights / class_weights.sum() * num_classes
weight_tensor = torch.tensor(class_weights, dtype=torch.float32)

# 7. Create PyTorch Dataset and DataLoader
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

# 8. Define a higher-capacity Transformer + deeper numeric MLP
print("STEP 8a: Instantiating UpdatedTransformerClassifier")
class UpdatedTransformerClassifier(nn.Module):
    def __init__(self,
                 num_codes,
                 embed_dim=32,
                 seq_len=10,
                 num_numeric=9,
                 n_heads=4,
                 ff_dim=256,
                 n_layers=4,
                 n_classes=11):
        super().__init__()
        # sequence branch
        self.embedding     = nn.Embedding(num_codes, embed_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, embed_dim))
        encoder_layer      = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            batch_first=True,
            dropout=0.1
        )
        self.transformer   = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        # numeric branch: deep MLP
        self.num_mlp = nn.Sequential(
            nn.Linear(num_numeric, ff_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(ff_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        # fusion + classifier head
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
        # fuse, normalize, classify
        combined = self.norm(torch.cat([x, n], dim=1))  # (B, 2D)
        return self.classifier(combined)                # (B, C)

print("STEP 8b: Moving model to device")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UpdatedTransformerClassifier(
    num_codes=len(le_seq.classes_),
    embed_dim=32,
    seq_len=max_segments,
    num_numeric=len(num_cols),
    n_heads=4,
    ff_dim=256,
    n_layers=4,
    n_classes=num_classes
).to(device)
print(f"STEP 8c: Model is on {device}")

# 8.1 Create weighted loss function
print("STEP 8d: About to create loss with class weights")
criterion = nn.CrossEntropyLoss(weight=weight_tensor.to(device))
print("STEP 8e: Criterion created")

# 8.2 Create optimizer (AdamW) and scheduler
print("STEP 8f: About to create optimizer (AdamW)")
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
scheduler = CosineAnnealingLR(optimizer, T_max=10)
print("STEP 8g: Optimizer and scheduler created")

# 9. Define training and evaluation procedures
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

# 10. Train the updated Transformer model
print("STEP 9: Entering training loop")
for epoch in range(1, 21):
    loss = train_epoch(train_loader)
    print(f"Epoch {epoch}: loss={loss:.4f}")

print("=== Updated Transformer Evaluation ===")
evaluate(test_loader)

# 11. Train and evaluate the XGBoost baseline
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

# 12. Permutation Importance on numeric features
print("STEP 11: Computing permutation importance for XGBoost")
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
