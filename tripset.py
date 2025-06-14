import faulthandler; faulthandler.enable()

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_selection import mutual_info_classif
import xgboost as xgb
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
import torch.backends.cudnn as cudnn

# Mac optimization
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple Silicon GPU (MPS)")
else:
    device = torch.device("cpu")
    print("Using CPU")
    # Set number of threads for optimal CPU performance
    torch.set_num_threads(8) 

print("STEP 1: Imports complete")

# 1. Load and preprocess data
print("\nSTEP 2: Loading and preprocessing CSV")
tf = 'trippub.csv'
df = pd.read_csv(tf)
print(f"Initial shape: {df.shape}")

# 2. Clean and prepare target variable
df = df.dropna(subset=['VEHTYPE'])
print(f"After removing NA from VEHTYPE: {df.shape}")

# Analyze target distribution
target_counts = df['VEHTYPE'].value_counts()
print("\nTarget distribution:")
print(target_counts.head(10))

# Filter out very rare classes (less than 0.1% of data)
min_samples = int(0.001 * len(df))
valid_classes = target_counts[target_counts >= min_samples].index
df = df[df['VEHTYPE'].isin(valid_classes)]
print(f"After filtering rare classes: {df.shape}")

le_target = LabelEncoder()
df['target'] = le_target.fit_transform(df['VEHTYPE'])
num_classes = len(le_target.classes_)
print(f"Number of classes: {num_classes}")

# 3. Optimized sequence encoding
print("\nSTEP 3: Optimized sequence encoding")
max_segments = 10  # Keep at 10 for faster processing
seq_cols = [f'ONTD_P{i}' for i in range(1, max_segments+1)]

# Create sequence features more efficiently
seq_raw = df[seq_cols].fillna('PAD').astype(str).values

# Vectorized sequence statistics extraction
def extract_sequence_features_vectorized(seq_data):
    """Extract statistical features from sequences - vectorized version"""
    features = np.zeros((len(seq_data), 5))
    
    for i, row in enumerate(seq_data):
        valid_codes = [code for code in row if code != 'PAD' and code != '0']
        if valid_codes:
            code_counts = Counter(valid_codes)
            features[i, 0] = len(valid_codes)  # Trip chain length
            features[i, 1] = len(code_counts)  # Number of unique destinations
            features[i, 2] = code_counts.most_common(1)[0][1]  # Max frequency
            features[i, 3] = np.std([code_counts[c] for c in code_counts])  # Frequency std
            features[i, 4] = features[i, 2] / features[i, 0] if features[i, 0] > 0 else 0  # Concentration ratio
    
    return features

print("Extracting sequence features...")
seq_stats = extract_sequence_features_vectorized(seq_raw)

# Encode sequences
le_seq = LabelEncoder()
all_codes = np.unique(seq_raw.flatten())
le_seq.fit(all_codes)
X_seq = le_seq.transform(seq_raw.flatten()).reshape(-1, max_segments)

# 4. Optimized feature engineering
print("\nSTEP 4: Optimized feature engineering")

# Core numeric features that are most predictive
core_num_cols = [
    'VMT_MILE','TRPMILES','HHSIZE','TRVLCMIN','HHVEHCNT',
    'NUMTRANS','TDTRPNUM','TRWAITTM','DWELTIME','TDWKND',
    'PUBTRANS','NUMONTRP','TRPHHACC'
]

# Create engineered features efficiently
print("Creating engineered features...")
eng_features = pd.DataFrame()
eng_features['trips_per_mile'] = df['TDTRPNUM'] / (df['TRPMILES'] + 1)
eng_features['avg_trip_time'] = df['TRVLCMIN'] / (df['TDTRPNUM'] + 1)
eng_features['vehicle_ratio'] = df['HHVEHCNT'] / (df['HHSIZE'] + 1)
eng_features['wait_time_ratio'] = df['TRWAITTM'] / (df['TRVLCMIN'] + 1)
eng_features['public_transit_usage'] = df['PUBTRANS'] * df['NUMTRANS']
eng_features['trip_complexity'] = df['NUMONTRP'] / (df['TDTRPNUM'] + 1)

# Combine features
X_num = pd.concat([df[core_num_cols], eng_features], axis=1)

# Fast missing value imputation
X_num = X_num.fillna(X_num.median())

# Use StandardScaler for faster processing
scaler = StandardScaler()
X_num_scaled = scaler.fit_transform(X_num)

# Combine with sequence statistics
X_combined = np.hstack([X_num_scaled, seq_stats])
print(f"Total features: {X_combined.shape[1]}")

y = df['target'].values

# 5. Train/test split
print("\nSTEP 5: Creating train/test split")
X_seq_train, X_seq_test, X_num_train, X_num_test, y_train, y_test = train_test_split(
    X_seq, X_combined, y, test_size=0.2, stratify=y, random_state=42
)

# 6. Compute balanced class weights
class_counts = np.bincount(y_train)
class_weights = len(y_train) / (num_classes * class_counts)
weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

# 7. Optimized DataLoader for CPU
print("\nSTEP 6: Building CPU-optimized DataLoaders")

class TripChainDataset(Dataset):
    def __init__(self, seq, num, labels):
        self.seq = torch.tensor(seq, dtype=torch.long)
        self.num = torch.tensor(num, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.seq[idx], self.num[idx], self.labels[idx]

# Smaller batch size for CPU
batch_size = 64

train_dataset = TripChainDataset(X_seq_train, X_num_train, y_train)
test_dataset = TripChainDataset(X_seq_test, X_num_test, y_test)

# No workers for Mac to avoid multiprocessing issues
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    num_workers=0
)

# 8. Lightweight but effective model for CPU
print("\nSTEP 7: Building CPU-optimized model")

class LightweightTransformerClassifier(nn.Module):
    def __init__(self,
                 num_codes,
                 embed_dim=32,  
                 seq_len=10,
                 num_numeric=None,
                 n_heads=4,  
                 ff_dim=256,  
                 n_layers=2, 
                 n_classes=11,
                 dropout=0.15):
        super().__init__()
        
        # Efficient sequence embedding
        self.embedding = nn.Embedding(num_codes, embed_dim, padding_idx=0)
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, embed_dim) * 0.02)
        self.embed_dropout = nn.Dropout(dropout)
        
        # Lightweight transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation='relu',  # ReLU is faster than GELU on CPU
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Simple pooling
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        # Efficient numeric processing
        self.num_projection = nn.Sequential(
            nn.Linear(num_numeric, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim)
        )
        
        # Final classifier with skip connection
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 2, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, n_classes)
        )
    
    def forward(self, seq, num):
        # Sequence processing
        x = self.embedding(seq)
        x = x + self.pos_embedding
        x = self.embed_dropout(x)
        x = self.transformer(x)
        
        # Efficient pooling
        x = x.transpose(1, 2)  # (B, D, L)
        x = self.pool(x).squeeze(-1)  # (B, D)
        
        # Numeric processing
        n = self.num_projection(num)
        
        # Combine and classify
        combined = torch.cat([x, n], dim=1)
        return self.classifier(combined)

# Initialize model
model = LightweightTransformerClassifier(
    num_codes=len(le_seq.classes_),
    embed_dim=32,
    seq_len=max_segments,
    num_numeric=X_combined.shape[1],
    n_heads=4,
    ff_dim=256,
    n_layers=2,
    n_classes=num_classes
).to(device)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# 9. Training setup with early stopping
criterion = nn.CrossEntropyLoss(weight=weight_tensor)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

# 10. Fast training functions
def train_epoch(loader):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for seq, num, labels in loader:
        seq, num, labels = seq.to(device), num.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(seq, num)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item() * seq.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return total_loss / total, 100. * correct / total

def evaluate(loader, return_predictions=False):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for seq, num, labels in loader:
            seq, num, labels = seq.to(device), num.to(device), labels.to(device)
            
            outputs = model(seq, num)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item() * seq.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if return_predictions:
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / total
    accuracy = 100. * correct / total
    
    if return_predictions:
        return avg_loss, accuracy, all_preds, all_targets
    return avg_loss, accuracy

# 11. Training with early stopping
print("\nSTEP 8: Training model")
best_acc = 0
patience_counter = 0
max_patience = 5

for epoch in range(1, 11):  # Max 30 epochs
    train_loss, train_acc = train_epoch(train_loader)
    val_loss, val_acc = evaluate(test_loader)
    
    scheduler.step(val_loss)
    
    print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    
    # Early stopping
    if val_acc > best_acc:
        best_acc = val_acc
        patience_counter = 0
        # Save best model state
        best_model_state = model.state_dict()
    else:
        patience_counter += 1
        if patience_counter >= max_patience:
            print(f"Early stopping at epoch {epoch}")
            break

# Load best model
model.load_state_dict(best_model_state)

# 12. Final evaluation
print("\n=== Final Neural Network Evaluation ===")
_, _, preds, targets = evaluate(test_loader, return_predictions=True)
mode_names = [str(c) for c in le_target.classes_]
print(classification_report(targets, preds, target_names=mode_names, zero_division=0))

# 13. XGBoost for comparison (CPU optimized)
print("\n=== XGBoost Baseline (CPU Optimized) ===")
xgb_params = {
    'n_estimators': 200,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'objective': 'multi:softprob',
    'eval_metric': 'mlogloss',
    'n_jobs': -1,  # Use all CPU cores
    'tree_method': 'hist',  # Fast histogram-based method
    'random_state': 42
}

xgb_clf = xgb.XGBClassifier(**xgb_params)
xgb_clf.fit(X_num_train, y_train)
y_pred = xgb_clf.predict(X_num_test)

print("XGBoost Classification Report:")
print(classification_report(y_test, y_pred, target_names=mode_names, zero_division=0))

# 14. Feature importance from XGBoost
feature_names = list(X_num.columns) + [f'seq_stat_{i}' for i in range(seq_stats.shape[1])]
importances = xgb_clf.feature_importances_
indices = np.argsort(importances)[::-1][:15]

print("\nTop 15 Most Important Features:")
for i, idx in enumerate(indices):
    print(f"{i+1}. {feature_names[idx]}: {importances[idx]:.4f}")

print("\nTraining complete!")
print(f"Best Neural Network Accuracy: {best_acc:.2f}%")
print(f"XGBoost Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
