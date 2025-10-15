import zipfile
import os

zip_filename = "/content/68e8d1d70b66d_student_resource.zip"

with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
    extract_path = "/content/extracted_files"
    zip_ref.extractall(extract_path)

print(f"Extracted to: {extract_path}")

for root, dirs, files in os.walk(extract_path):
    level = root.replace(extract_path, '').count(os.sep)
    indent = ' ' * 2 * level
    print(f"{indent}{os.path.basename(root)}/")
    subindent = ' ' * 2 * (level + 1)
    for f in files:
        print(f"{subindent}{f}")



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

base_path = "/content/extracted_files/student_resource/dataset"

train_path = f"{base_path}/train.csv"
test_path  = f"{base_path}/test.csv"

train_df = pd.read_csv(train_path)
test_df  = pd.read_csv(test_path)

print("Train shape:", train_df.shape)
print("Test shape :", test_df.shape)

train_df.head()


!pip install catboost lightgbm timm sentence-transformers


# Full improved pipeline (run with GPU). Debug mode uses a smaller subset for fast iteration.

# ====== 0. Requirements (run once) ======
# !pip install -q sentence-transformers timm catboost lightgbm xgboost optuna

# ====== 1. Imports ======
import os, gc, re, math, time
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

import timm
from sentence_transformers import SentenceTransformer

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

import lightgbm as lgb
from catboost import CatBoostRegressor

# ====== 2. Config & flags ======
DEBUG = False           # True = run on small subset to test pipeline (fast). Set to False for full run.
USE_IMAGES = True       # set False to skip image embedding extraction (faster)
USE_TEXT_EMB = True     # set False to skip sentence-transformer embedding
N_FOLDS = 5
SEED = 42
BATCH_SIZE = 128
NN_EPOCHS = 50 if not DEBUG else 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)

# ====== 3. Paths ======
base_path = "/content/extracted_files/student_resource/dataset"
image_dir = "/content/extracted_files/images"  # images must be downloaded into this folder
train_csv = os.path.join(base_path, "train.csv")
test_csv  = os.path.join(base_path, "test.csv")

# ====== 4. Utility functions ======
def smape_np(y_true, y_pred):
    eps = 1e-9
    num = np.abs(y_pred - y_true)
    den = (np.abs(y_true) + np.abs(y_pred)) / 2.0 + eps
    return 100.0 * np.mean(num / den)

def extract_ipq(text):
    text = str(text).lower()
    m = re.search(r'(?:pack of|pack|pcs|pc|pieces|x)\s*([0-9]{1,4})', text)
    if m: return float(m.group(1))
    m2 = re.search(r'([0-9]+(?:\.[0-9]+)?)\s*(ml|l|g|kg|oz|count|ct|tablet|capsule|piece|pcs|pc)', text)
    if m2: return float(m2.group(1))
    m3 = re.search(r'([0-9]{1,4})', text)
    if m3: return float(m3.group(1))
    return 1.0

# ====== 5. Load data ======
train_df = pd.read_csv(train_csv)
test_df  = pd.read_csv(test_csv)

if DEBUG:
    train_df = train_df.sample(2000, random_state=SEED).reset_index(drop=True)
    test_df  = test_df.sample(500, random_state=SEED).reset_index(drop=True)

print("Train shape:", train_df.shape, "Test shape:", test_df.shape)

# ====== 6. Basic feature engineering ======
for df in (train_df, test_df):
    df['catalog_content'] = df['catalog_content'].astype(str)
    df['ipq'] = df['catalog_content'].apply(extract_ipq)
    df['text_length'] = df['catalog_content'].apply(len)
    df['word_count'] = df['catalog_content'].apply(lambda x: len(x.split()))
    df['digits_count'] = df['catalog_content'].apply(lambda x: sum(c.isdigit() for c in x))
    # optional: price per unit (only for train)
if 'price' in train_df.columns:
    train_df['price_per_ipq'] = train_df['price'] / (train_df['ipq'].replace(0,1))
    # but do not use price_per_ipq as feature for training directly (leak); ok for EDA only

numeric_cols = ["ipq", "text_length", "word_count", "digits_count"]
X_numeric_train = train_df[numeric_cols].values
X_numeric_test  = test_df[numeric_cols].values

# We'll train on raw prices (not log) to directly optimize SMAPE.
y_train = train_df['price'].values.astype(np.float32)

# ====== 7. Text features: TF-IDF + SVD + Sentence-Transformer embeddings ======
print("Building TF-IDF+SVD...")
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2), min_df=3)
tfidf_train = tfidf.fit_transform(train_df['catalog_content'])
tfidf_test  = tfidf.transform(test_df['catalog_content'])
svd = TruncatedSVD(n_components=100, random_state=SEED)
X_tfidf_train = svd.fit_transform(tfidf_train)
X_tfidf_test  = svd.transform(tfidf_test)
print("TF-IDF+SVD shapes:", X_tfidf_train.shape, X_tfidf_test.shape)

X_text_train = np.zeros((len(train_df), 0))
X_text_test  = np.zeros((len(test_df), 0))
if USE_TEXT_EMB:
    print("Computing sentence-transformer embeddings (this can take time)...")
    txt_model = SentenceTransformer('all-mpnet-base-v2', device=DEVICE)
    X_text_train = txt_model.encode(train_df['catalog_content'].tolist(), batch_size=32, show_progress_bar=True)
    X_text_test  = txt_model.encode(test_df['catalog_content'].tolist(), batch_size=32, show_progress_bar=True)
    print("Text emb shapes:", X_text_train.shape, X_text_test.shape)

# ====== 8. Image embeddings (EfficientNet-B3) ======
X_img_train = np.zeros((len(train_df), 0))
X_img_test  = np.zeros((len(test_df), 0))
if USE_IMAGES:
    print("Computing image embeddings (this can take a lot of time and GPU memory)...")
    img_model = timm.create_model('efficientnet_b3', pretrained=True, num_classes=0).to(DEVICE)
    img_model.eval()
    transform = timm.data.transforms_factory.create_transform(input_size=300, is_training=False)

    def encode_images(df):
        embs = []
        for sid in tqdm(df['sample_id'].tolist()):
            pjpg = os.path.join(image_dir, f"{sid}.jpg")
            if not os.path.exists(pjpg):
                # try jpg/png fallback
                pjpg = os.path.join(image_dir, f"{sid}.png")
            if os.path.exists(pjpg):
                try:
                    img = Image.open(pjpg).convert('RGB')
                    img_t = transform(img).unsqueeze(0).to(DEVICE)
                    with torch.no_grad():
                        out = img_model(img_t).cpu().numpy().reshape(-1)
                    embs.append(out)
                except Exception as e:
                    embs.append(np.zeros(img_model.num_features, dtype=np.float32))
            else:
                embs.append(np.zeros(img_model.num_features, dtype=np.float32))
        return np.vstack(embs)

    X_img_train = encode_images(train_df)
    X_img_test  = encode_images(test_df)
    print("Image emb shapes:", X_img_train.shape, X_img_test.shape)

# ====== 9. Combine features and scale ======
to_hstack_train = [X_numeric_train, X_tfidf_train]
to_hstack_test  = [X_numeric_test,  X_tfidf_test]
if USE_TEXT_EMB:
    to_hstack_train.append(X_text_train)
    to_hstack_test.append(X_text_test)
if USE_IMAGES:
    to_hstack_train.append(X_img_train)
    to_hstack_test.append(X_img_test)

X_train_full = np.hstack(to_hstack_train)
X_test_full  = np.hstack(to_hstack_test)
print("Combined feature shapes:", X_train_full.shape, X_test_full.shape)

scaler = StandardScaler()
X_train_full = scaler.fit_transform(X_train_full)
X_test_full  = scaler.transform(X_test_full)

# To speed up debugging, you may reduce dimensions (optional)
# ====== 10. Neural network for OOF predictions ======
class PriceDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None
    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]

class FusionNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

# SMAPE loss in torch operates on raw prices
def torch_smape(preds, targets, eps=1e-6):
    numer = torch.abs(preds - targets)
    denom = (torch.abs(preds) + torch.abs(targets) + eps) / 2.0
    return torch.mean(numer / denom) * 100.0

# ====== 11. OOF training with KFold (NN) ======
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
oof_preds = np.zeros(len(X_train_full), dtype=np.float32)
test_preds_folds = np.zeros((N_FOLDS, len(X_test_full)), dtype=np.float32)

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train_full), 1):
    print(f"\n=== NN Fold {fold}/{N_FOLDS} ===")
    X_tr, y_tr = X_train_full[tr_idx], y_train[tr_idx]
    X_val, y_val = X_train_full[val_idx], y_train[val_idx]

    train_ds = PriceDataset(X_tr, y_tr)
    val_ds   = PriceDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=2)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE*2, shuffle=False, pin_memory=True, num_workers=2)

    model = FusionNN(X_train_full.shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)
    best_val = 1e9
    patience = 6
    no_imp = 0

    for epoch in range(1, NN_EPOCHS+1):
        model.train()
        train_losses = []
        for Xb, yb in train_loader:
            Xb = Xb.to(DEVICE); yb = yb.to(DEVICE)
            opt.zero_grad()
            preds = model(Xb)
            loss = torch_smape(preds, yb)
            loss.backward()
            opt.step()
            train_losses.append(loss.item())
        # validation
        model.eval()
        val_preds = []
        val_trues = []
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb = Xb.to(DEVICE); yb = yb.to(DEVICE)
                p = model(Xb).cpu().numpy()
                val_preds.append(p)
                val_trues.append(yb.cpu().numpy())
        val_preds = np.concatenate(val_preds)
        val_trues = np.concatenate(val_trues)
        val_sm = smape_np(val_trues, val_preds)
        scheduler.step(val_sm)
        print(f"Fold{fold} Epoch{epoch} TrainSMAPE:{np.mean(train_losses):.3f} ValSMAPE:{val_sm:.3f}")
        if val_sm + 1e-5 < best_val:
            best_val = val_sm
            no_imp = 0
            torch.save(model.state_dict(), f"nn_best_fold{fold}.pt")
        else:
            no_imp += 1
            if no_imp >= patience:
                print("Early stopping NN")
                break

    # load best and predict OOF + test
    model.load_state_dict(torch.load(f"nn_best_fold{fold}.pt"))
    model.eval()
    with torch.no_grad():
        # OOF
        vals = []
        val_loader_all = DataLoader(val_ds, batch_size=BATCH_SIZE*2, shuffle=False)
        for Xb, _ in val_loader_all:
            vals.append(model(Xb.to(DEVICE)).cpu().numpy())
        oof_preds[val_idx] = np.concatenate(vals)
        # test
        test_dl = DataLoader(PriceDataset(X_test_full), batch_size=BATCH_SIZE*2, shuffle=False)
        tvals = []
        for Xb in test_dl:
            tvals.append(model(Xb.to(DEVICE)).cpu().numpy())
        test_preds_folds[fold-1] = np.concatenate(tvals)

    # cleanup
    del model, opt, train_loader, val_loader, val_loader_all, test_dl
    torch.cuda.empty_cache()
    gc.collect()

# Aggregate NN outputs
nn_oof = oof_preds.copy()
nn_test_mean = test_preds_folds.mean(axis=0)
print("NN OOF SMAPE:", smape_np(y_train, nn_oof))

# ====== 12. Build meta features and stack ======
meta_train = np.column_stack([nn_oof, X_numeric_train])
meta_test  = np.column_stack([nn_test_mean, X_numeric_test])

# Train LightGBM on real price
lgb_params = {
    'n_estimators': 1500, 'learning_rate': 0.02, 'num_leaves': 128,
    'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 5,
    'random_state': SEED, 'verbosity': -1
}
lgb_clf = lgb.LGBMRegressor(**lgb_params)
lgb_clf.fit(meta_train, y_train, eval_set=[(meta_train, y_train)])


# Train CatBoost on real price
cb_clf = CatBoostRegressor(iterations=1200, learning_rate=0.02, depth=8, l2_leaf_reg=3,
                           loss_function='MAE', random_seed=SEED)
cb_clf.fit(meta_train, y_train, eval_set=(meta_train, y_train))

# Train a simple Ridge as a blender on the meta predictions (blend using OOF)
pred_l_g = lgb_clf.predict(meta_train)
pred_c_g = cb_clf.predict(meta_train)
stack_X = np.column_stack([pred_l_g, pred_c_g])
ridge = Ridge(alpha=1.0)
ridge.fit(stack_X, y_train)

# meta test predictions
pred_l_test = lgb_clf.predict(meta_test)
pred_c_test = cb_clf.predict(meta_test)
stack_test = np.column_stack([pred_l_test, pred_c_test])
blend_test = ridge.predict(stack_test)  # final blended prediction (real price)

# ====== 13. Postprocessing ======
# Clip negative, apply moderate smoothing for extreme high values
blend_test = np.clip(blend_test, 0.0, None)
upper = np.percentile(train_df['price'].values, 99.0)
median_price = np.median(train_df['price'].values)
blend_test = np.where(blend_test > upper * 1.5, 0.6*blend_test + 0.4*median_price, blend_test)

# Optional calibration: scale predictions to match train median if heavy bias seen
# scale = np.median(train_df['price'])/np.median(blend_test)
# blend_test *= scale

# ====== 14. Evaluation on train (OOF) ======
# Build OOF blended train prediction for final OOF SMAPE
pred_l_oof = lgb_clf.predict(np.column_stack([nn_oof, X_numeric_train]))
pred_c_oof = cb_clf.predict(np.column_stack([nn_oof, X_numeric_train]))
blend_oof = ridge.predict(np.column_stack([pred_l_oof, pred_c_oof]))
print("Final OOF SMAPE (stacked):", smape_np(train_df['price'].values, blend_oof))

# ====== 15. Save submission ======
submission = pd.DataFrame({
    "sample_id": test_df['sample_id'],
    "price": blend_test
})
submission.to_csv("test_out_stack_final.csv", index=False)
print("Saved test_out_stack_final.csv")


# Run once in a notebook cell (uncomment if not ins
!pip install -q sentence-transformers timm catboost lightgbm xgboost


import os, gc, re, math, time
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

import timm
from sentence_transformers import SentenceTransformer

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

import lightgbm as lgb
from catboost import CatBoostRegressor

# Config flags
DEBUG = False          # True for quick runs
USE_IMAGES = True      # Set False to skip images to save time/memory
USE_TEXT_EMB = True    # Sentence-transformer on/off
N_FOLDS = 5
SEED = 42
BATCH_SIZE = 128 if not DEBUG else 64
NN_EPOCHS = 60 if not DEBUG else 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)


base_path = "/content/extracted_files/student_resource/dataset"
train_csv = os.path.join(base_path, "train.csv")
test_csv  = os.path.join(base_path, "test.csv")
image_dir = "/content/extracted_files/images"

train_df = pd.read_csv(train_csv)
test_df  = pd.read_csv(test_csv)

if DEBUG:
    train_df = train_df.sample(2000, random_state=SEED).reset_index(drop=True)
    test_df  = test_df.sample(500, random_state=SEED).reset_index(drop=True)

print("Train shape:", train_df.shape, "Test shape:", test_df.shape)


def extract_ipq(text):
    text = str(text).lower()
    m = re.search(r'(?:pack of|pack|pcs|pc|pieces|x)\s*([0-9]{1,4})', text)
    if m: return float(m.group(1))
    m2 = re.search(r'([0-9]+(?:\.[0-9]+)?)\s*(ml|l|g|kg|oz|count|ct|tablet|capsule|piece|pcs|pc)', text)
    if m2: return float(m2.group(1))
    m3 = re.search(r'([0-9]{1,4})', text)
    if m3: return float(m3.group(1))
    return 1.0

for df in (train_df, test_df):
    df['catalog_content'] = df['catalog_content'].astype(str)
    df['ipq'] = df['catalog_content'].apply(extract_ipq)
    df['text_length'] = df['catalog_content'].apply(len)
    df['word_count'] = df['catalog_content'].apply(lambda x: len(x.split()))
    df['digits_count'] = df['catalog_content'].apply(lambda x: sum(c.isdigit() for c in x))

numeric_cols = ["ipq", "text_length", "word_count", "digits_count"]
X_numeric_train = train_df[numeric_cols].values
X_numeric_test  = test_df[numeric_cols].values

# Use log1p target (recommended for stability)
y_train_raw = train_df['price'].values.astype(np.float32)
y_train = np.log1p(y_train_raw)   # model target


print("Building TF-IDF + SVD...")
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2), min_df=3)
tfidf_train = tfidf.fit_transform(train_df['catalog_content'])
tfidf_test  = tfidf.transform(test_df['catalog_content'])

svd = TruncatedSVD(n_components=100, random_state=SEED)
X_tfidf_train = svd.fit_transform(tfidf_train)
X_tfidf_test  = svd.transform(tfidf_test)

print("TF-IDF+SVD shapes:", X_tfidf_train.shape, X_tfidf_test.shape)


X_text_train = np.zeros((len(train_df), 0))
X_text_test  = np.zeros((len(test_df), 0))
if USE_TEXT_EMB:
    print("Computing sentence-transformer embeddings...")
    txt_model = SentenceTransformer('all-mpnet-base-v2', device=DEVICE)
    X_text_train = txt_model.encode(train_df['catalog_content'].tolist(), batch_size=32, show_progress_bar=True)
    X_text_test  = txt_model.encode(test_df['catalog_content'].tolist(), batch_size=32, show_progress_bar=True)
    print("Text emb shapes:", X_text_train.shape, X_text_test.shape)


X_img_train = np.zeros((len(train_df), 0))
X_img_test  = np.zeros((len(test_df), 0))
if USE_IMAGES:
    print("Computing image embeddings (EfficientNet-B3)...")
    img_model = timm.create_model('efficientnet_b3', pretrained=True, num_classes=0).to(DEVICE)
    img_model.eval()
    transform = timm.data.transforms_factory.create_transform(input_size=300, is_training=False)

    def encode_images(df):
        embs = []
        for sid in tqdm(df['sample_id'].tolist()):
            pjpg = os.path.join(image_dir, f"{sid}.jpg")
            if not os.path.exists(pjpg):
                pjpg = os.path.join(image_dir, f"{sid}.png")
            if os.path.exists(pjpg):
                try:
                    img = Image.open(pjpg).convert('RGB')
                    img_t = transform(img).unsqueeze(0).to(DEVICE)
                    with torch.no_grad():
                        out = img_model(img_t).cpu().numpy().reshape(-1)
                    embs.append(out)
                except Exception:
                    embs.append(np.zeros(img_model.num_features, dtype=np.float32))
            else:
                embs.append(np.zeros(img_model.num_features, dtype=np.float32))
        return np.vstack(embs)

    X_img_train = encode_images(train_df)
    X_img_test  = encode_images(test_df)
    print("Image emb shapes:", X_img_train.shape, X_img_test.shape)


X_img_train = np.zeros((len(train_df), 0))
X_img_test  = np.zeros((len(test_df), 0))
if USE_IMAGES:
    print("Computing image embeddings (EfficientNet-B3)...")
    img_model = timm.create_model('efficientnet_b3', pretrained=True, num_classes=0).to(DEVICE)
    img_model.eval()
    transform = timm.data.transforms_factory.create_transform(input_size=300, is_training=False)

    def encode_images(df):
        embs = []
        for sid in tqdm(df['sample_id'].tolist()):
            pjpg = os.path.join(image_dir, f"{sid}.jpg")
            if not os.path.exists(pjpg):
                pjpg = os.path.join(image_dir, f"{sid}.png")
            if os.path.exists(pjpg):
                try:
                    img = Image.open(pjpg).convert('RGB')
                    img_t = transform(img).unsqueeze(0).to(DEVICE)
                    with torch.no_grad():
                        out = img_model(img_t).cpu().numpy().reshape(-1)
                    embs.append(out)
                except Exception:
                    embs.append(np.zeros(img_model.num_features, dtype=np.float32))
            else:
                embs.append(np.zeros(img_model.num_features, dtype=np.float32))
        return np.vstack(embs)

    X_img_train = encode_images(train_df)
    X_img_test  = encode_images(test_df)
    print("Image emb shapes:", X_img_train.shape, X_img_test.shape)


# Stack embedding matrices for PCA individually
from sklearn.decomposition import PCA

# TEXT PCA -> reduce to 128 (if text embeddings present)
if USE_TEXT_EMB:
    print("PCA on text embeddings...")
    pca_text = PCA(n_components=128, random_state=SEED)
    X_text_train_p = pca_text.fit_transform(X_text_train)
    X_text_test_p  = pca_text.transform(X_text_test)
else:
    X_text_train_p = np.zeros((len(train_df),0))
    X_text_test_p  = np.zeros((len(test_df),0))

# IMAGE PCA -> reduce to 256 (if images present)
if USE_IMAGES:
    print("PCA on image embeddings...")
    pca_img = PCA(n_components=256, random_state=SEED)
    X_img_train_p = pca_img.fit_transform(X_img_train)
    X_img_test_p  = pca_img.transform(X_img_test)
else:
    X_img_train_p = np.zeros((len(train_df),0))
    X_img_test_p  = np.zeros((len(test_df),0))

print("Reduced shapes:", X_text_train_p.shape, X_img_train_p.shape)


# Stack embedding matrices for PCA individually
from sklearn.decomposition import PCA

# TEXT PCA -> reduce to 128 (if text embeddings present)
if USE_TEXT_EMB:
    print("PCA on text embeddings...")
    pca_text = PCA(n_components=128, random_state=SEED)
    X_text_train_p = pca_text.fit_transform(X_text_train)
    X_text_test_p  = pca_text.transform(X_text_test)
else:
    X_text_train_p = np.zeros((len(train_df),0))
    X_text_test_p  = np.zeros((len(test_df),0))

# IMAGE PCA -> reduce to 256 (if images present)
if USE_IMAGES:
    print("PCA on image embeddings...")
    pca_img = PCA(n_components=256, random_state=SEED)
    X_img_train_p = pca_img.fit_transform(X_img_train)
    X_img_test_p  = pca_img.transform(X_img_test)
else:
    X_img_train_p = np.zeros((len(train_df),0))
    X_img_test_p  = np.zeros((len(test_df),0))

print("Reduced shapes:", X_text_train_p.shape, X_img_train_p.shape)


to_stack_train = [X_numeric_train, X_tfidf_train, X_text_train_p, X_img_train_p]
to_stack_test  = [X_numeric_test,  X_tfidf_test,  X_text_test_p,  X_img_test_p]

# filter out empty arrays
to_stack_train = [a for a in to_stack_train if a.size>0]
to_stack_test  = [a for a in to_stack_test  if a.size>0]

X_train_full = np.hstack(to_stack_train)
X_test_full  = np.hstack(to_stack_test)

print("Combined feature shapes:", X_train_full.shape, X_test_full.shape)

scaler = StandardScaler()
X_train_full = scaler.fit_transform(X_train_full)
X_test_full  = scaler.transform(X_test_full)


class PriceDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None
    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]

class FusionNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

def smape_np(y_true, y_pred):
    eps = 1e-9
    num = np.abs(y_pred - y_true)
    den = (np.abs(y_true) + np.abs(y_pred)) / 2.0 + eps
    return 100.0 * np.mean(num / den)

def torch_smape(preds, targets, eps=1e-6):
    # preds and targets are in log1p space (we train in log1p)
    p = torch.expm1(preds)
    t = torch.expm1(targets)
    numer = torch.abs(p - t)
    denom = (torch.abs(p) + torch.abs(t) + eps) / 2.0
    return torch.mean(numer / denom) * 100.0


kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
oof_preds_log = np.zeros(len(X_train_full), dtype=np.float32)   # log-space preds
test_preds_log_folds = np.zeros((N_FOLDS, len(X_test_full)), dtype=np.float32)

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train_full), 1):
    print(f"\n=== NN Fold {fold}/{N_FOLDS} ===")
    X_tr, y_tr = X_train_full[tr_idx], y_train[tr_idx]   # y_train is log1p
    X_val, y_val = X_train_full[val_idx], y_train[val_idx]

    train_ds = PriceDataset(X_tr, y_tr)
    val_ds = PriceDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE*2, shuffle=False, num_workers=2, pin_memory=True)

    model = FusionNN(X_train_full.shape[1]).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)
    best_val = 1e9
    patience = 6
    no_imp = 0

    for epoch in range(1, NN_EPOCHS+1):
        model.train()
        train_losses = []
        for Xb, yb in train_loader:
            Xb = Xb.to(DEVICE); yb = yb.to(DEVICE)
            optimizer.zero_grad()
            preds_log = model(Xb)
            loss = torch_smape(preds_log, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        # Validation
        model.eval()
        val_preds = []
        val_trues = []
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb = Xb.to(DEVICE); yb = yb.to(DEVICE)
                p = model(Xb).cpu().numpy()
                val_preds.append(p)
                val_trues.append(yb.cpu().numpy())
        val_preds = np.concatenate(val_preds)
        val_trues = np.concatenate(val_trues)
        val_sm = smape_np(np.expm1(val_trues), np.expm1(val_preds))
        scheduler.step(val_sm)
        print(f"Fold{fold} Epoch{epoch} TrainSMAPE:{np.mean(train_losses):.3f} ValSMAPE:{val_sm:.3f}")
        if val_sm + 1e-6 < best_val:
            best_val = val_sm
            no_imp = 0
            torch.save(model.state_dict(), f"nn_best_fold{fold}.pt")
        else:
            no_imp += 1
            if no_imp >= patience:
                print("Early stopping NN")
                break

    # load best and predict OOF + test for this fold
    model.load_state_dict(torch.load(f"nn_best_fold{fold}.pt"))
    model.eval()
    with torch.no_grad():
        # OOF
        val_loader_all = DataLoader(val_ds, batch_size=BATCH_SIZE*2, shuffle=False)
        preds_val = []
        for Xb, _ in val_loader_all:
            preds_val.append(model(Xb.to(DEVICE)).cpu().numpy())
        oof_preds_log[val_idx] = np.concatenate(preds_val)
        # test preds
        test_dl = DataLoader(PriceDataset(X_test_full), batch_size=BATCH_SIZE*2, shuffle=False)
        tpreds = []
        for Xb in test_dl:
            tpreds.append(model(Xb.to(DEVICE)).cpu().numpy())
        test_preds_log_folds[fold-1] = np.concatenate(tpreds)

    # cleanup
    del model, optimizer, train_loader, val_loader, val_loader_all, test_dl
    torch.cuda.empty_cache()
    gc.collect()

# Convert NN outputs from log-space to price
nn_oof = np.expm1(oof_preds_log)
nn_test_mean = np.expm1(test_preds_log_folds.mean(axis=0))
print("NN OOF SMAPE:", smape_np(y_train_raw, nn_oof))


# Meta features: NN OOF (raw price) + numeric features (original)
meta_train = np.column_stack([nn_oof, X_numeric_train])
meta_test  = np.column_stack([nn_test_mean, X_numeric_test])

# LightGBM (train on raw price)
lgb_params = {
    'n_estimators': 1500, 'learning_rate': 0.02, 'num_leaves': 128,
    'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 5,
    'random_state': SEED
}
lgb_clf = lgb.LGBMRegressor(**lgb_params)
# older lightgbm versions: don't pass verbose kw to fit
lgb_clf.fit(meta_train, y_train_raw)   # training on raw price

# CatBoost (train on raw price)
cb_clf = CatBoostRegressor(iterations=1200, learning_rate=0.02, depth=8, l2_leaf_reg=3, loss_function='MAE', random_seed=SEED)
cb_clf.fit(meta_train, y_train_raw)

# Build stacked (Ridge) on top of LGB + Cat predictions (OOF)
pred_l_oof = lgb_clf.predict(meta_train)
pred_c_oof = cb_clf.predict(meta_train)
stack_X = np.column_stack([pred_l_oof, pred_c_oof])
ridge = Ridge(alpha=1.0)
ridge.fit(stack_X, y_train_raw)

# Test predictions via stack
pred_l_test = lgb_clf.predict(meta_test)
pred_c_test = cb_clf.predict(meta_test)
stack_test = np.column_stack([pred_l_test, pred_c_test])
blend_test = ridge.predict(stack_test)


# Clip and smooth extremes
blend_test = np.clip(blend_test, 0.0, None)
upper = np.percentile(train_df['price'].values, 99.0)
median_price = np.median(train_df['price'].values)
blend_test = np.where(blend_test > upper * 1.5, 0.6*blend_test + 0.4*median_price, blend_test)

# Compute final OOF blended score on training set
pred_l_oof_test = lgb_clf.predict(meta_train)
pred_c_oof_test = cb_clf.predict(meta_train)
blend_oof = ridge.predict(np.column_stack([pred_l_oof_test, pred_c_oof_test]))
print("Final OOF SMAPE (stacked):", smape_np(train_df['price'].values, blend_oof))

# Save submission
submission = pd.DataFrame({"sample_id": test_df['sample_id'], "price": blend_test})
submission.to_csv("test_out_stacked_final.csv", index=False)
print("Saved test_out_stacked_final.csv")
