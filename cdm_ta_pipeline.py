import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml, load_breast_cancer, load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FastICA
from sklearn.feature_selection import SelectKBest, f_regression, f_classif, RFE
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, silhouette_score
import time

def standardize(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler

def _offdiag_stats(M):
    mask = ~np.eye(M.shape[0], dtype=bool)
    vals = np.abs(M[mask])
    return float(vals.max()), float(vals.mean())

def collinearity_report(X, feature_names=None, title="", plot=False):
    Sigma = np.cov(X, rowvar=False)
    Corr  = np.corrcoef(X, rowvar=False)

    s = np.linalg.svd(X, full_matrices=False, compute_uv=False)
    smin = s[-1] if s[-1] > 0 else 1e-12
    kappa = float(s[0] / smin)

    max_abs_r, mean_abs_r = _offdiag_stats(Corr)

    t = f" [{title}]" if title else ""
    print(f"\n--- Collinearity Report{t} ---")
    print(f"Features (d): {X.shape[1]} | Samples (n): {X.shape[0]}")
    print(f"Condition Number (X): {kappa:,.3f}")
    print(f"Max |corr| off-diagonal: {max_abs_r:,.3f}")
    print(f"Mean |corr| off-diagonal: {mean_abs_r:,.3f}")

    if plot:
        plt.figure(figsize=(6,5))
        im = plt.imshow(Corr, interpolation='nearest')
        plt.title(f"Correlation Heatmap{t}")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        if feature_names is not None and len(feature_names) == Corr.shape[0] and Corr.shape[0] <= 30:
            plt.xticks(range(len(feature_names)), feature_names, rotation=90)
            plt.yticks(range(len(feature_names)), feature_names)
        else:
            plt.xticks([]); plt.yticks([])
        plt.tight_layout()
        plt.show()

    return {
        "covariance": Sigma,
        "correlation": Corr,
        "condition_number": kappa,
        "max_abs_corr_offdiag": max_abs_r,
        "mean_abs_corr_offdiag": mean_abs_r,
    }

def pca_fit_transform(X, tau=0.90, title=None, plot=True, random_state=0):
    pca_full = PCA(random_state=random_state).fit(X)
    evr = pca_full.explained_variance_ratio_
    evr_cum = np.cumsum(evr)
    k = int(np.searchsorted(evr_cum, tau) + 1)

    if plot:
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].plot(np.arange(1, len(evr)+1), evr, marker='o')
        ax[0].set_title(f"Scree Plot{'' if not title else ' - ' + title}")
        ax[0].set_xlabel("Component #"); ax[0].set_ylabel("Explained Variance Ratio")
        ax[1].plot(np.arange(1, len(evr_cum)+1), evr_cum, marker='o')
        ax[1].axhline(tau, ls='--'); ax[1].axvline(k, ls='--')
        ax[1].set_ylim(0, 1.01)
        ax[1].set_title(f"Cumulative EVR (k={k}, τ={tau})")
        ax[1].set_xlabel("Component #"); ax[1].set_ylabel("Cumulative EVR")
        plt.tight_layout(); plt.show()

    pca_k = PCA(n_components=k, random_state=random_state).fit(X)
    Z = pca_k.transform(X)
    return {"Z": Z, "k": k, "pca": pca_k, "evr": evr, "evr_cum": evr_cum}

def ica_fit_transform(X, n_components, random_state=0):
    ica = FastICA(n_components=n_components, random_state=random_state, max_iter=1000)
    Z = ica.fit_transform(X)
    return {"Z": Z, "ica": ica}

def svd_transform(X, tau=0.90, title=None, plot=True):
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    power = s**2
    power_cum = np.cumsum(power) / power.sum()
    k = int(np.searchsorted(power_cum, tau) + 1)

    if plot:
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].plot(np.arange(1, len(s)+1), power / power.sum(), marker='o')
        ax[0].set_title(f"SVD Scree{'' if not title else ' - ' + title}")
        ax[0].set_xlabel("Singular Value #"); ax[0].set_ylabel("Explained (σ² ratio)")
        ax[1].plot(np.arange(1, len(power_cum)+1), power_cum, marker='o')
        ax[1].axhline(tau, ls='--'); ax[1].axvline(k, ls='--')
        ax[1].set_ylim(0, 1.01)
        ax[1].set_title(f"Cumulative (k={k}, τ={tau})")
        ax[1].set_xlabel("Singular Value #"); ax[1].set_ylabel("Cumulative")
        plt.tight_layout(); plt.show()

    Z = X @ Vt[:k].T
    return {"Z": Z, "k": k, "s": s, "Vt": Vt}

def sgd_convergence(Xtr, ytr, Xte, yte, n_epochs=120, alpha=1e-4, random_state=0):
    sgd = SGDRegressor(penalty="l2", alpha=alpha, max_iter=1, tol=None,
                       warm_start=True, random_state=random_state)
    mses = []
    for _ in range(n_epochs):
        sgd.fit(Xtr, ytr)
        mses.append(mean_squared_error(yte, sgd.predict(Xte)))
    return np.array(mses)

def kmeans_eval(X, n_clusters=3, title=""):
    t0 = time.time()
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit(X)
    t1 = time.time()
    sil = silhouette_score(X, km.labels_) if len(np.unique(km.labels_)) > 1 else np.nan
    print(f"[KMeans{title}] inertia={km.inertia_:.2f}, iter={km.n_iter_}, time={t1 - t0:.3f}s, silhouette={sil:.3f}")
    return {"model": km, "silhouette": sil}

# ---------- Step 1 ----------
boston = fetch_openml(name="boston", version=1, as_frame=True)
X_reg = boston.data.values
y_reg = boston.target.values.astype(float)
feature_names_reg = list(boston.feature_names) if hasattr(boston, "feature_names") else list(boston.data.columns)
X_reg_scaled, scaler_reg = standardize(X_reg)

breast = load_breast_cancer(as_frame=True)
X_clf = breast.data.values
y_clf = breast.target.values
feature_names_clf = list(breast.feature_names)
target_names_clf = list(breast.target_names)
X_clf_scaled, scaler_clf = standardize(X_clf)

iris = load_iris(as_frame=True)
X_clu = iris.data.values
y_clu = iris.target.values
feature_names_clu = list(iris.feature_names)
X_clu_scaled, scaler_clu = standardize(X_clu)

# ---------- Step 2 ----------
rep_reg = collinearity_report(X_reg_scaled, feature_names_reg, title="Regression (Boston Housing)", plot=True)
rep_clf = collinearity_report(X_clf_scaled, feature_names_clf, title="Classification (Breast Cancer)", plot=True)
rep_clu = collinearity_report(X_clu_scaled, feature_names_clu, title="Clustering (Iris)", plot=True)

# ---------- Step 3 ----------
TAU = 0.90
pca_reg = pca_fit_transform(X_reg_scaled, tau=TAU, title="Boston Housing", plot=True)
ica_reg = ica_fit_transform(X_reg_scaled, n_components=pca_reg["k"])
svd_reg = svd_transform(X_reg_scaled, tau=TAU, title="Boston Housing", plot=True)
print(f"[Boston] k_PCA={pca_reg['k']}, k_SVD={svd_reg['k']}")

pca_clf = pca_fit_transform(X_clf_scaled, tau=TAU, title="Breast Cancer", plot=True)
ica_clf = ica_fit_transform(X_clf_scaled, n_components=pca_clf["k"])
svd_clf = svd_transform(X_clf_scaled, tau=TAU, title="Breast Cancer", plot=True)
print(f"[BreastCancer] k_PCA={pca_clf['k']}, k_SVD={svd_clf['k']}")

pca_clu = pca_fit_transform(X_clu_scaled, tau=TAU, title="Iris", plot=True)
ica_clu = ica_fit_transform(X_clu_scaled, n_components=pca_clu["k"])
svd_clu = svd_transform(X_clu_scaled, tau=TAU, title="Iris", plot=True)
print(f"[Iris] k_PCA={pca_clu['k']}, k_SVD={svd_clu['k']}")

# ---------- Step 4 ----------
k_reg = pca_reg["k"]
skb_reg = SelectKBest(score_func=f_regression, k=k_reg).fit(X_reg_scaled, y_reg)
idx_reg_skb = skb_reg.get_support(indices=True)
names_reg_skb = [feature_names_reg[i] for i in idx_reg_skb]
print(f"[Boston][SelectKBest - f_regression] k={k_reg}")
print("indices:", idx_reg_skb)
print("features:", names_reg_skb)

rfe_reg = RFE(estimator=LinearRegression(), n_features_to_select=k_reg, step=1).fit(X_reg_scaled, y_reg)
idx_reg_rfe = rfe_reg.get_support(indices=True)
names_reg_rfe = [feature_names_reg[i] for i in idx_reg_rfe]
print(f"[Boston][RFE - LinearRegression] k={k_reg}")
print("indices:", idx_reg_rfe)
print("features:", names_reg_rfe)

k_clf = pca_clf["k"]
skb_clf = SelectKBest(score_func=f_classif, k=k_clf).fit(X_clf_scaled, y_clf)
idx_clf_skb = skb_clf.get_support(indices=True)
names_clf_skb = [feature_names_clf[i] for i in idx_clf_skb]
print(f"[BreastCancer][SelectKBest - f_classif] k={k_clf}")
print("indices:", idx_clf_skb)
print("features:", names_clf_skb)

logreg = LogisticRegression(max_iter=1000, solver="liblinear")
rfe_clf = RFE(estimator=logreg, n_features_to_select=k_clf, step=1).fit(X_clf_scaled, y_clf)
idx_clf_rfe = rfe_clf.get_support(indices=True)
names_clf_rfe = [feature_names_clf[i] for i in idx_clf_rfe]
print(f"[BreastCancer][RFE - LogisticRegression] k={k_clf}")
print("indices:", idx_clf_rfe)
print("features:", names_clf_rfe)

# ---------- Step 5 ----------
n_reg = X_reg_scaled.shape[0]
idx_reg = np.arange(n_reg)
idx_tr_reg, idx_te_reg = train_test_split(idx_reg, test_size=0.2, random_state=0, shuffle=True)

Xtr_reg_main, Xte_reg_main = X_reg_scaled[idx_tr_reg], X_reg_scaled[idx_te_reg]
Z_reg = pca_reg["Z"]; Ztr_reg, Zte_reg = Z_reg[idx_tr_reg], Z_reg[idx_te_reg]
Xtr_reg_skb,  Xte_reg_skb  = Xtr_reg_main[:, idx_reg_skb], Xte_reg_main[:, idx_reg_skb]
Xtr_reg_rfe,  Xte_reg_rfe  = Xtr_reg_main[:, idx_reg_rfe], Xte_reg_main[:, idx_reg_rfe]

ytr_reg, yte_reg = y_reg[idx_tr_reg], y_reg[idx_te_reg]

def fit_eval_linear(Xtr, Xte, ytr, yte):
    m = LinearRegression().fit(Xtr, ytr)
    yhat = m.predict(Xte)
    return mean_squared_error(yte, yhat), r2_score(yte, yhat), float(np.linalg.norm(m.coef_))

mse_main, r2_main, nb_main   = fit_eval_linear(Xtr_reg_main, Xte_reg_main, ytr_reg, yte_reg)
mse_pca,  r2_pca,  nb_pca    = fit_eval_linear(Ztr_reg,       Zte_reg,       ytr_reg, yte_reg)
mse_skb,  r2_skb,  nb_skb    = fit_eval_linear(Xtr_reg_skb,   Xte_reg_skb,   ytr_reg, yte_reg)
mse_rfe,  r2_rfe,  nb_rfe    = fit_eval_linear(Xtr_reg_rfe,   Xte_reg_rfe,   ytr_reg, yte_reg)

print("\n[Regression: Linear | aligned split]")
print(f"Main            -> MSE={mse_main:.4f}, R2={r2_main:.4f}, ||beta||={nb_main:.3f}")
print(f"PCA(k={pca_reg['k']})   -> MSE={mse_pca:.4f}, R2={r2_pca:.4f}, ||beta||={nb_pca:.3f}")
print(f"SelectKBest(k={k_reg})-> MSE={mse_skb:.4f}, R2={r2_skb:.4f}, ||beta||={nb_skb:.3f}")
print(f"RFE(k={k_reg})        -> MSE={mse_rfe:.4f}, R2={r2_rfe:.4f}, ||beta||={nb_rfe:.3f}")

epochs = 120
mse_path_main = sgd_convergence(Xtr_reg_main, ytr_reg, Xte_reg_main, yte_reg, n_epochs=epochs)
mse_path_pca  = sgd_convergence(Ztr_reg,       ytr_reg, Zte_reg,       yte_reg, n_epochs=epochs)
plt.figure(figsize=(6,4))
plt.plot(np.arange(1, epochs+1), mse_path_main, label="Main")
plt.plot(np.arange(1, epochs+1), mse_path_pca,  label=f"PCA (k={pca_reg['k']})")
plt.xlabel("Epoch"); plt.ylabel("MSE on Test"); plt.title("SGD Convergence (Boston) - aligned split")
plt.legend(); plt.tight_layout(); plt.show()

print("\n[Clustering: Iris, k=3]")
km_main = kmeans_eval(X_clu_scaled, n_clusters=3, title=" - Main")
km_pca  = kmeans_eval(pca_clu["Z"], n_clusters=3, title=f" - PCA(k={pca_clu['k']})")

n_clf = X_clf_scaled.shape[0]
idx_clf = np.arange(n_clf)
idx_tr_clf, idx_te_clf = train_test_split(idx_clf, test_size=0.2, random_state=0, shuffle=True)

Xtr_clf_main, Xte_clf_main = X_clf_scaled[idx_tr_clf], X_clf_scaled[idx_te_clf]
Z_clf = pca_clf["Z"]; Ztr_clf, Zte_clf = Z_clf[idx_tr_clf], Z_clf[idx_te_clf]
Xtr_clf_skb,  Xte_clf_skb  = Xtr_clf_main[:, idx_clf_skb], Xte_clf_main[:, idx_clf_skb]
Xtr_clf_rfe,  Xte_clf_rfe  = Xtr_clf_main[:, idx_clf_rfe], Xte_clf_main[:, idx_clf_rfe]
ytr_clf, yte_clf = y_clf[idx_tr_clf], y_clf[idx_te_clf]

def acc_of(model, Xtr, Xte, ytr, yte):
    m = model.fit(Xtr, ytr)
    return accuracy_score(yte, m.predict(Xte))

knn = KNeighborsClassifier(n_neighbors=5)
acc_knn_main = acc_of(knn, Xtr_clf_main, Xte_clf_main, ytr_clf, yte_clf)
acc_knn_pca  = acc_of(KNeighborsClassifier(n_neighbors=5), Ztr_clf, Zte_clf, ytr_clf, yte_clf)
acc_knn_skb  = acc_of(KNeighborsClassifier(n_neighbors=5), Xtr_clf_skb, Xte_clf_skb, ytr_clf, yte_clf)
acc_knn_rfe  = acc_of(KNeighborsClassifier(n_neighbors=5), Xtr_clf_rfe, Xte_clf_rfe, ytr_clf, yte_clf)

print("\n[Classification: KNN(k=5) | aligned split]")
print(f"Main              -> Acc={acc_knn_main:.4f}")
print(f"PCA(k={pca_clf['k']})   -> Acc={acc_knn_pca:.4f}")
print(f"SelectKBest(k={k_clf})-> Acc={acc_knn_skb:.4f}")
print(f"RFE(k={k_clf})        -> Acc={acc_knn_rfe:.4f}")

rf = RandomForestClassifier(n_estimators=200, random_state=0)
acc_rf_main = acc_of(rf, Xtr_clf_main, Xte_clf_main, ytr_clf, yte_clf)
acc_rf_pca  = acc_of(RandomForestClassifier(n_estimators=200, random_state=0), Ztr_clf, Zte_clf, ytr_clf, yte_clf)
acc_rf_skb  = acc_of(RandomForestClassifier(n_estimators=200, random_state=0), Xtr_clf_skb, Xte_clf_skb, ytr_clf, yte_clf)
acc_rf_rfe  = acc_of(RandomForestClassifier(n_estimators=200, random_state=0), Xtr_clf_rfe, Xte_clf_rfe, ytr_clf, yte_clf)

print("\n[Classification: RandomForest | aligned split]")
print(f"Main              -> Acc={acc_rf_main:.4f}")
print(f"PCA(k={pca_clf['k']})   -> Acc={acc_rf_pca:.4f}")
print(f"SelectKBest(k={k_clf})-> Acc={acc_rf_skb:.4f}")
print(f"RFE(k={k_clf})        -> Acc={acc_rf_rfe:.4f}")

# ---------- Step 6 ----------
reg_df = pd.DataFrame({
    "space": ["Main", f"PCA(k={pca_reg['k']})", f"SelectKBest(k={k_reg})", f"RFE(k={k_reg})"],
    "MSE":   [mse_main, mse_pca, mse_skb, mse_rfe],
    "R2":    [r2_main,  r2_pca,  r2_skb,  r2_rfe]
})

clu_df = pd.DataFrame({
    "space": ["Main", f"PCA(k={pca_clu['k']})"],
    "inertia":    [km_main["model"].inertia_, km_pca["model"].inertia_],
    "silhouette": [km_main["silhouette"],     km_pca["silhouette"]],
})

clf_df = pd.DataFrame({
    "method": ["KNN","KNN","KNN","KNN","RandomForest","RandomForest","RandomForest","RandomForest"],
    "space":  ["Main", f"PCA(k={pca_clf['k']})", f"SelectKBest(k={k_clf})", f"RFE(k={k_clf})",
               "Main", f"PCA(k={pca_clf['k']})", f"SelectKBest(k={k_clf})", f"RFE(k={k_clf})"],
    "accuracy": [acc_knn_main, acc_knn_pca, acc_knn_skb, acc_knn_rfe,
                 acc_rf_main,  acc_rf_pca,  acc_rf_skb,  acc_rf_rfe]
})

print("\n=== Regression Summary ===")
print(reg_df.round(4).to_string(index=False))
print("\n=== Clustering (Iris, k=3) Summary ===")
print(clu_df.round(4).to_string(index=False))
print("\n=== Classification (Breast Cancer) Summary ===")
print(clf_df.round(4).to_string(index=False))

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].bar(reg_df["space"], reg_df["MSE"])
ax[0].set_title("Regression MSE (Boston)")
ax[0].set_ylabel("MSE")
ax[0].tick_params(axis='x', rotation=20)
ax[1].bar(reg_df["space"], reg_df["R2"])
ax[1].set_title("Regression R² (Boston)")
ax[1].set_ylabel("R²")
ax[1].tick_params(axis='x', rotation=20)
plt.tight_layout(); plt.show()

fig, ax = plt.subplots(1, 2, figsize=(9, 4))
ax[0].bar(clu_df["space"], clu_df["inertia"])
ax[0].set_title("KMeans Inertia (Iris, k=3)")
ax[0].set_ylabel("Inertia (↓ lower is better)")
ax[1].bar(clu_df["space"], clu_df["silhouette"])
ax[1].set_title("KMeans Silhouette (Iris, k=3)")
ax[1].set_ylabel("Silhouette (↑ higher is better)")
plt.tight_layout(); plt.show()

labels = ["KNN-Main","KNN-PCA",f"KNN-SelectK(k={k_clf})",f"KNN-RFE(k={k_clf})",
          "RF-Main","RF-PCA",f"RF-SelectK(k={k_clf})",f"RF-RFE(k={k_clf})"]
acc_vals = [acc_knn_main, acc_knn_pca, acc_knn_skb, acc_knn_rfe,
            acc_rf_main,  acc_rf_pca,  acc_rf_skb,  acc_rf_rfe]
plt.figure(figsize=(11,4))
plt.bar(np.arange(len(labels)), acc_vals)
plt.xticks(np.arange(len(labels)), labels, rotation=25, ha='right')
plt.ylabel("Accuracy")
plt.title("Classification Accuracy (Breast Cancer) - Main vs PCA vs SelectKBest vs RFE")
plt.tight_layout(); plt.show()
