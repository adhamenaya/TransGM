# TransGM: Transferable Gravity Models

This repository implements **TransGM**, a three-step framework for transferring spatial interaction models between cities by quantifying **land-use feature divergence** and applying **adaptive regularisation**.

---

## Methodology

TransGM operates in **three stages**:

### **1. Spatial Divergence Encoding**
We first measure the differences in **Point-of-Interest (POI)** and land-use distributions between two cities.

1. **Spatial Grid Normalisation**  
   - Each city’s POIs are projected onto a fixed-size grid (e.g., `8×8`) using adaptive bivariate Gaussian kernel density estimation.  
   - This ensures feature values are **probabilities** rather than raw counts, making them comparable across cities of different sizes.

2. **Local Spatial Divergence**  
   - For each grid cell, we form a **patch** containing the cell and its neighbours.  
   - Patches are converted into histograms over POI types, with bin widths optimised via **leave-one-cell-out cross-validation**.  
   - The **Jensen–Shannon Divergence (JSD)** is computed between corresponding patches in the two cities:  
     \[
     \mathrm{JSD}(P_i \parallel Q_i) = \frac{1}{2} KL(P_i \parallel M_i) + \frac{1}{2} KL(Q_i \parallel M_i)
     \]
   - We average these local divergences to form the **overall divergence score** for each POI type.

---

### **2. Source Model Training**
We extend the **gravity model** to incorporate POI-based attractiveness:

\[
T_{ij} = O_i^\alpha \cdot \left( \sum_{k=1}^n \lambda_k \cdot \mathrm{POI}_{j,k} \right) \cdot d_{ij}^{-\beta}
\]

- \(O_i\) – trip potential from origin cell `i`  
- \(\lambda_k\) – attraction weight for POI type `k`  
- \(d_{ij}\) – distance between cells `i` and `j`  
- \(\alpha, \beta\) – trip generation and distance decay parameters

**Estimation:**
- The model is linearised via log-transform and trained using **ridge regression** (L2-regularised) to stabilise coefficients.
- Optimisation uses **L-BFGS-B** to respect physical constraints (e.g., non-negative weights).

---

### **3. Adaptive Transfer to Target City**
We initialise the target city model using parameters from the source city, then **adapt** them using a small amount of target data.

The transfer objective:

\[
L_{\mathrm{transfer}} =
\underbrace{\mathrm{MSE}(y_t, X_t \lambda_t)}_{\text{Prediction Loss}} +
\underbrace{K \cdot f_{\mathrm{DA}}(\delta(\mathrm{POI})) \cdot \lVert \lambda_t - \lambda_s \rVert^2}_{\text{Domain Adaptation Penalty}}
\]

- **Prediction Loss**: Ensures fit to target city observations.  
- **Domain Adaptation Penalty**: Restricts changes to \(\lambda\) based on POI divergence:
  - If features are similar (low divergence), changes are penalised more heavily.
  - If features are dissimilar (high divergence), adaptation is allowed more freedom.

---

## Workflow in Code

### **1. Data Preparation**
```python
from oddata import ODData

od_coventry = ODData(spacing_km=3, size=(8, 8))
od_coventry.load_data('coventry')
od_coventry.preprocess()

od_birmingham = ODData(spacing_km=4, size=(8, 8))
od_birmingham.load_data('Birmingham')
od_birmingham.preprocess()
````

### **2. Spatial Divergence Computation**

```python
from divergence import Div

div = Div()
results = {}
for i, category in enumerate(sorted(od_coventry.categories)):
    grid_cov = od_coventry.destinations[:, :, i]
    grid_bir = od_birmingham.destinations[:, :, i]
    results[category] = div.compute(grid_cov, grid_bir)
```

### **3. Model Training & Transfer**

```python
from dataset import DataSet
from transgm import TransGM

tgm = TransGM()
tgm.load_data(DataSet('birmingham'))
tgm.load_data(DataSet('coventry'))

# Model 1: Train source model
model = tgm.fit('birmingham', serial=1)

# Model 2: Transfer to target city
model.transfer('coventry')

# Model 3: Adapt with divergence-aware regularisation
model.load_divs(results).adapt('coventry')
```

---

## License

MIT License – see `LICENSE` for details.

```

---

This keeps your README in **GitHub-friendly format** but now fully incorporates your methodology’s key concepts:  
- **Spatial grid normalisation**  
- **Local patch-based divergence with JSD**  
- **Gravity model with ridge regression**  
- **Adaptive divergence-aware transfer**  

If you want, I can also **add a figure diagramming the three-step TransGM process** so the README is visually aligned with the paper. That would make it much easier for readers to understand at a glance.
```
