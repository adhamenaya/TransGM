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
   - The **Jensen–Shannon Divergence (JSD)** is computed between corresponding patches in the two cities.
   - We average these local divergences to form the **overall divergence score** for each POI type.

---

### **2. Source Model Training**
We extend the **gravity model** to incorporate POI-based attractiveness:

<img alt="Gravity Model" src="/res/fig1.png" width="800"/>

- \(O_i\) – trip potential from origin cell `i`  
- \(\lambda_k\) – attraction weight for POI type `k`  
- \(d_{ij}\) – distance between cells `i` and `j`  
- \(\alpha, \beta\) – trip generation and distance decay parameters

**Estimation:**
- The model is linearised via log-transform and trained using **ridge regression** (L2-regularised) to stabilise coefficients.
- Optimisation uses **L-BFGS-B** to respect physical constraints (e.g., non-negative weights).

---

### **3. Adaptive Transfer to Target City**
We initialise the target city model using parameters from the source city, then **adapt** them using a small amount of target data:

<img alt="Adaptive Transfer" src="/res/fig2.png" width="800"/>

- **Prediction Loss**: Ensures fit to target city observations.  
- **Domain Adaptation Penalty**: Restricts changes to \(\lambda\) based on POI divergence:
  - If features are similar (low divergence), changes are penalised more heavily.
  - If features are dissimilar (high divergence), adaptation is allowed more freedom.

---

## Workflow

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

