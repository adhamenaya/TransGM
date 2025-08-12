# from spatialgrid import SpatialGrid as sp
import spatial as sp
import numpy as np
from scipy.stats import gaussian_kde

# class Div:
#     def compute(self, grid1, grid2, weighted=False, spatial=True, neighborhood_size=3, num_bins=10, standardize_lisa=True):
#         def jsd(p, q):
#             """Calculates the Jensen-Shannon divergence."""
#
#             def kld(p, q):
#                 """Calculates the Kullback-Leibler divergence."""
#                 p = np.array(p)
#                 q = np.array(q)
#                 epsilon = 1e-10
#                 p = np.where(p == 0, epsilon, p)
#                 q = np.where(q == 0, epsilon, q)
#                 return np.sum(p * np.log2(p / q))
#
#             m = 0.5 * (p + q)
#             jsd = 0.5 * kld(p, m) + 0.5 * kld(q, m)
#             return jsd
#
#         def hist(neighborhood, num_bins, normalize=True):
#             """Calculates the histogram of a neighborhood."""
#             flat_neighborhood = neighborhood.flatten()
#
#             if normalize:
#                 flat_neighborhood = normalize_to_01(flat_neighborhood)
#
#             flat_neighborhood = flat_neighborhood[~np.isnan(flat_neighborhood)]
#             hist, _ = np.histogram(flat_neighborhood, bins=num_bins)
#             sum_hist = np.sum(hist)
#
#             if sum_hist == 0:
#                 return np.zeros(num_bins)  # Return zero vector
#
#             return hist / sum_hist
#
#
#
#         def kde(neighborhood, bandwidth=0.1):
#             """Calculates the Kernel Density Estimate (KDE) of a neighborhood."""
#             flat_neighborhood = neighborhood.flatten()
#             flat_neighborhood = flat_neighborhood[~np.isnan(flat_neighborhood)]
#
#             if len(flat_neighborhood) == 0:
#                 return np.array([])  # Return empty array if no valid data
#
#             # KDE estimation using Gaussian kernels
#             kde = gaussian_kde(flat_neighborhood, bw_method=bandwidth)
#
#             # Generate a range of values to evaluate the KDE over
#             x_values = np.linspace(np.min(flat_neighborhood), np.max(flat_neighborhood), 1000)
#
#             # Evaluate the KDE over the x_values
#             kde_values = kde(x_values)
#
#             return x_values, kde_values
#
#
#         def hist0(neighborhood):
#             P = neighborhood.flatten()
#             P /= np.sum(P)
#             return P
#
#
#         def neighbors(grid_vals, i, j, neighborhood_size):
#             """Extracts the neighborhood around a cell."""
#             half_size = neighborhood_size // 2
#             return grid_vals[
#                    max(i - half_size, 0): min(i + half_size + 1, rows),
#                    max(j - half_size, 0): min(j + half_size + 1, cols)]
#
#         def normalize_lisa_minmax(data):
#             """Normalizes LISA values using min-max scaling."""
#             min_val = np.min(data)
#             max_val = np.max(data)
#             if max_val - min_val == 0:
#                 return np.zeros_like(data)
#             return (data - min_val) / (max_val - min_val)
#         def normalize_to_01(data):
#             """Normalize a numpy array to the range [0, 1]."""
#             data_min = data.min()
#             data_max = data.max()
#             if data_max == data_min:
#                 return np.zeros_like(data)  # or np.full_like(data, fill_value) with a default
#             return (data - data_min) / (data_max - data_min)



import numpy as np

class Div:
    def jsd(self, p, q, base=2):
        """Calculates the Jensen–Shannon divergence."""
        # Ensure histograms are aligned and normalized
        p, q, common_bin = self.pad_histograms(p, q)
        p = np.array(p, dtype=float)
        q = np.array(q, dtype=float)
        p /= p.sum()
        q /= q.sum()

        # Define Kullback–Leibler divergence
        def kld(a, b):
            epsilon = 1e-10
            a = np.where(a == 0, epsilon, a)
            b = np.where(b == 0, epsilon, b)
            return np.sum(a * (np.log(a / b) / np.log(base)))

        # Jensen–Shannon divergence
        m = 0.5 * (p + q)
        jsd_value = 0.5 * kld(p, m) + 0.5 * kld(q, m)
        return jsd_value, common_bin

    def pad_histograms(self, p, q):
        max_len = max(len(p), len(q))
        p = np.pad(p,(0, max_len - len(p)), mode='constant')
        q = np.pad(q,(0, max_len - len(q)), mode='constant')
        return p, q, max_len
    def compute_hist(self,data, bins=3):
        hist,_ = np.histogram(data, bins)
        hist = hist + 1e-10
        density = hist / np.sum(hist)
        return density

    def best_bin(self, data):
        results = {}
        n = len(data)

        for b in range(int(n/2),n):
            ll = 0 # log likelihood
            bins_edge = np.linspace(np.min(data), np.max(data), b + 1)
            for i in range(n):
                data_loo = np.delete(data, i)
                p_hat = self.compute_hist(data_loo, b)

                # bin index of left-out point
                bin_index = np.digitize(data[i], bins_edge) - 1
                bin_index = np.clip(bin_index, 0, b - 1)

                prob = p_hat[bin_index]
                ll += np.log(prob)

            avg_ll = ll/n
            results[b] = avg_ll
        best_b = max(results, key=results.get)
        return best_b

    def hist(self, neighborhood):
        best_b = self.best_bin(neighborhood.flatten())
        return self.compute_hist(neighborhood, best_b), best_b

    def local_div_map(self, grid1, grid2, spatial=True, neighborhood_size=3):
        if grid1.shape != grid2.shape:
            raise ValueError("Grid shapes must match.")

        rows, cols = grid1.shape
        dissimilarity_grid = np.zeros((rows, cols))
        bins1 = np.zeros_like(grid1)
        bins2 = np.zeros_like(grid2)
        if spatial:
            for i in range(rows):
                for j in range(cols):
                    n1 = sp.get_window(grid1, i, j, neighborhood_size)
                    n2 = sp.get_window(grid2, i, j, neighborhood_size)

                    p1, b1 = self.hist(n1)
                    p2, b2 = self.hist(n2)
                    # Ensure both histograms are of the same size
                    js_divergence, common_bin = self.jsd(p1, p2)
                    dissimilarity_grid[i, j] = js_divergence
                    bins1[i, j] = common_bin
                    bins2[i, j] = common_bin

            return dissimilarity_grid, bins1, bins2
        else:
            return self.jsd(grid1, grid2), -1, -1

    def compute(self, grid1, grid2, epsilon=1e-8):
        gjsd, _ = self.jsd(grid1, grid2)
        ljsd, a, b = self.local_div_map(grid1, grid2)
        idf = -1 / np.log(gjsd + epsilon)
        ljsd_normalized = ljsd / (np.max(ljsd) + epsilon)
        return np.mean(ljsd_normalized * idf)