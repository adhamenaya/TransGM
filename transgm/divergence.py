# -----------------------------------------------------------------------------
# Author: Adham Enaya
# PhD Supervisors: Prof. Chen Zhong and Prof. Micheal Batty
# Affiliation: Centre for Advanced Spatial Analysis, UCL
# Project Title: TransGM: Transferable Gravity Models for Adaptive Urban Policy
# Funding: European Research Council (ERC) under the EU Horizon 2020 program (No. 949670)
# Date: 2025-08-10
# -----------------------------------------------------------------------------

import spatial as sp
import numpy as np


class SpatialDivergence:
    def jsd(self, p, q):
        """Calculates the Jensen-Shannon divergence."""
        # First make sure both distributions are of the same size
        p, q, common_bin = self.pad_histograms(p, q)

        def kld(p, q):
            """Calculates the Kullback-Leibler divergence."""
            p = np.array(p)
            q = np.array(q)
            epsilon = 1e-10
            p = np.where(p == 0, epsilon, p)
            q = np.where(q == 0, epsilon, q)
            return np.sum(p * np.log2(p / q))

        m = 0.5 * (p + q)
        jsd = 0.5 * kld(p, m) + 0.5 * kld(q, m)
        return jsd, common_bin

    def pad_histograms(self, p, q):
        """Pads histograms to the same length."""
        max_len = max(len(p), len(q))
        p = np.pad(p, (0, max_len - len(p)), mode='constant')
        q = np.pad(q, (0, max_len - len(q)), mode='constant')
        return p, q, max_len

    def compute_hist(self, data, bins=3):
        """Computes normalized histogram density."""
        hist, _ = np.histogram(data, bins)
        hist = hist + 1e-10
        density = hist / np.sum(hist)
        return density

    def best_bin(self, data):
        """Finds the optimal number of bins for histogram using leave-one-out log likelihood."""
        results = {}
        n = len(data)

        for b in range(int(n / 2), n):
            ll = 0  # log likelihood
            bins_edge = np.linspace(np.min(data), np.max(data), b + 1)
            for i in range(n):
                data_loo = np.delete(data, i)
                p_hat = self.compute_hist(data_loo, b)

                # bin index of left-out point
                bin_index = np.digitize(data[i], bins_edge) - 1
                bin_index = np.clip(bin_index, 0, b - 1)

                prob = p_hat[bin_index]
                ll += np.log(prob)

            avg_ll = ll / n
            results[b] = avg_ll

        best_b = max(results, key=results.get)
        return best_b

    def hist(self, neighborhood):
        """Computes histogram and best bin for given neighborhood."""
        best_b = self.best_bin(neighborhood.flatten())
        return self.compute_hist(neighborhood, best_b), best_b

    def compute(self, grid1, grid2, spatial=True, neighborhood_size=3):
        """Computes spatial divergence between two grids."""
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