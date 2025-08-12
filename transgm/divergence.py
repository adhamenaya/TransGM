from spatialgrid import SpatialGrid as sp
import numpy as np

class Divergence:
    def jsd(self, p, q, base=2):
        p, q, common_bin = self.pad_histograms(p, q)
        p = np.array(p, dtype=float)
        q = np.array(q, dtype=float)
        p /= p.sum()
        q /= q.sum()

        def kld(a, b):
            epsilon = 1e-10
            a = np.where(a == 0, epsilon, a)
            b = np.where(b == 0, epsilon, b)
            return np.sum(a * (np.log(a / b) / np.log(base)))

        m = 0.5 * (p + q)
        jsd_value = 0.5 * kld(p, m) + 0.5 * kld(q, m)
        return jsd_value, common_bin

    def pad_histograms(self, p, q):
        max_len = max(len(p), len(q))
        p = np.pad(p, (0, max_len - len(p)), mode='constant')
        q = np.pad(q, (0, max_len - len(q)), mode='constant')
        return p, q, max_len

    def compute_hist(self, data, bins=3):
        hist, _ = np.histogram(data, bins)
        hist = hist + 1e-10
        density = hist / np.sum(hist)
        return density

    def best_bin(self, data):
        results = {}
        n = len(data)

        for b in range(int(n/2), n):
            ll = 0
            bins_edge = np.linspace(np.min(data), np.max(data), b + 1)
            for i in range(n):
                data_loo = np.delete(data, i)
                p_hat = self.compute_hist(data_loo, b)
                bin_index = np.digitize(data[i], bins_edge) - 1
                bin_index = np.clip(bin_index, 0, b - 1)
                prob = p_hat[bin_index]
                ll += np.log(prob)
            avg_ll = ll / n
            results[b] = avg_ll
        best_b = max(results, key=results.get)
        return best_b

    def hist(self, neighborhood):
        best_b = self.best_bin(neighborhood.flatten())
        return self.compute_hist(neighborhood, best_b), best_b

    # Fallback if sp.get_window fails or unavailable
    def get_window(self, grid, i, j, size):
        half = size // 2
        row_min = max(i - half, 0)
        row_max = min(i + half + 1, grid.shape[0])
        col_min = max(j - half, 0)
        col_max = min(j + half + 1, grid.shape[1])
        return grid[row_min:row_max, col_min:col_max]

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
                    # Try to use sp.get_window else fallback
                    try:
                        n1 = sp.get_window(grid1, i, j, neighborhood_size)
                        n2 = sp.get_window(grid2, i, j, neighborhood_size)
                    except Exception:
                        n1 = self.get_window(grid1, i, j, neighborhood_size)
                        n2 = self.get_window(grid2, i, j, neighborhood_size)

                    p1, b1 = self.hist(n1)
                    p2, b2 = self.hist(n2)
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