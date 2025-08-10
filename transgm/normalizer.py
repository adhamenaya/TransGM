import numpy as np
from KDEpy import FFTKDE
from KDEpy.bw_selection import silvermans_rule


def calculate(data0, size0, verbose=False, actual_xy=True):
    # Compute bandwidths using Silverman's rule
    bw1 = silvermans_rule(data0[:, [0]])
    bw2 = silvermans_rule(data0[:, [1]])

    # Scale the data using the computed bandwidths
    # Ensure broadcasting works correctly by dividing with a 1D array
    data_scaled = data0 / np.array([bw1, bw2])

    # Perform KDE on the scaled data
    kde = FFTKDE(bw=1).fit(data_scaled)  # Expecting shape (n_obs, dims)
    x_scaled, y_scaled = kde.evaluate((size0, size0))  # Evaluate on a 32x32 grid

    # Scale back the x values to the original domain
    x = x_scaled * np.array([bw1, bw2])

    # Adjust the y values for the scaling factor
    y = y_scaled / (bw1 * bw2)

    # Verify that the integral is 1
    dx = (x[-1, 0] - x[0, 0]) / (size0 - 1)  # Spacing in x direction
    dy = (x[-1, 1] - x[0, 1]) / (size0 - 1)  # Spacing in y direction

    integ = np.sum(y * dx * dy)
    if verbose:
        print(f"Integral of the KDE: {integ}")  # Should be close to 1
    if actual_xy:
        return x, y, integ  # x: can be 2D..etc.
    else:  # index based
        x = [(x, y) for x in range(size0) for y in range(size0)]
        return x, y, integ


def kde(df_data, size=32, verbose=False, auto_size=False, actual_xy=True):
    assert 'x' in df_data.columns, "'x' column is missing in the DataFrame"
    assert 'y' in df_data.columns, "'y' column is missing in the DataFrame"

    # Convert the selected columns of the DataFrame to a NumPy array
    data = df_data[['x', 'y']].to_numpy()

    if auto_size:
        for size in range(10, 100):
            x, y, integ2 = calculate(data, size, verbose=verbose, actual_xy=actual_xy)
            if abs(integ2 - 1) < 0.02:
                print(f"Selected size: {size}")
                return x, y, 1
        return x, y, 1  # Fallback return in case no suitable size is found
    else:
        return calculate(data, size, verbose=verbose, actual_xy=actual_xy)

#
# def grid(x_data, y_data, size=32, verbose=False):
#
