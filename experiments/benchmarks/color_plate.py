from matplotlib.colors import LinearSegmentedColormap

traj_color = '#6F6E6E'
x0_color = '#E93A01'

# background color
colors = ["#F9F9F9", "#7BCFEA"]
custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)


__all__ = ["x0_color", "traj_color", "custom_cmap"]