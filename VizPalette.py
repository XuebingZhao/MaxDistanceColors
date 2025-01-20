# Used to generate color palettes for visualizations.
# By simulating the color points repelling each other, to maximize the delta E between them.

from time import perf_counter as timer

import numpy as np
from scipy.spatial import cKDTree, Delaunay
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
import colour

colour.set_domain_range_scale("1")
RGB_SPACES = [
    'ACES2065-1', 'ACEScc', 'ACEScct', 'ACEScg', 'ACESproxy',
    'ARRI Wide Gamut 3', 'ARRI Wide Gamut 4',
    'Adobe RGB (1998)', 'Adobe Wide Gamut RGB', 'Apple RGB',
    'Best RGB', 'Beta RGB', 'Blackmagic Wide Gamut',
    'CIE RGB', 'Cinema Gamut', 'ColorMatch RGB', 'DCDM XYZ',
    'DCI-P3', 'DCI-P3-P',
    'DJI D-Gamut', 'DRAGONcolor', 'DRAGONcolor2', 'DaVinci Wide Gamut',
    'Display P3',
    'Don RGB 4', 'EBU Tech. 3213-E', 'ECI RGB v2', 'ERIMM RGB',
    'Ekta Space PS 5', 'F-Gamut', 'FilmLight E-Gamut',
    'ITU-R BT.2020', 'ITU-R BT.470 - 525', 'ITU-R BT.470 - 625', 'ITU-R BT.709',
    'ITU-T H.273 - 22 Unspecified', 'ITU-T H.273 - Generic Film',
    'Max RGB', 'N-Gamut',
    'NTSC (1953)', 'NTSC (1987)', 'P3-D65', 'PLASA ANSI E1.54', 'Pal/Secam',
    'ProPhoto RGB', 'Protune Native',
    'REDWideGamutRGB', 'REDcolor', 'REDcolor2', 'REDcolor3', 'REDcolor4',
    'RIMM RGB', 'ROMM RGB', 'Russell RGB',
    'S-Gamut', 'S-Gamut3', 'S-Gamut3.Cine',
    'SMPTE 240M', 'SMPTE C',
    'Sharp RGB', 'V-Gamut', 'Venice S-Gamut3', 'Venice S-Gamut3.Cine', 'Xtreme RGB',
    'aces', 'adobe1998', 'prophoto', 'sRGB'
]
CAM_SPACES = [
    'CAM02-LCD', 'CAM02-SCD', 'CAM02-UCS', 'CAM16-LCD', 'CAM16-SCD', 'CAM16-UCS',
    'CAM02LCD', 'CAM02SCD', 'CAM02UCS', 'CAM16LCD', 'CAM16SCD', 'CAM16UCS'
]
DE_SPACES_MAP = {
    'CIE 1976': 'CIE Lab',
    'CIE 1994': 'CIE Lab',
    'CIE 2000': 'CIE Lab',
    'CMC': 'CIE Lab',
    'DIN99': 'CIE Lab',
    'ITP': 'ICtCp',
    'CAM02-LCD': 'CAM02LCD',
    'CAM02-SCD': 'CAM02SCD',
    'CAM02-UCS': 'CAM02UCS',
    'CAM16-LCD': 'CAM16LCD',
    'CAM16-SCD': 'CAM16SCD',
    'CAM16-UCS': 'CAM16UCS'
}
# 根据颜色空间设置标签
LABEL_DICT = {
    'CIE XYZ': ["X", "Y", "Z"],
    'CIE xyY': ["x", "y", "Y"],
    'CIE Lab': ["L*", "a*", "b*"],
    'CIE Luv': ["L*", "u*", "v*"],
    'CIE 1976 UCS': ["u'", "v'", "L"],
    'CIE UCS': ["U", "V", "W"],
    'CIE 1960 UCS': ["u", "v", "V"],
    'CIE UVW': ["U*", "V*", "W*"],
    'Hunter Lab': ["L", "a", "b"],
    'Hunter Rdab': ["Rd", "a", "b"],
    'DIN99': ["L_{99}", "a_{99}", "b_{99}"],
    'ICaCb': ["I", "C_A", "C_Bb"],
    'IgPgTg': ["I_G", "P_G", "T_G"],
    'IPT': ["I", "P", "T"],
    'IPT Ragoo 2021': ["I", "P", "T"],
    'hdr-CIELAB': ["L_{hdr}", "a_{hdr}", "b_{hdr}"],
    'hdr-IPT': ["I_{hdr}", "P_{hdr}", "T_{hdr}"],
    'Oklab': ["L", "a", "b"],
    'OSA UCS': ["L", "j", "g"],
    'ProLab': ["L", "a", "b"],
    'Yrg': ["Y", "r", "g"],
    'Jzazaz': ["J_z", "a_z", "b_z"],
    'Izazbz': ["I_z", "a_z", "b_z"],
    'YCbCr': ["Y'", "Cb", "Cr"],
    'YcCbcCrc': ["Y'", "Cbc'", "Crc'"],
    'YCoCg': ["Y", "Co", "Cg"],
    'ICtCp': ["I", "C_T", "C_P"],
    'HSV': ["Hue", "Saturation", "Value"],
    'HSL': ["Hue", "Saturation", "Lightness"],
    'HCL': ["Hue", "Chroma", "Lightness"],
    'CMY': ["Cyan", "Magenta", "Yellow"],
    'IHLS': ["H", "Y", "S"],
}
SIM_PARAMS = {
    'low': {
        'dt': 0.001,
        't_end': 500,
        'steps': 20000,
        't_tol': 0.0005,
        'skip': 2,
        'damping': 0.95
    },
    'medium': {
        'dt': 0.0001,
        't_end': 1000,
        'steps': 100000,
        't_tol': 0.0002,
        'skip': 3,
        'damping': 0.99
    },
    'high': {
        'dt': 0.0001,
        't_end': 2000,
        'steps': 300000,
        't_tol': 0.00005,
        'skip': 5,
        'damping': 0.998
    }
}


def auto_convert(a, source='sRGB', target='CIE XYZ'):
    """
    Automatic color space conversion.
    :param a: Array of colors to convert.
    :param source: input color space.
    :param target: output color space.
    :return: Converted array.
    """
    if source == target:
        return a

    # Remove CAM space's extra '-'
    if source in CAM_SPACES:
        source = source.replace('-', '')
    if target in CAM_SPACES:
        target = target.replace('-', '')

    # Convert between RGB spaces
    if source in RGB_SPACES and target in RGB_SPACES:
        return colour.RGB_to_RGB(a, source, target, apply_cctf_encoding=True, apply_cctf_decoding=True)

    # Convert from RGB to other spaces
    if source in RGB_SPACES:
        XYZ = colour.RGB_to_XYZ(a, source, apply_cctf_decoding=True)
        return colour.convert(XYZ, 'CIE XYZ', target)

    # Convert from other spaces to RGB
    if target in RGB_SPACES:
        XYZ = colour.convert(a, source, 'CIE XYZ')
        return colour.XYZ_to_RGB(XYZ, target, apply_cctf_encoding=True)

    # Convert between non-RGB spaces
    return colour.convert(a, source, target)


def deltaE(color_a, color_b, color_space='sRGB', metric='CIE 2000'):
    """
    Given two colors in a given color space, calculate the deltaE value using the specified uniform.
    :param color_a: Input color a.
    :param color_b: Input color b.
    :param color_space: Color space of the input colors.
    :param metric: In which uniform to calculate the deltaE value. Supported values:
     'CIE 1976', 'CIE 1994', 'CIE 2000', 'CMC', 'DIN99', 'ITP',
     'CAM02-LCD', 'CAM02-SCD', 'CAM02-UCS', 'CAM16-LCD', 'CAM16-SCD', 'CAM16-UCS'.
    :return: DeltaE value.
    """
    if metric in ['CIE 1976', 'CIE 1994', 'CIE 2000', 'CMC', 'DIN99']:
        method_space = 'CIE Lab'
    elif metric in ['ITP']:
        method_space = 'ICtCp'
    elif metric in ['CAM02-LCD', 'CAM02-SCD', 'CAM02-UCS', 'CAM16-LCD', 'CAM16-SCD', 'CAM16-UCS']:
        method_space = metric.replace('-', '')
    else:
        raise ValueError(f"Unsupported deltaE uniform: {metric}")

    if color_space != method_space:
        color_a = auto_convert(color_a, color_space, method_space)
        color_b = auto_convert(color_b, color_space, method_space)

    delta_e = colour.delta_E(color_a, color_b, method=metric)

    if method_space not in ['CIE Lab', 'ICtCp']:
        delta_e *= 100

    return delta_e


def get_boundary_hull(res=9, boundary='sRGB', workspace='CAM16-UCS'):
    """
    Generate a Delaunay triangulation of the boundary of the given color space in the given workspace.
    :param res: Grid resolution of each dimension.
    :param boundary: Color space to generate the boundary.
    :param workspace: Color space to project the boundary into.
    :return: Delaunay triangulation of the boundary.
    """
    x = np.linspace(0, 1, res)
    p = np.array(np.meshgrid(x, x, x)).reshape(3, -1).T
    table = auto_convert(p, boundary, workspace)
    hull = Delaunay(table)
    return hull


def in_bounds(p, boundary='sRGB', workspace='CAM16-UCS', hull=None):
    """
    Determine whether a point is within the boundary of the given color space in the given workspace.
    :param p: Coordinate of the given point.
    :param boundary: Given color space.
    :param workspace: Working color space.
    :param hull: [Optional] Delaunay triangulation of the boundary. If not provided, use converted colors to determine.
    :return: Boolean indicating whether the point is within the boundary.
    """
    if hull is None:
        s = auto_convert(p, workspace, boundary)
        valid = np.all((s >= 0) & (s <= 1), axis=-1)
    else:
        valid = hull.find_simplex(p) >= 0

    return valid


def init(num, bound_func, seed=None, **kwargs):
    """
    Initialize a set of random points within the given boundary.
    :param num: Number of points to generate.
    :param bound_func: Function to determine whether a point is within the boundary.
    :param seed: [Optional] Random seed.
    :param kwargs: Other arguments to pass to the bound_func.
    :return: numpy array of shape (num, 3) containing the generated points.
    """
    if seed is not None:
        np.random.seed(seed)
    points = np.empty((0, 3), dtype=np.float32)
    while len(points) < num:
        batch_size = max(num - len(points), 100)  # 一次生成的数量
        batch = np.random.uniform(-1, 1, (batch_size, 3))
        # 过滤掉超出边界的点
        valid = bound_func(batch, **kwargs)
        valid_points = batch[valid]
        points = np.vstack([points, valid_points])

    return points[:num]


def compute_forces(pos, tree, kf=2):
    """
    Compute the forces acting on each particle based on its proximity to its k-nearest neighbors.
    :param pos: Positions of all particles. (N, 3)
    :param tree: KDTree of the positions.
    :param kf: Scaling factor for the force.
    :return: Forces vector acting on each particle. (N, 3)
    """
    # Query the k-1 nearest neighbors for each particle
    dists, indices = tree.query(pos, k=2)
    # Nearest k-1 particles to each particle
    nearest_particles = pos[indices[:, 1:]]  # 形状: (N, k-1, 3)
    force_directions = nearest_particles - pos[:, np.newaxis, :]  # 形状: (N, k-1, 3)
    dists = np.maximum(dists[:, 1:], 1e-10)  # 避免除零，形状: (N, k-1)
    # Compute the force magnitudes and directions
    force_magnitudes = -kf / dists ** 1  # 力的大小，形状: (N, k-1)
    forces = force_directions * force_magnitudes[:, :, np.newaxis]  # 力，形状: (N, k-1, 3)
    forces = forces.sum(axis=1)  # 总的力，形状: (N, 3)

    return forces


def update_positions(s, v, a, dt, damping=0.99):
    """
    Update the positions and velocities of the particles.
    :param s: Positions of all particles. (N, 3)
    :param v: Velocities of all particles. (N, 3)
    :param a: Accelerations of all particles. (N, 3)
    :param dt: Timestep.
    :param damping: Damping factor for the velocity.
    :return: Updated positions and velocities. (N, 3), (N, 3)
    """
    s = s + v * dt  # + 0.5 * a * dt ** 2
    v = damping * v + a * dt
    return s, v


# @profile
def deal_out_of_bounds(s, v, dt, **kwargs):
    """
    Deal with particles that are out of bounds.
    :param s: Positions of all particles. (N, 3)
    :param v: Velocities of all particles. (N, 3)
    :param dt: Timestep.
    :param kwargs: Other kwargs to pass to the in_bounds function.
    :return: Updated positions and velocities. (N, 3), (N, 3)
    """
    # Determine which particles are out of bounds
    out_of_bounds = ~in_bounds(s, **kwargs)

    if np.any(out_of_bounds):
        # Shift particles back by one step
        s[out_of_bounds] -= v[out_of_bounds] * dt

        # Simulate a scattering event on the out-of-bounds particles
        num_out = np.sum(out_of_bounds)
        random_vectors = np.random.normal(0, 1, (num_out, s.shape[1]))
        random_vectors /= np.linalg.norm(random_vectors, axis=1, keepdims=True)

        # Update the velocity of the out-of-bounds particles
        v_magnitudes = np.linalg.norm(v[out_of_bounds], axis=1)  # Their original speeds
        v[out_of_bounds] = 0.99 * v_magnitudes[:, np.newaxis] * random_vectors  # New velocity vector

    return s, v


def maximize_delta_e(
        num, p0=None, source='sRGB', uniform='CAM16-UCS',
        dt=1E-4, t_end=1000, t_tol=1E-4, steps=1000, skip=10, damping=0.99,
        seed=None):
    """
    Main function to maximize deltaE.
    :param num: Number of particles to simulate.
    :param p0: Given initial colors. (N, 3)
    :param source: Input color space. Support color spaces that all componets are in range [0, 1],
     including all RGB spaces, and CMY space.
    :param uniform: Working color space. Must be a perceptually uniform space, such as CAM16-UCS.
    :param dt: Initial timestep.
    :param t_end: End time.
    :param t_tol: Tolerance for the maximum change in position when updating the timestep.
    :param steps: Maximum number of steps to simulate.
    :param skip: Number of steps to skip between each update of the KTree, minmium distance, and step size.
    :param damping: Damping factor for the velocity.
    :param seed: Random seed.
    :return: Generated colors and their corresponding time and step values.
    """

    # Transform input color space bounds to uniform bounds
    hull = get_boundary_hull(boundary=source, workspace=uniform)
    # hull = None

    # kwargs for in_bounds function
    kwarg = {'boundary': source, 'workspace': uniform, 'hull': hull}

    # Deal with given initial points
    if p0 is None:
        p0 = [[]]
    p0 = auto_convert(np.array(p0), source=source, target=uniform)
    num_fixed = len(p0)

    # Initialize the positions and velocities of the colors points.
    pos = init(num, in_bounds, seed=seed, **kwarg)
    if p0.shape[0] > 0:
        pos = np.vstack([p0, pos])
    vel = np.zeros_like(pos, dtype=np.float32)

    # Initialize the KDTree and other variables
    time = 0
    step = dt
    all_time = []
    d_mins = []
    max_dmin = 0
    best_pos = pos
    all_step = []
    tree = cKDTree(pos)

    # Main simulation loop
    for i in range(steps):
        if time >= t_end:
            break
        forces = compute_forces(pos, tree)
        forces[:num_fixed] = 0  # Fixed particles don't move
        new_pos, new_vel = update_positions(pos, vel, forces, step, damping=damping)
        new_pos, new_vel = deal_out_of_bounds(new_pos, new_vel, step, **kwarg)

        if i % skip == 0:
            # Update the KDTree
            tree = cKDTree(new_pos)

            # Calculate the minimum distance between points
            dists, _ = tree.query(new_pos, k=2)  # k=2 是为了排除自身
            dmin = np.min(dists[:, 1])  # 最近邻距离中最小的非自身距离
            d_mins.append(dmin)

            # Update the timestep
            max_deltas = np.max(np.linalg.norm(new_pos - pos, axis=1))
            if max_deltas < t_tol:
                step *= 1.2
            elif step > dt:
                step /= 1.2

            all_step.append(step)
            all_time.append(time + step)

            if dmin > max_dmin:
                max_dmin = dmin
                best_pos = new_pos

        time += step
        pos = new_pos
        vel = new_vel

    # Reorder the points by distance from the first point
    dists = np.linalg.norm(best_pos - best_pos[0], axis=1)
    best_pos = best_pos[np.argsort(dists)]

    # Convert the colors back to the source space
    colors = auto_convert(best_pos, source=uniform, target=source)
    # print(f"Max and Min values: {np.max(colors)}, {np.min(colors)}")

    return np.array(all_time), np.array(d_mins) * 100, np.array(colors), np.array(all_step)


def run(numbers, given_colors, target_space, metric_space, quality, seed=None):
    # Deal with default colors list
    if given_colors is None:
        given_colors = np.array([[1, 1, 1]])
    if len(np.shape(np.array(given_colors))) == 1:
        given_colors = np.array([given_colors])

    kwargs = SIM_PARAMS.get(quality, SIM_PARAMS['medium'])

    # Execute the simulation
    t0 = timer()
    times, dmin_list, points, dt_list = maximize_delta_e(
        numbers, given_colors, target_space, metric_space,
        seed=seed,
        **kwargs
    )
    t1 = timer()
    # print(f"Simulation time: {(t1 - t0) * 1000:.2f} ms")

    hex_colors = colour.notation.RGB_to_HEX(np.clip(points, 0, 1))  # 转换为HEX格式

    return hex_colors, dmin_list.max(), points, times, dmin_list, dt_list


def single_run(numbers, given_colors=None, target_space='sRGB', metric_space='CAM16-UCS', quality="medium"):
    hex_colors, dmin_max, points, times, dmin_list, dt_list = run(
        numbers, given_colors, target_space, metric_space, quality
    )

    # 可视化结果
    plot_points(points, times, dmin_list, "Time", r"Minimum $\Delta E$", source_space=target_space, target_space=metric_space)

    print(f"First 20 generated points in HEX format:\n{hex_colors[:20]}\n")
    print(f"Best ΔE_min: {dmin_max:.2f}")

    return hex_colors, dmin_max


def multi_run(numbers, given_colors=None, target_space='sRGB', metric_space='CAM16-UCS', quality="low", num_runs=20):
    # Deal with default colors list
    if given_colors is None:
        given_colors = np.array([[1, 1, 1]])
    if len(np.shape(np.array(given_colors))) == 1:
        given_colors = np.array([given_colors])

    # Execute the simulation
    t0 = timer()
    dmin_maxs = [0]
    best_points = None
    results = []

    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(run, numbers, given_colors, target_space, metric_space, quality, seed=i): i for i in range(num_runs)}
        for future in as_completed(futures):
            i = futures[future]
            try:
                hex_colors, dmin_max, points, times, dmin_list, dt_list = future.result()
                print(f"Run {i + 1}/{num_runs} completed with ΔE_max: {dmin_max:.2f}")
                if dmin_max > np.max(dmin_maxs):
                    best_points = points
                dmin_maxs.append(dmin_max)
                results.append((points, times, dmin_list, dt_list))
            except Exception as e:
                print(f"Run {i + 1}/{num_runs} failed with error: {e}")

    t3 = timer()
    print(f"Total time: {(t3 - t0) * 1000:.2f} ms")
    print(f"Best ΔE_max: {np.max(dmin_maxs):.2f}")

    # 可视化结果
    if best_points is not None:
        plot_points(best_points, np.arange(len(dmin_maxs)), dmin_maxs, "Runs", r"Maximum $\Delta E$", source_space=target_space,
                    target_space=metric_space)

    hex_colors = colour.notation.RGB_to_HEX(np.clip(best_points, 0, 1))  # 转换为HEX格式

    print(f"First 20 generated points in HEX format:\n{hex_colors[:20]}\n")

    return hex_colors, np.max(dmin_maxs)


def plot_points(a, x, y, xlabel="X", ylabel="Y", source_space='sRGB', target_space='CIE xyY'):
    colour.set_domain_range_scale("1")
    a = np.array(a)

    if target_space in RGB_SPACES:
        labels = ['Red', 'Green', 'Blue']
    elif target_space in CAM_SPACES:
        labels = ['J', 'a', 'b']
        target_space = target_space.replace('-', '')
    else:
        labels = LABEL_DICT.get(target_space, ["unknown x", "unknown y", "unknown z"])

    colors = auto_convert(a, source_space, target_space)
    if target_space in RGB_SPACES:
        srgb_colors = a
    else:
        srgb_colors = auto_convert(a, source_space, 'sRGB')
    srgb_colors = np.clip(srgb_colors, 0, 1)

    if labels[0][0] in ["L", "J", "I", "Y", ]:
        labels = labels[1:] + labels[:1]  # 调整标签顺序
        colors = np.hstack((colors[:, 1:], colors[:, :1]))  # 调整坐标顺序

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.scatter(colors[:, 0], colors[:, 1], colors[:, 2], c=srgb_colors, marker='o')
    ax.set_title(f"Points in {target_space} space")
    ax.set_xlabel(f"${labels[0]}$")
    ax.set_ylabel(f"${labels[1]}$")
    ax.set_zlabel(f"${labels[2]}$")
    x_lim = np.max(np.abs(colors[:, 0]))
    y_lim = np.max(np.abs(colors[:, 1]))
    xy_lim = max(x_lim, y_lim)
    # z_lim = np.max(np.abs(colors[:, 2]))
    ax.set_xlim([-xy_lim, xy_lim])
    ax.set_ylim([-xy_lim, xy_lim])
    ax.set_zlim([0, 1])
    ax.set_box_aspect((1, 1, 1))

    ax_2 = fig.add_subplot(1, 2, 2)
    ax_2.semilogx(x, y, label=ylabel, color='tab:blue')
    ax_2.set_title(ylabel + " vs " + xlabel)
    ax_2.set_xlabel(xlabel)
    ax_2.set_ylabel(ylabel)
    ax_2.legend()

    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    # single_run(3, quality='low')
    multi_run(3, quality='medium', num_runs=16)

    # import csv
    #
    # result = [[0, 0, [None]]]
    # for n in np.arange(256, 257, 1):
    #     hexs, de = main(n)
    #     result.append([n, de, hexs])
    #
    #     csv_file = "result3.csv"
    #     with open(csv_file, "w", newline='') as f:
    #         writer = csv.writer(f)
    #         writer.writerow(["Number of Colors", "Delta E", "Colors"])
    #         for row in result:
    #             new_row = [row[0], row[1], *row[2]]
    #             writer.writerow(new_row)
