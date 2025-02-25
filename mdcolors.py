# Used to generate color palettes for visualizations.
# By simulating the color points repelling each other, to maximize the delta E between them.

import os
from time import perf_counter as timer
from PIL import Image, ImageCms

import numpy as np
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button
from concurrent.futures import ProcessPoolExecutor, as_completed
import colour
import open3d as o3d


# Resize factor for brightness
KL = 1.0
# Maximize contrast between adjacent colors in the output list
MAX_ADJ_CONTRAST = False
# Set Parameters for converting to/from CMYK icc profiles
cdir = os.path.dirname(os.path.abspath(__file__))
CMYK_PARAMS = {
    'cmyk_icc': os.path.join(cdir, 'data/JapanColor2001Coated.icc'),
    'srgb_icc': os.path.join(cdir, 'data/sRGB Color Space Profile.icm'),
    'rec2020_icc': os.path.join(cdir, 'data/Rec2020.icm'),
    # 'renderingIntent': ImageCms.Intent.RELATIVE_COLORIMETRIC,
    # 'renderingIntent': ImageCms.Intent.ABSOLUTE_COLORIMETRIC,
    'flags': ImageCms.Flags.BLACKPOINTCOMPENSATION,
    # 'interm_space': 'ITU-R BT.2020',
    'output_cmyk': True,  # Else output RGB, which might oversaturated.
}
CMYK_PARAMS.setdefault('interm_space', 'sRGB')
if CMYK_PARAMS['interm_space'] == 'ITU-R BT.2020':
    CMYK_PARAMS['rgb_icc'] = CMYK_PARAMS['rec2020_icc']
else:
    CMYK_PARAMS['rgb_icc'] = CMYK_PARAMS['srgb_icc']
CMYK_PARAMS.setdefault('renderingIntent', ImageCms.Intent.PERCEPTUAL)
CMYK_PARAMS.setdefault('flags', ImageCms.Flags.NONE)
CMYK_PARAMS.setdefault('output_cmyk', False)

# Set colourspace
colour.set_domain_range_scale("1")
RGB_SPACES = sorted(colour.RGB_COLOURSPACES)
CAM_SPACES = [
    'CAM02-LCD', 'CAM02-SCD', 'CAM02-UCS', 'CAM16-LCD', 'CAM16-SCD', 'CAM16-UCS',
    'CAM02LCD', 'CAM02SCD', 'CAM02UCS', 'CAM16LCD', 'CAM16SCD', 'CAM16UCS'
]
UCS_SPACES = [
    'CIE Lab', 'DIN99', 'DIN99b', 'DIN99c', 'DIN99d', 'ICaCb', 'Oklab', 'Jzazbz', 'ICtCp', *CAM_SPACES[-6:]
]
# Set labels based on color space
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
    'DIN99b': ["L_{99}", "a_{99}", "b_{99}"],
    'DIN99c': ["L_{99}", "a_{99}", "b_{99}"],
    'DIN99d': ["L_{99}", "a_{99}", "b_{99}"],
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
    'Jzazbz': ["J_z", "a_z", "b_z"],
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
LABEL_DICT = {key.lower(): value for key, value in LABEL_DICT.items()}  # Convert keys to lowercase

# Simulation parameters to choose from
SIM_PARAMS = {
    'fast': {
        'dt': 0.001,
        't_end': 500,
        'steps': 5000,
        't_tol': 0.002,
        'damping': 0.95,
        'hull_res': 11,
    },
    'medium': {
        'dt': 0.0001,
        't_end': 1000,
        'steps': 10000,
        't_tol': 0.001,
        'damping': 0.99,
        'hull_res': 15,
    },
    'slow': {
        'dt': 0.0001,
        't_end': 2000,
        'steps': 100000,
        't_tol': 0.0002,
        'damping': 0.998,
        'hull_res': 17,
    },
}

CMYK_TO_RGB_TRANSFROM = ImageCms.buildTransformFromOpenProfiles(
    ImageCms.getOpenProfile(CMYK_PARAMS['cmyk_icc']),
    ImageCms.getOpenProfile(CMYK_PARAMS['rgb_icc']),
    inMode='CMYK', outMode='RGB',
    renderingIntent=CMYK_PARAMS['renderingIntent'],
    flags=CMYK_PARAMS['flags']
)
RGB_TO_CMYK_TRANSFORM = ImageCms.buildTransformFromOpenProfiles(
    ImageCms.getOpenProfile(CMYK_PARAMS['rgb_icc']),
    ImageCms.getOpenProfile(CMYK_PARAMS['cmyk_icc']),
    inMode='RGB', outMode='CMYK',
    renderingIntent=CMYK_PARAMS['renderingIntent'],
    flags=CMYK_PARAMS['flags']
)


def cmyk_to_rgb(cmyk_colors):
    """
    Convert CMYK colors to RGB using ICC profiles.

    :param cmyk_colors: List of CMYK colors to convert, where each color is a list of four integers [C, M, Y, K],
                        or a single color list.
    :return: List of converted RGB colors, where each color is a list [R, G, B] with values in the range 0.0-1.0.
    """
    # Ensure input is a 2D array and scale to 0-255
    cmyk_array = (np.atleast_2d(cmyk_colors) * 255.0).astype(np.uint8).reshape(-1, 1, 4)
    cmyk_image = Image.fromarray(cmyk_array, 'CMYK')

    rgb_image = ImageCms.applyTransform(cmyk_image, CMYK_TO_RGB_TRANSFROM)
    rgb_colors = np.array(rgb_image).reshape(-1, 3) / 255.0
    rgb_colors = np.clip(rgb_colors, 0.0, 1.0)

    # If input is a single color, return a 1D array
    if len(cmyk_colors) == 1 and isinstance(cmyk_colors[0], (int, float)):
        rgb_colors = rgb_colors.reshape(-1)

    # Scale RGB values to 0.0-1.0
    return rgb_colors


def rgb_to_cmyk(rgb_colors):
    """
    Convert RGB colors to CMYK using ICC profiles.

    :param rgb_colors: List of RGB colors to convert, where each color is a list of three floats [R, G, B],
                       or a single color list with values in the range 0.0-1.0.
    :return: List of converted CMYK colors, where each color is a list [C, M, Y, K] with values in the range 0.0-1.0.
    """
    # Ensure input is a 2D array and scale to 0-255
    rgb_array = (np.atleast_2d(rgb_colors) * 255.0).astype(np.uint8).reshape(-1, 1, 3)
    rgb_image = Image.fromarray(rgb_array, 'RGB')

    cmyk_image = ImageCms.applyTransform(rgb_image, RGB_TO_CMYK_TRANSFORM)
    cmyk_colors = np.array(cmyk_image).reshape(-1, 4) / 255.0
    cmyk_colors = np.clip(cmyk_colors, 0.0, 1.0)

    # If input is a single color, return a 1D array
    if len(rgb_colors) == 1 and isinstance(rgb_colors[0], (int, float)):
        cmyk_colors = cmyk_colors.reshape(-1)

    return cmyk_colors


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

    if source in ['CMYK', 'CMY']:
        a = np.atleast_2d(a)
        if a.shape[1] == 3:
            a = np.column_stack((a, np.zeros((a.shape[0], 1))))
        rgb = cmyk_to_rgb(a)
        if len(np.array(a).shape) == 1:
            rgb = rgb.flatten()
        return auto_convert(rgb, CMYK_PARAMS['interm_space'], target)

    if target in ['CMYK', 'CMY']:
        rgb = auto_convert(a, source, CMYK_PARAMS['interm_space'])
        return rgb_to_cmyk(rgb)

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
        return auto_convert(XYZ, 'CIE XYZ', target)

    # Convert from other spaces to RGB
    if target in RGB_SPACES:
        XYZ = auto_convert(a, source, 'CIE XYZ')
        return colour.XYZ_to_RGB(XYZ, target, apply_cctf_encoding=True)

    # Convert from DIN99 to other spaces
    if source in ['DIN99b', 'DIN99c', 'DIN99d']:
        XYZ = colour.DIN99_to_XYZ(a, method=source)
        return auto_convert(XYZ, 'CIE XYZ', target)

    # Convert from other spaces to DIN99
    if target in ['DIN99b', 'DIN99c', 'DIN99d']:
        XYZ = auto_convert(a, source, 'CIE XYZ')
        return colour.XYZ_to_DIN99(XYZ, method=target)

    # Convert between non-RGB spaces
    return colour.convert(a, source, target)


def deltaE(color_a, color_b, color_space='sRGB', method='CIE 2000') -> np.ndarray[float] | float:
    if method in ['CIE 1976', 'CIE 1994', 'CIE 2000', 'CMC', 'DIN99']:
        method_space = 'CIE Lab'
    elif method in ['ITP']:
        method_space = 'ICtCp'
    elif method in ['CAM02-LCD', 'CAM02-SCD', 'CAM02-UCS', 'CAM16-LCD', 'CAM16-SCD', 'CAM16-UCS']:
        method_space = method.replace('-', '')
    else:
        raise ValueError(f"Unsupported deltaE method: {method}")

    if color_space != method_space:
        color_a = auto_convert(color_a, color_space, method_space)
        color_b = auto_convert(color_b, color_space, method_space)

    delta_e = colour.delta_E(color_a, color_b, method=method)

    if method_space not in ['CIE Lab', 'ICtCp']:
        delta_e *= 100

    return delta_e


def _o3d_contains(scene, points):
    t = o3d.core.Tensor([points], dtype=o3d.core.float32)
    distances = scene.compute_signed_distance(t)
    return distances.numpy()[0] < 0.001


def _o3d_build_bound(points, alpha=None):
    """
    使用Open3D的Alpha Shapes生成曲面并转换为trimesh对象
    :param points: (n,3) numpy数组，输入点云
    :param alpha: 可选，Alpha值（自动计算时设为None）
    :return: (trimesh.Trimesh, o3d.t.geometry.RaycastingScene)
    """
    # 转换为Open3D点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # 自动计算alpha值
    max_spacing = np.max(pcd.compute_nearest_neighbor_distance())
    if alpha is not None:
        alpha = max(max_spacing * 1.5, alpha)

    # 生成Alpha Shape网格
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
    # 清理三角形
    mesh.compute_vertex_normals()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_non_manifold_edges()

    # 创建Open3D射线投射场景
    scene = o3d.t.geometry.RaycastingScene()
    mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene.add_triangles(mesh_t)

    return mesh, scene


def get_boundary_hull(bound_space='sRGB', workspace='CAM16LCD', res=11, show=False):
    """
    Generate a Delaunay triangulation of the boundary of the given color space in the given workspace.
    :param bound_space: Color space to generate the boundary.
    :param workspace: Color space to project the boundary into.
    :param res: Grid resolution of each dimension.
    :return: Delaunay triangulation of the boundary.
    :param show: Whether to show the boundary.
    """
    x = np.linspace(0, 1, res)
    if bound_space in ['CMYK', 'CMY']:
        k = np.linspace(0, 1, int(res * 0.5))
        p = np.array(np.meshgrid(*[x] * 3, k)).reshape(4, -1).T
        boundary_mask = np.any((p == 0) | (p == 1), axis=1)
        sp = p[boundary_mask]
    else:
        sp = np.array(np.meshgrid(*[x] * 3)).reshape(3, -1).T

    table = auto_convert(sp, bound_space, workspace)
    # validate_result(table, workspace, workspace, show=True)
    if bound_space in ['CMYK', 'CMY']:
        table = auto_convert(auto_convert(table, workspace, bound_space), bound_space, workspace)
    table[:, 0] *= KL

    hull, scene = _o3d_build_bound(table, 1.5 / res)

    if show:
        validate_result(table, workspace, workspace, hull, show=True)

    return hull, scene


def in_bounds(p, bound_space='sRGB', workspace='CAM16LCD', bounds=None):
    """
    Determine whether a point is within the boundary of the given color space in the given workspace.
    :param p: Coordinate of the given point.
    :param bound_space: Given color space.
    :param workspace: Working color space.
    :param bounds: [Optional] Open3D mesh of the boundary. If not provided, use converted colors to determine.
    :return: Boolean indicating whether the point is within the boundary.
    """
    if bounds is None:
        s = auto_convert(p, workspace, bound_space)
        valid = np.all((s >= 0) & (s <= 1), axis=-1)
    else:
        valid = _o3d_contains(bounds, p)

    return valid


def _init_points(num, bound_func, seed=None, **kwargs):
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
        batch = np.random.normal(0, 1, (batch_size, 3))
        batch /= np.linalg.norm(batch, axis=1, keepdims=True)  # 归一化
        batch *= 0.1
        batch += [0.5, 0, 0]
        # 过滤掉超出边界的点
        valid = bound_func(batch, **kwargs)
        valid_points = batch[valid]
        points = np.vstack([points, valid_points])

    return points[:num]


def _compute_forces(pos, tree, kf=2):
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


def _update_positions(s, v, a, dt, damping=0.99):
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


def _deal_out_of_bounds(s, v, dt, **kwargs):
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
        v[out_of_bounds] = 0.995 * v_magnitudes[:, np.newaxis] * random_vectors  # New velocity vector

    return s, v


def arrange_points(p):
    """
    Fast arrange points by Greedy Method
    :param p: point list
    :return: np.array of arranged points
    """
    p_c = p.copy()
    if len(p_c) == 0:
        return p_c
    path = [p_c[0]]
    remaining = p_c[1:]

    if remaining.size == 0:
        return np.array(path)

    distances = np.linalg.norm(remaining - path[0], axis=1)
    idx = np.argmax(distances)
    path.append(remaining[idx])
    remaining = np.delete(remaining, idx, axis=0)
    while remaining.shape[0] > 0:
        left = path[0]
        right = path[-1]

        d_left = np.linalg.norm(remaining - left, axis=1)
        d_right = np.linalg.norm(remaining - right, axis=1)
        max_d = np.maximum(d_left, d_right)
        idx = np.argmax(max_d)

        if d_left[idx] > d_right[idx]:
            path.insert(0, remaining[idx])
        else:
            path.append(remaining[idx])
        remaining = np.delete(remaining, idx, axis=0)

    return np.array(path)


def _compute_dmin_in_list(perm):
    """Calculate the minimum distance between adjacent points in the arrangement"""
    if len(perm) < 2:
        return 0
    distances = np.linalg.norm(np.diff(perm, axis=0), axis=1)
    return np.min(distances)


def sa_arrange(p):
    """
    Arrange points using Simulated Annealing algorithm
    :param p: point list
    :return: np.array of arranged points
    """
    n = p.shape[0]
    if n <= 2:
        return p.copy()

    initial_temp = 1
    cooling_rate = 0.99
    min_temp = 1e-2 / (n ** 2)
    iterations_per_temp = min(max(1, int(n / 5)), 100)

    # Initialize the current permutation by greedy method
    current_perm = p  # arrange_points(p)
    current_min_dist = _compute_dmin_in_list(current_perm)

    best_perm = current_perm.copy()
    best_min_dist = current_min_dist

    temp = initial_temp
    while temp > min_temp:
        for _ in range(iterations_per_temp):
            # Generate a new solution: randomly swap the positions of two non-initial points
            i, j = np.random.choice(range(1, n), size=2, replace=False)  # Keep the first point at the beginning
            new_perm = current_perm.copy()
            new_perm[[i, j]] = new_perm[[j, i]]

            new_min_dist = _compute_dmin_in_list(new_perm)

            # Metropolis criterion
            delta = current_min_dist - new_min_dist
            if delta < 0 or np.exp(-delta / temp) > np.random.rand():
                current_perm = new_perm
                current_min_dist = new_min_dist

                if current_min_dist > best_min_dist:
                    best_perm = current_perm.copy()
                    best_min_dist = current_min_dist

        # Cool down temperature
        temp *= cooling_rate

    return best_perm


def compute_min_distance(pos, tree=None):
    if tree is None:
        tree = KDTree(pos)
    dists, _ = tree.query(pos, k=2, workers=1)  # k=2 to exclude self
    return np.min(dists[:, 1])  # Minimum distance to nearest neighbor


def simulated_annealing(pos, t_initial=8E-4, t_final=1E-9,
                        cooling_rate=0.998, steps=20000,
                        num_fixed=1, **bound_kwargs):
    """
    Simulated annealing version to maximize minimum distance between particles.

    :param pos: Initial points
    :param t_initial: Initial temperature
    :param t_final: Final temperature
    :param cooling_rate: Geometric cooling factor (0 < cooling_rate < 1)
    :param steps: Maximum number of iterations
    :param num_fixed: Number of particles to be fixed
    """
    # Initial setup
    current_min_dist = compute_min_distance(pos)
    best_pos = pos.copy()
    best_min_dist = current_min_dist
    all_dmin = []
    all_pos = [pos.copy()]

    # Simulated annealing parameters
    temperature = t_initial
    iteration = 0

    while temperature > t_final and iteration < steps:
        # Propose a new configuration by perturbing a random particle
        new_pos = pos.copy()
        # movable_indices = np.random.randint(num_fixed, len(pos))
        movable_indices = np.random.choice(range(num_fixed, len(pos)), size=min(len(pos) - num_fixed, 1), replace=False)
        # movable_indices = range(num_fixed, len(pos))
        perturbations = np.random.normal(0, 1.0 * best_min_dist * temperature ** 0.5, size=(len(movable_indices), 3))
        new_pos[movable_indices] += perturbations

        # Boundary check
        out_of_bounds = ~in_bounds(new_pos[movable_indices], **bound_kwargs)
        if np.all(out_of_bounds):
            continue

        new_pos[movable_indices] = np.where(out_of_bounds[:, None], pos[movable_indices], new_pos[movable_indices])

        # if not in_bounds(new_pos[movable_indices], **bound_kwargs):
        #     continue

        # Evaluate new configuration
        new_min_dist = compute_min_distance(new_pos)
        # Energy difference (negative because we maximize distance)
        delta_energy = -(new_min_dist - current_min_dist)

        # Acceptance probability
        if delta_energy < 0 or np.random.rand() < np.exp(-delta_energy / temperature):
            pos = new_pos
            current_min_dist = new_min_dist

            # Update best solution if improved
            if current_min_dist > best_min_dist:
                best_pos = pos.copy()
                best_min_dist = current_min_dist

        # Cooling schedule
        temperature *= cooling_rate
        iteration += 1

        # Record data periodically (e.g., every 10 steps)
        if iteration % 10 == 0:
            all_dmin.append(current_min_dist)
            all_pos.append(pos.copy())

    return best_pos, np.array(all_dmin), np.array(all_pos)


def maximize_delta_e(
        num, p0=None, source='sRGB', uniform='CAM16LCD',
        dt=1E-4, t_end=1000, t_tol=1E-4, steps=1000, damping=0.99,
        seed=None, **kwargs):
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
    :param damping: Damping factor for the velocity.
    :param seed: Random seed.
    :return: Generated colors and their corresponding time and step values.
    """
    if uniform not in UCS_SPACES:
        raise ValueError(f"'{uniform}' is not a perceptually uniform space. "
                         f"Please use a uniform space in:\n {UCS_SPACES}")

    # Transform input color space bounds to uniform bounds
    hull_res = kwargs.get('hull_res', 11)
    _, bounds = kwargs.get('bounds', get_boundary_hull(source, workspace=uniform, res=hull_res, show=False))

    # kwargs for in_bounds function
    bound_kwargs = {'bound_space': source, 'workspace': uniform, 'bounds': bounds}

    # Deal with given initial points
    if p0 is None:
        p0 = [[]]
    p0 = np.atleast_2d(p0)
    if source in ['CMY', 'CMYK'] and p0.shape[1] == 3:
        p0 = auto_convert(p0, CMYK_PARAMS.get('interm_space', 'sRGB'), uniform)
    else:
        p0 = auto_convert(p0, source, uniform)
    num_fixed = len(p0)
    p0[:, 0] *= KL

    # Initialize the positions and velocities of the colors points.
    pos = _init_points(num - num_fixed, in_bounds, seed=seed, **bound_kwargs)
    if num_fixed > 0:
        pos = np.vstack([p0, pos])
    vel = np.zeros_like(pos, dtype=np.float32)

    # Initialize the KDTree and other variables
    time = 0
    step = dt
    best_pos = pos.copy()
    best_dmin = compute_min_distance(pos)
    all_dmin = []
    all_pos = [pos.copy()]
    tree = KDTree(pos)

    # Main simulation loop
    for i in range(steps):
        if time >= t_end:
            break
        forces = _compute_forces(pos, tree)
        forces[:num_fixed] = 0  # Fixed particles don't move
        new_pos, new_vel = _update_positions(pos, vel, forces, step, damping=damping)
        new_pos, new_vel = _deal_out_of_bounds(new_pos, new_vel, step, **bound_kwargs)

        # Update the timestep
        t_tol = min(t_tol, 10 / (i + 1))
        max_deltas = np.max(np.linalg.norm(new_pos - pos, axis=1))
        if max_deltas < t_tol:
            step *= 1.2
        elif step > dt:
            step /= 1.2

        time += step
        pos = new_pos
        vel = new_vel

        # Update the KDTree
        tree = KDTree(new_pos)

        # Record data periodically (e.g., every 10 steps)
        if i % 10 == 0:
            # Calculate the minimum distance between points
            dmin = compute_min_distance(new_pos, tree)
            if dmin > best_dmin:
                best_dmin = dmin
                best_pos = new_pos

            all_dmin.append(dmin)
            all_pos.append(pos.copy())

    # Fine tune the colors by simulated annealing
    best_pos, dmin_sa, pos_sa = simulated_annealing(best_pos, num_fixed=num_fixed, **bound_kwargs)
    all_dmin = np.concatenate((all_dmin, dmin_sa), axis=0)
    all_pos = np.concatenate((all_pos, pos_sa), axis=0)

    # Rearrange the points by distance from the first point
    dists_to_start = np.linalg.norm(best_pos - best_pos[0], axis=1)
    best_pos = best_pos[np.argsort(dists_to_start)]
    # Arrange the points to maximize contrast
    if MAX_ADJ_CONTRAST:
        best_pos[0:] = sa_arrange(best_pos[0:])

    # Convert the colors back to the source space
    # For CMY space, convert back to RGB if output_cmyk is False
    if source in ['CMY', 'CMYK'] and not CMYK_PARAMS['output_cmyk']:
        source = CMYK_PARAMS.get('interm_space', 'sRGB')
    best_pos[:, 0] /= KL
    colors = auto_convert(best_pos, uniform, source)
    colors = np.clip(colors, 0, 1)
    colors[np.isnan(colors)] = 0

    return np.array(colors), all_dmin * 100, all_pos


def _get_space_labels(target_space):
    if target_space in RGB_SPACES:
        return ['Red', 'Green', 'Blue']
    elif target_space in CAM_SPACES:
        return ['J', 'a', 'b']
    else:
        return LABEL_DICT.get(target_space.lower(), ["unknown x", "unknown y", "unknown z"])


def _swap_labels_and_coords(labels, coords):
    if labels[0][0] in ["L", "J", "I", "Y"]:
        labels = labels[1:] + labels[:1]
        if isinstance(coords, list):
            coords = [np.roll(arr, -1, axis=-1) for arr in coords]
        else:
            coords = np.roll(coords, -1, axis=-1)
    return labels, coords


def _setup_3d_axes(ax: plt.Axes, labels):
    """
    Set up the properties of a 3D axes using the current xlim and ylim from ax.
    """
    # x_min, x_max = ax.get_xlim()
    # y_min, y_max = ax.get_ylim()
    # x_lim = np.max(np.abs([x_min, x_max]))
    # y_lim = np.max(np.abs([y_min, y_max]))
    # xy_lim = max(x_lim, y_lim)
    xy_lim = 0.5 * KL

    ax.set_xlabel(f"${labels[0]}$")
    ax.set_ylabel(f"${labels[1]}$")
    ax.set_zlabel(f"${labels[2]}$")
    ax.set_xlim([-xy_lim, xy_lim])
    ax.set_ylim([-xy_lim, xy_lim])
    ax.set_zlim([0, 1])
    ax.set_box_aspect((1, 1, 1))


def validate_result(points, source_space, target_space, hull=None, show=False):
    pos = auto_convert(points, source_space, target_space)
    _pos = pos.copy()
    _pos[:, 0] *= KL
    tree = KDTree(_pos)
    dists, nbr_idxs = tree.query(_pos, k=[2])
    dists = dists[:, 0] * 100.0

    # check deltaE CIE2000
    # nbrs = points[nbr_idxs[:, 0]]
    # de_forward = deltaE(points, nbrs, source_space)
    # de_backward = deltaE(nbrs, points, source_space)
    # dists = np.min([de_forward, de_backward], axis=0)
    # print("ΔE(CIE2000) of the points:\n", dists)

    de_min = np.min(dists)

    if show:
        labels = _get_space_labels(target_space)

        # Generate boundary hull
        if hull is None:
            hull, _ = get_boundary_hull(source_space, workspace=target_space)
        points = np.asarray(hull.vertices, dtype=np.float32)
        points[:, 0] /= KL
        triangles = np.asarray(hull.triangles, dtype=np.int32)

        # Adjust the order of labels and axis
        labels, [pos, points] = _swap_labels_and_coords(labels, [pos, points])

        # Plot the points in 3D with distances as colors
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(*pos.T, c=dists, cmap='viridis', s=50, alpha=1)
        ax.plot_trisurf(*points.T, triangles=triangles, alpha=0.5, color='#ccc')

        # Add a color bar which maps values to colors
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=10)
        cbar.set_label(r'$\Delta E$')

        # Set labels
        ax.set_title(rf'$\Delta E_{{min}}={de_min:.2f}$')
        _setup_3d_axes(ax, labels)

        # Set the view angle to -z direction
        ax.view_init(elev=90, azim=-90)
        ax.set_proj_type('ortho')

        # Show plot
        fig.tight_layout()
        plt.show()

    return de_min


def run(nums, given_colors, color_space='sRGB', uniform_space='CAM16LCD', quality='medium', seed=None, **kwargs):
    # Deal with default colors list
    if given_colors is None:
        given_colors = np.array([[1, 1, 1]])
    given_colors = np.atleast_2d(given_colors)

    kwargs.update(SIM_PARAMS.get(quality, SIM_PARAMS['medium']))

    # Execute the simulation
    colors, dmin_list, history = maximize_delta_e(
        nums, given_colors, color_space, uniform_space, seed=seed, **kwargs
    )

    # Convert the colors to HEX format
    rgb = colors
    if color_space in ["CMYK", "CMY"] and CMYK_PARAMS['output_cmyk']:  # Convert CMYK to RGB to generate HEX
        rgb = auto_convert(rgb, color_space, CMYK_PARAMS.get('interm_space', 'sRGB'))
    hex_colors = colour.notation.RGB_to_HEX(np.clip(rgb, 0, 1))

    return hex_colors, dmin_list.max(), colors, dmin_list, history


def single_run(nums, given_colors=None, color_space='sRGB', uniform_space='CAM16LCD', quality="fast", **kwargs):
    _color_space = color_space
    if color_space in ["CMYK", "CMY"] and not CMYK_PARAMS['output_cmyk']:
        _color_space = CMYK_PARAMS.get('interm_space', 'sRGB')
    # Execute the simulation
    t0 = timer()
    hex_colors, dmin, points, dmin_list, history = run(
        nums, given_colors, color_space, uniform_space, quality, **kwargs)
    t1 = timer()
    print(f"Total time: {(t1 - t0) * 1000:.2f} ms")
    print(f"Best δE: {dmin:.2f}")

    # 可视化结果
    real_dmin = validate_result(points, _color_space, uniform_space,
                                # show=True,
                                )
    plot_points_and_line(history, np.arange(len(dmin_list)), dmin_list, "Steps", r"Minimum $\Delta E$",
                         _color_space, uniform_space, **kwargs)
    plot_color_palette(hex_colors, points, color_space=color_space,
                       title=rf"{nums} {color_space} colors, $\Delta E_{{min}}={real_dmin:.1f}$ @ {uniform_space}")

    print(f"First 20 generated points in HEX format:\n{hex_colors[:20]}\n")

    return hex_colors, real_dmin


def smart_job_list(jobs, ref_workers):
    last_group_jobs = jobs % ref_workers
    if 0 < last_group_jobs < ref_workers / 2:
        return smart_job_list(jobs, ref_workers - 1)

    full_groups = jobs // ref_workers
    job_list = [ref_workers] * full_groups + ([last_group_jobs] if last_group_jobs != 0 else [])

    return job_list


def multi_run(nums, given_colors=None,
              color_space='sRGB', uniform_space='CAM16LCD', quality="fast", num_runs=20, show=True, **kwargs):
    _color_space = color_space
    if color_space in ["CMYK", "CMY"] and not CMYK_PARAMS['output_cmyk']:
        _color_space = CMYK_PARAMS.get('interm_space', 'sRGB')

    # Execute the simulation
    t0 = timer()
    dmins = []
    best_points = None
    best_hexs = None
    best_dmin = None

    workers = smart_job_list(num_runs, ref_workers=os.cpu_count() // 2 - 1)[0]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                run, nums, given_colors, color_space, uniform_space, quality, seed=i, **kwargs
            ): i for i in range(num_runs)
        }
        for future in as_completed(futures):
            i = futures[future]
            hex_colors, dmin, points, _, _ = future.result()
            dmin = validate_result(points, _color_space, uniform_space, show=False)
            print(f"Run {i + 1}/{num_runs} completed with ΔE: {dmin:.2f}")
            if best_dmin is None or dmin > best_dmin:
                best_points = points
                best_hexs = hex_colors
                best_dmin = dmin

            dmins.append(dmin)

    t1 = timer()
    print(f"Total time: {(t1 - t0) * 1000:.2f} ms")
    print(f"Best ΔE: {best_dmin:.2f}")

    # 可视化结果
    if best_points is not None:
        if show:
            plot_points_and_line(best_points, np.arange(len(dmins)), dmins, "Runs", r"$\Delta E_{min}$",
                                 _color_space, uniform_space)
            validate_result(best_points, _color_space, uniform_space,
                            # show=show,
                            )
            plot_color_palette(
                best_hexs, best_points, color_space=color_space,
                title=rf"{nums} {color_space} colors, $\Delta E_{{min}}={best_dmin:.1f}$ @ {uniform_space}")

        print(f"First 20 generated points in HEX format:\n{best_hexs[:20]}\n")

    return best_hexs, best_dmin


def plot_points_and_line(a, x, y, xlabel="X", ylabel="Y", source_space='sRGB', target_space='CIE xyY', **kwargs):
    colour.set_domain_range_scale("1")

    original_ndim = a.ndim

    # Set plot labels
    labels = _get_space_labels(target_space)

    # Generate show colors, if ndim > 2, asume the colors are in the tartget space already
    interm_space = CMYK_PARAMS.get('interm_space', 'sRGB')
    if original_ndim <= 2:
        colors = auto_convert(a, source_space, target_space)
        show_colors = auto_convert(a, source_space, interm_space) if target_space not in RGB_SPACES else a
    else:
        colors = a
        show_colors = [auto_convert(c, target_space, interm_space) for c in colors]
    show_colors = np.clip(show_colors, 0, 1)

    # Generate boundary hull
    hull_res = SIM_PARAMS.get(kwargs.get('quality', 'fast'), None).get('hull_res', 11)
    hull, _ = get_boundary_hull(source_space, workspace=target_space, res=hull_res)
    points = np.asarray(hull.vertices, dtype=np.float32)
    points[:, 0] /= KL
    triangles = np.asarray(hull.triangles, dtype=np.int32)

    # Adjust the order of labels and axis
    labels, [colors, points] = _swap_labels_and_coords(labels, [colors, points])

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax_2 = fig.add_subplot(1, 2, 2)

    scatter = ax.scatter([], [], [], marker='o', alpha=1.0)
    ax.plot_trisurf(*points.T, triangles=triangles, alpha=0.4, color='w')
    ax.set_title(f"Points in {target_space} space")
    _setup_3d_axes(ax, labels)

    if len(x) > 100:
        ax_2.semilogx(x, y, label=ylabel, color='tab:blue')
    else:
        ax_2.plot(x, y, label=ylabel, color='tab:blue')
    ax_2.set_title(ylabel + " vs " + xlabel)
    ax_2.set_xlabel(xlabel)
    ax_2.set_ylabel(ylabel)
    ax_2.legend()
    fig.tight_layout()

    if original_ndim == 3:
        playing = [True]
        max_show_frames = 300
        speed_factor = max(int(len(a) / max_show_frames), 1)

        def _init():
            scatter._offsets3d = ([], [], [])
            return scatter,

        colors = colors[::speed_factor]
        show_colors = show_colors[::speed_factor]

        def update(frame):
            p = colors[frame]
            scatter._offsets3d = (p[:, 0], p[:, 1], p[:, 2])
            scatter.set_color(c=show_colors[frame])
            return scatter,

        ani = FuncAnimation(
            fig, update, frames=len(colors), init_func=_init, blit=False,
            interval=33
        )

        ax_button = plt.axes((0.05, 0.9, 0.1, 0.04))  # 按钮位置 [left, bottom, width, height]
        button = Button(ax_button, 'Pause/Play')

        def toggle_play(e):
            if playing[0]:
                ani.event_source.stop()
            else:
                ani.event_source.start()
            playing[0] = not playing[0]

        button.on_clicked(toggle_play)
    else:
        ax.scatter(colors[:, 0], colors[:, 1], colors[:, 2], marker='o', alpha=1.0, c=show_colors)

    plt.show()


def plot_color_palette(hex_colors, original, color_space='sRGB', title="Color Palette"):
    """
    将颜色调色板可视化为带有 HEX 值的色块。

    :param hex_colors: Hex code for the ploted colors
    :param original: Original Colors
    :param color_space: Color space of original colors
    :param title: Plot tilte string
    """
    scale_range = 255
    if color_space in ["CMYK", "CMY"]:
        scale_range = 100
        if not CMYK_PARAMS['output_cmyk']:
            hex_colors = np.array(hex_colors, dtype=str)
            hex_to_rgb = colour.notation.HEX_to_RGB(hex_colors)
            original = auto_convert(hex_to_rgb, CMYK_PARAMS.get('interm_space', 'sRGB'), 'CMYK')

    n_colors = len(hex_colors)
    if n_colors > 256:
        simple_annotation = True
    else:
        simple_annotation = False

    # Calculate the number of rows and columns
    cols = int(n_colors ** 0.5) * 1
    rows = (n_colors + cols - 1) // cols

    fig, ax = plt.subplots(figsize=(max(min(cols * 1.5, 15), 8), max(rows * 0.6, 6)))
    ax.set_xlim(0, cols)
    ax.set_ylim(rows, 0)
    ax.axis('off')
    block_width = 1
    block_height = 1
    text_y_pos = 0.5

    brightness = auto_convert(original, color_space, 'CAM16UCS')[:, 0]
    for i, hex_color in enumerate(hex_colors):
        col = i % cols
        row = i // cols
        color_block = Rectangle((col, row), block_width, block_height, color=hex_color)
        ax.add_patch(color_block)

        text_color = 'black' if brightness[i] > 0.5 else 'white'
        if simple_annotation:
            text = f"{hex_color}"
        else:
            formatted_array = np.array2string(original[i],
                                              formatter={'float_kind': lambda x: f"{int(x * scale_range)}"})
            text = f"{hex_color}\n{formatted_array}"
        ax.text(col + block_width / 2, row + text_y_pos, text, ha='center', va='center', color=text_color)

    plt.title(title, pad=20)
    plt.tight_layout(pad=2)
    plt.show()


if __name__ == '__main__':
    ### Single function test.
    # print(255*auto_convert([0.38, 0.0, 0.0], 'CAM16UCS', 'sRGB'))
    get_boundary_hull("sRGB", "CIE Lab", 11, show=True)

    ### Run the simulation.
    # single_run(9, [1, 1, 1], color_space='sRGB', quality='medium')
    # single_run(9, [1, 1, 1], color_space='CMYK', uniform_space='DIN99d', quality='fast')
    # multi_run(9, [1, 1, 1], color_space='sRGB', quality='fast', uniform_space='DIN99d', num_runs=16)
    # multi_run(9, [1, 1, 1], color_space='CMYK', quality='fast', num_runs=16)
    # single_run(6)
