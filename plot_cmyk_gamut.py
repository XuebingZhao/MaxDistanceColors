from PIL import Image, ImageCms
import os

import numpy as np
import matplotlib.pyplot as plt
from colour.plotting import plot_RGB_colourspaces_in_chromaticity_diagram_CIE1931, plot_RGB_colourspaces_in_chromaticity_diagram_CIE1976UCS

from VizPalette import auto_convert

# ICC 配置文件路径（请根据实际情况修改）
cmyk_profile_path = r'.\JapanColor2001Coated.icc'
rgb_profile_path = r'C:\Windows\System32\spool\drivers\color\Rec2020.icm'
cmyk_profile_name = os.path.splitext(os.path.basename(cmyk_profile_path))[0].split(' ')[0]
rgb_profile_name = os.path.splitext(os.path.basename(rgb_profile_path))[0].split(' ')[0]


def cmyk_to_rgb(cmyk_colors, cmyk_profile, rgb_profile):
    """
    利用 ICC 配置文件将 CMYK 色彩转换为 RGB 色彩。

    :param cmyk_colors: 待转换的 CMYK 颜色列表，每个颜色为四个浮点数 (C, M, Y, K) 的元组，
                        或者传入单一颜色元组。
    :param cmyk_profile: CMYK 色彩空间对应的 ICC 配置文件路径。
    :param rgb_profile: 目标 RGB 色彩空间对应的 ICC 配置文件路径。
    :return: 转换后的 RGB 颜色列表，每个颜色以 [R, G, B] 的形式表示（范围 0–1）。
    """
    transform = ImageCms.buildTransformFromOpenProfiles(
        ImageCms.getOpenProfile(cmyk_profile),
        ImageCms.getOpenProfile(rgb_profile),
        inMode='CMYK', outMode='RGB',
        renderingIntent=ImageCms.Intent.ABSOLUTE_COLORIMETRIC
    )

    cmyk_array = ((np.atleast_2d(cmyk_colors)*255.0)
                  .astype(np.uint8).reshape(-1, 1, 4))
    cmyk_image = Image.fromarray(cmyk_array, 'CMYK')

    rgb_image = ImageCms.applyTransform(cmyk_image, transform)
    rgb_colors = np.array(rgb_image).reshape(-1, 3)

    # 如果输入是单个颜色，返回一维数组
    if len(cmyk_colors) == 1 and isinstance(cmyk_colors[0], (int, float)):
        rgb_colors = rgb_colors.reshape(-1)

    return rgb_colors/255.0


def rgb_to_cmyk(rgb_colors, rgb_profile, cmyk_profile):
    """
    利用 ICC 配置文件将 RGB 色彩转换为 CMYK 色彩。

    :param rgb_colors: 待转换的 RGB 颜色列表，每个颜色为三个浮点数 (R, G, B) 的元组，
                       或者传入单一颜色元组。
    :param rgb_profile: RGB 色彩空间对应的 ICC 配置文件路径。
    :param cmyk_profile: 目标 CMYK 色彩空间对应的 ICC 配置文件路径。
    :return: 转换后的 CMYK 颜色列表，每个颜色以 [C, M, Y, K] 的形式表示（范围 0–1）。
    """
    transform = ImageCms.buildTransformFromOpenProfiles(
        ImageCms.getOpenProfile(rgb_profile),
        ImageCms.getOpenProfile(cmyk_profile),
        inMode='RGB', outMode='CMYK',
        renderingIntent=ImageCms.Intent.ABSOLUTE_COLORIMETRIC
    )

    rgb_array = ((np.atleast_2d(rgb_colors)*255.0)
                 .astype(np.uint8).reshape(-1, 1, 3))
    rgb_image = Image.fromarray(rgb_array, 'RGB')

    cmyk_image = ImageCms.applyTransform(rgb_image, transform)
    cmyk_colors = np.array(cmyk_image).reshape(-1, 4) / 255.0

    # 如果输入是单个颜色，返回一维数组
    if len(rgb_colors) == 1 and isinstance(rgb_colors[0], (int, float)):
        cmyk_colors = cmyk_colors.reshape(-1)

    return cmyk_colors



def rgb_to_xy(rgb):
    """
    利用 colour.convert 将 sRGB 色彩（0-255 范围）转换到 xyY 空间，
    并返回 xy 两个色度坐标。

    :param rgb: [R, G, B] 列表或元组，值范围 0–255。
    :return: (x, y)
    """
    # 归一化至 [0,1] 范围，因为 colour.convert 默认以该范围为输入
    rgb_normalized = np.array(rgb) #/ 255.0
    # 直接使用 colour.convert 完成从 sRGB 到 xyY 的转换
    xyY = auto_convert(rgb_normalized, 'ITU-R BT.2020', 'CIE xyY')
    return xyY[:, :2]

def rgb_to_uv(rgb):
    """
    利用 colour.convert 将 sRGB 色彩（0-255 范围）转换到 uv 空间，
    并返回 uv 两个色度坐标。

    :param rgb: [R, G, B] 列表或元组，值范围 0–255。
    :return: (u, v)
    """
    # 归一化至 [0,1] 范围，因为 colour.convert 默认以该范围为输入
    rgb_normalized = np.array(rgb) #/ 255.0
    Luv = auto_convert(rgb_normalized, 'ITU-R BT.2020', 'CIE 1976 UCS')
    return Luv[:, :2]


def sample_gamut_edges(resolution=50):
    """
    沿着立方体边界上关键顶点构成的闭合路径采样 CMYK 色点（K 固定为 0）。
    路径定义为：
        [1,0,0] → [1,1,0] → [0,1,0] → [0,1,1] → [0,0,1] → [1,0,1] → [1,0,0]

    :param resolution: 每条边采样点数（包括起点和终点）。
    :return: 采样得到的 CMYK 色点列表，每个点为 (C, M, Y, 0)。
    """
    vertices = [
        np.array([1, 0, 0]),
        np.array([1, 1, 0]),
        np.array([0, 1, 0]),
        np.array([0, 1, 1]),
        np.array([0, 0, 1]),
        np.array([1, 0, 1]),
        np.array([1, 0, 0])
    ]
    samples = []
    # 对每条边进行线性插值
    for i in range(len(vertices) - 1):
        start = vertices[i]
        end = vertices[i + 1]
        # 用 linspace 插值（包含端点）
        for t in np.linspace(0, 1, resolution, endpoint=True):
            point = (1 - t) * start + t * end
            # 生成 CMYK 点，K 固定为 0，四舍五入取整
            c_val, m_val, y_val = point
            samples.append((c_val, m_val, y_val, 0))
    return samples


def plot_cmyk_gamut_edges(cmyk_profile, rgb_profile, resolution=50):
    """
    绘制 CMYK 色域边界在 CIE 1931 色度图中的分布。采样方式只沿着指定的立方体边界路径采样，
    因此计算点较少，但足以展示 gamut 的外轮廓。

    :param cmyk_profile: CMYK 色彩空间的 ICC 配置文件路径。
    :param rgb_profile: sRGB 色彩空间的 ICC 配置文件路径。
    :param resolution: 每边所采样的点数。
    """
    # 采样 CMYK 点（K 固定为 0）
    cmyk_samples = sample_gamut_edges(resolution)

    # 转换为 sRGB
    rgb_samples = cmyk_to_rgb(cmyk_samples, cmyk_profile, rgb_profile)


    # 绘制 CIE 1931 色度图背景
    # _, ax = plot_RGB_colourspaces_in_chromaticity_diagram_CIE1931(
    #     ["sRGB", "Adobe RGB (1998)", "DCI-P3", "ITU-R BT.2020"], show=False)
    # ax.plot(*rgb_to_xy(rgb_samples).T, linewidth=1.5, marker=None, color='k', label=cmyk_profile_name)

    # 绘制 CIE 1976 UCS 色度图背景
    _, ax = plot_RGB_colourspaces_in_chromaticity_diagram_CIE1976UCS(
        ["sRGB", "Adobe RGB (1998)", "DCI-P3", "ITU-R BT.2020"], show=False)
    ax.plot(*rgb_to_uv(rgb_samples).T, linewidth=1.5, marker=None, color='k', label=cmyk_profile_name)

    ax.set_title("Color Gamuts")
    ax.legend()
    plt.show()


if __name__ == "__main__":
    # 示例：多个 CMYK 转换
    multiple_cmyk = [
        (0.5, 0.5, 0.5, 0.5),
        (1, 0, 1, 0),
        (1, 1, 0, 0)
    ]
    print(f"{cmyk_profile_name}转{rgb_profile_name}：", cmyk_to_rgb(multiple_cmyk, cmyk_profile_path, rgb_profile_path))

    # 示例：单个 rgb 转换
    single_rgb = (0, 0, 0.1)
    print(f"{rgb_profile_name}转{cmyk_profile_name}：", rgb_to_cmyk(single_rgb, rgb_profile_path, cmyk_profile_path))

    # 绘制沿立方体边界采样的 CMYK gamut 边界
    plot_cmyk_gamut_edges(cmyk_profile_path, rgb_profile_path, resolution=255)
