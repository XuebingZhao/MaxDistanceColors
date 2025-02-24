import csv
import numpy as np

from mdcolors import multi_run

if __name__ == '__main__':
    # sRGB_CAM16
    result = [[0, 0, [None]]]
    for n in np.arange(1, 52, 1):
        csv_file = "sRGB_CAM16.csv"
        hexs, de = multi_run(n+1, [1, 1, 1], color_space='sRGB', quality='medium', num_runs=32, show=False)
        hexs = [f"'{h}'" for h in hexs]
        result.append([n+1, de, hexs])

        with open(csv_file, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Number of Colors", "Delta E", "Colors"])
            for row in result:
                new_row = [row[0], row[1], *row[2]]
                writer.writerow(new_row)

    # sRGB_x_Japan2001Coated_CAM16
    result = [[0, 0, [None]]]
    for n in np.arange(1, 52, 1):
        csv_file = "sRGB_x_Japan2001Coated_CAM16.csv"
        hexs, de = multi_run(n+1, [1, 1, 1], color_space='CMYK', quality='medium', num_runs=32, show=False)
        hexs = [f"'{h}'" for h in hexs]
        result.append([n+1, de, hexs])

        with open(csv_file, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Number of Colors", "Delta E", "Colors"])
            for row in result:
                new_row = [row[0], row[1], *row[2]]
                writer.writerow(new_row)

    # # sRGB_DIN99d
    # result = [[0, 0, [None]]]
    # for n in np.arange(1, 52, 1):
    #     csv_file = "sRGB_DIN99d.csv"
    #     hexs, de = multi_run(n+1, [1, 1, 1], color_space='sRGB',
    #                          uniform_space='DIN99d', quality='medium', num_runs=32, show=False)
    #     hexs = [f"'{h}'" for h in hexs]
    #     result.append([n+1, de, hexs])
    #
    #     with open(csv_file, "w", newline='') as f:
    #         writer = csv.writer(f)
    #         writer.writerow(["Number of Colors", "Delta E", "Colors"])
    #         for row in result:
    #             new_row = [row[0], row[1], *row[2]]
    #             writer.writerow(new_row)
    #
    # # sRGB_x_Japan2001Coated_DIN99d
    # result = [[0, 0, [None]]]
    # for n in np.arange(1, 52, 1):
    #     csv_file = "sRGB_x_Japan2001Coated_DIN99d.csv"
    #     hexs, de = multi_run(n+1, [1, 1, 1], color_space='CMYK',
    #                          uniform_space='DIN99d', quality='medium', num_runs=32, show=False)
    #     hexs = [f"'{h}'" for h in hexs]
    #     result.append([n+1, de, hexs])
    #
    #     with open(csv_file, "w", newline='') as f:
    #         writer = csv.writer(f)
    #         writer.writerow(["Number of Colors", "Delta E", "Colors"])
    #         for row in result:
    #             new_row = [row[0], row[1], *row[2]]
    #             writer.writerow(new_row)
    #
    # # sRGB_Oklab
    # result = [[0, 0, [None]]]
    # for n in np.arange(1, 52, 1):
    #     csv_file = "sRGB_Oklab.csv"
    #     hexs, de = multi_run(n+1, [1, 1, 1], color_space='sRGB',
    #                          uniform_space='Oklab', quality='medium', num_runs=32, show=False)
    #     hexs = [f"'{h}'" for h in hexs]
    #     result.append([n+1, de, hexs])
    #
    #     with open(csv_file, "w", newline='') as f:
    #         writer = csv.writer(f)
    #         writer.writerow(["Number of Colors", "Delta E", "Colors"])
    #         for row in result:
    #             new_row = [row[0], row[1], *row[2]]
    #             writer.writerow(new_row)
    #
    # # sRGB_x_Japan2001Coated_DIN99d
    # result = [[0, 0, [None]]]
    # for n in np.arange(1, 52, 1):
    #     csv_file = "sRGB_x_Japan2001Coated_Oklab.csv"
    #     hexs, de = multi_run(n+1, [1, 1, 1], color_space='CMYK',
    #                          uniform_space='Oklab', quality='medium', num_runs=32, show=False)
    #     hexs = [f"'{h}'" for h in hexs]
    #     result.append([n+1, de, hexs])
    #
    #     with open(csv_file, "w", newline='') as f:
    #         writer = csv.writer(f)
    #         writer.writerow(["Number of Colors", "Delta E", "Colors"])
    #         for row in result:
    #             new_row = [row[0], row[1], *row[2]]
    #             writer.writerow(new_row)
