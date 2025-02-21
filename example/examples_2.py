from mdcolors import multi_run

if __name__ == '__main__':
    # Run several times and choose the best one
    # multi_run(9, num_runs=20)

    # Constrained in CMYK space
    multi_run(9, color_space='CMYK')

    # Combine with quality parameters
    # multi_run(9, num_runs=20, quality='medium')

    # Using other Uniform-Color-Space
    # multi_run(9, unifrom_space='DIN99d', num_runs=20)
    # multi_run(9, unifrom_space='CIE Lab', num_runs=20)
    # multi_run(9, unifrom_space='Oklab', num_runs=20)

