from mdcolors import multi_run

if __name__ == '__main__':
    # Run more times and choose the best one
    multi_run(9, num_runs=20)

    # Combine with quality parameters
    # multi_run(9, num_runs=20, quality='medium')

