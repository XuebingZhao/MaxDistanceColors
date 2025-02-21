from mdcolors import single_run

# Just run
single_run(9)

# More colors
# single_run(100)

# Constrained in CMYK space
# single_run(9, color_space='CMYK')

# Using DIN99d as Uniform-Color-Space
# single_run(9, uniform_space='DIN99d')
# single_run(9, uniform_space='Oklab', hull_type='concave')  # Use concave hull for better results, but much slower

# Spend more time on higher quality, 'fast'->'medium'->'slow'
# single_run(9, quality='medium')

# Specify given colors
# single_run(9, given_colors=[[1, 1, 1], [0, 0, 0]])
