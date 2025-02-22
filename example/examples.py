from mdcolors import single_run

# Just run
single_run(9)

# More colors
# colors, de = single_run(100)
# print(f"ΔE_min: {de}")
# print(f"Colors: \n{colors}")

# Constrained in CMYK space
# single_run(16, color_space='CMYK')

# Using DIN99d as Uniform-Color-Space
# single_run(9, uniform_space='DIN99d')

# Using Oklab as Uniform-Color-Space. Use concave hull for better results, but slower
# single_run(9, uniform_space='Oklab', hull_type='concave')

# Spend more time on higher quality, 'fast'->'medium'->'slow'
# single_run(9, quality='medium')

# Specify given colors
# single_run(9, given_colors=[[1, 1, 1], [0, 0, 0]])
