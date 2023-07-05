import matplotlib.pyplot as plt

data = [
    [3312, 0.00326172262793546, [0.1844, 0.993], 0.0032001986728229515],
    [8263, 0.026779505762861894, [0.1582, 0.9798], 0.02601446945616992],
    [5100, 0.029980562476618906, [0.2111, 0.9946], 0.029445493748564866],
    [1842, 0.04303549577833043, [0.1819, 0.9505], 0.02966330909945973],
    [2284, 0.05914726082581786, [0.1272, 0.9693], 0.058151010445953055]
]


# data = [
#     [3312, [0.1844, 0.993], 0.00326172262793546], 
#     [5233, [0.1781, 0.998], 0.005429074173440355],
#     [247, [0.1822, 0.9861], 0.007506607180706293],
#     [8286, [0.1838, 0.9787], 0.015013911002986826],
#     [6626, [0.1989, 0.9959], 0.017902269307161704]
# ]


# Extracting the radius and center values
radius = [item[1] for item in data]
center_x = [item[2][0] for item in data]
center_y = [item[2][1] for item in data]

# Plotting the circles and center points
fig, ax = plt.subplots()
for i in range(len(data)):
    circle = plt.Circle((center_x[i], center_y[i]), radius[i], fill=False)
    ax.add_patch(circle)
    ax.scatter(center_x[i], center_y[i], color='red', marker='o')

# Setting the plot limits
# ax.set_xlim(0.0, 0.3)
# ax.set_ylim(0.8, 1.1)
ax.autoscale()


# Adding labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Circle Plot')

# Displaying the plot
plt.show()

