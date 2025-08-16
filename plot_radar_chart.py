import matplotlib.pyplot as plt
import numpy as np

# plt.figure(figsize=(9, 4))
# data = [0, 98.2, 97.25, 99.21, 98.74, 99.20, 99.31, 99.55, 99.60]
# labels = ["", "DeiT\n@448$^2$", "MAE\n@224$^2$", "D-iGPT\n@224$^2$", 'MOCOv3\n@224$^2$',
#           'DINO\n@224$^2$', 'iBOT\n@224$^2$', 'DINOv2\n@392$^2$', 'DINOv2-R\n@392$^2$']
# # colors = ["royalblue", "green", 'green', 'green', 'green', 'green', "green", 'green', 'green', 'green']
# colors = ["gray", "royalblue", 'royalblue', 'royalblue', 'royalblue', 'royalblue',
#           "royalblue", 'royalblue', 'royalblue', 'royalblue']
#
# ax = plt.gca()
# ax.set_ylim(96, 100)
# plt.tick_params(left=True, right=False, labelleft=True,
#                 labelbottom=True, bottom=False)
#
# plt.bar(range(len(data)), data, color=colors, width=0.9, alpha=0.5)
# plt.xticks(range(len(data)), labels)
# plt.ylabel("Image AUROC")
# plt.grid(linestyle='-', axis='y', alpha=0.5)
# plt.show()

#
# plt.figure(figsize=(8, 5))
# data = [93.1, 90.95, 92.34, 92.83, 93.48, 93.33, 94.83, 94.79]
# labels = ["MambaAD", "DeiT\n@448$^2$", "MAE\n@224$^2$", "D-iGPT\n@224$^2$", 'MOCOv3\n@224$^2$',
#           'DINO\n@224$^2$', 'iBOT\n@224$^2$', 'DINOv2\n@392$^2$', 'DINOv2-R\n@392$^2$']
#
# colors = ["royalblue", "green", 'green', 'green', 'green', 'green', "green", 'green', 'green', 'green']
#
# ax = plt.gca()
# ax.set_ylim(90, 95)
# plt.tick_params(left=True, right=False, labelleft=True,
#                 labelbottom=True, bottom=False)
#
# plt.bar(range(len(data)), data, color=colors, width=0.85)
# plt.xticks(range(len(data)), labels)
# plt.ylabel("Pixel AUPRO")
# plt.show()


# fig, ax1 = plt.subplots(figsize=(9, 4))
# # 设置第二个y轴
# ax2 = ax1.twinx()
#
# data = [0, 98.2, 97.25, 99.21, 98.74, 99.20, 99.31, 99.55, 99.60]
# data2 = [0, 0, 68.0, 80.5, 76.7, 78.2, 79.5, 84.5, 84.8]
#
# labels = ["", "DeiT\n@448$^2$", "MAE\n@224$^2$", "D-iGPT\n@224$^2$", 'MOCOv3\n@224$^2$',
#           'DINO\n@224$^2$', 'iBOT\n@224$^2$', 'DINOv2\n@392$^2$', 'DINOv2-R\n@392$^2$']
#
# bars1 = ax1.bar(np.arange(len(data)), data, 0.9, alpha=0.5, color='royalblue')
# bars2 = ax2.bar(np.arange(len(data)), data2, 0.9, alpha=0.5, color='gray')
#
# ax1.set_ylim(96, 100)
# ax2.set_ylim(60, 100)
#
# # plt.tick_params(left=True, right=False, labelleft=True,
# #                 labelbottom=True, bottom=False)
#
# plt.xticks(range(len(data)), labels)
# # plt.ylabel("Image AUROC")
# # plt.grid(linestyle='-', axis='y', alpha=0.5)
# plt.show()

""""
Scaling
"""
# Data
ACC = [99.26, 99.60, 99.70]
GMACs = [26.3, 104.7, 413.5]
parameters = [37.4, 148.0, 275.3]

# Create the scatter plot
plt.figure(figsize=(9, 5))

# Scale the parameters for better visualization (adjust the scaling factor as needed)
scale_factor = 40
sizes = [p * scale_factor for p in parameters]

# Create scatter plot with varying marker sizes
scatter = plt.scatter(GMACs, ACC, s=sizes, alpha=0.6, c='blue')

# Customize the plot
plt.xlabel('Computation Cost (GMACs)', fontsize=12)
plt.ylabel('AUROC (%)', fontsize=12)
# plt.title('Model Performance: Accuracy vs Computation Cost\nMarker size represents number of parameters (M)', fontsize=14)

# Add grid for better readability
plt.grid(True, linestyle='--', alpha=0.7)

# Annotate each point with its parameter value
# for i, param in enumerate(parameters):
#     plt.annotate(f'{param}M params',
#                  (GMACs[i], ACC[i]),
#                  xytext=(10, 10),
#                  textcoords='offset points',
#                  fontsize=10)

# Adjust layout and axis limits
plt.margins(x=0.1)
plt.ylim(99, 100)

# Show the plot
plt.tight_layout()
plt.show()

""""
Scaling
"""
# Data
ACC = [65.21, 67.22, 69.29, 69.24]
GMACs = [53.7, 77.1, 104.7, 136.4]
isize = [280, 336, 392, 448]

# Create the scatter plot
plt.figure(figsize=(9, 5))

# Scale the parameters for better visualization (adjust the scaling factor as needed)
scale_factor = 0.05
sizes = [p * p * scale_factor for p in isize]

# Create scatter plot with varying marker sizes
scatter = plt.scatter(GMACs, ACC, s=sizes, alpha=0.6, c='green')

# Customize the plot
plt.xlabel('Computation Cost (GMACs)', fontsize=12)
plt.ylabel('Pixel AP (%)', fontsize=12)
# plt.title('Model Performance: Accuracy vs Computation Cost\nMarker size represents number of parameters (M)', fontsize=14)

# Add grid for better readability
plt.grid(True, linestyle='--', alpha=0.7)

# Adjust layout and axis limits
plt.margins(x=0.1)
plt.ylim(62, 72)

# Show the plot
plt.tight_layout()
plt.show()

"""
backbone
"""
plt.figure(figsize=(9, 5))

backbone = ["MAE", "D-iGPT", 'MOCOv3', 'DINO', 'iBOT', 'DINOv2', 'DINOv2-R']
ACC = [97.25, 99.21, 98.74, 99.20, 99.31, 99.55, 99.60]
LP = [68.0, 80.5, 76.7, 78.2, 79.5, 84.5, 84.8]
scatter = plt.scatter(LP, ACC, s=100, alpha=0.8, c='red')

plt.xlabel('ImageNet LP', fontsize=12)
plt.ylabel('AUROC', fontsize=12)
# plt.title('Model Performance: Accuracy vs Computation Cost\nMarker size represents number of parameters (M)', fontsize=14)

for i, net in enumerate(backbone):
    plt.annotate(net,
                 (LP[i], ACC[i]),
                 xytext=(10, 10),
                 textcoords='offset points',
                 fontsize=10)

# Add grid for better readability
plt.grid(True, linestyle='--', alpha=0.7)
plt.ylim(97, 100)

plt.tight_layout()
plt.show()


"""
backbone
"""
plt.figure(figsize=(9, 5))

backbone = ["MAE", "D-iGPT", 'MOCOv3', 'DINO', 'iBOT', 'DINOv2', 'DINOv2-R']
ACC = [97.25, 99.21, 98.74, 99.20, 99.31, 99.55, 99.60]
LP = [68.0, 80.5, 76.7, 78.2, 79.5, 84.5, 84.8]
scatter = plt.scatter(LP, ACC, s=100, alpha=0.8, c='red')

plt.xlabel('ImageNet LP', fontsize=12)
plt.ylabel('AUROC', fontsize=12)
# plt.title('Model Performance: Accuracy vs Computation Cost\nMarker size represents number of parameters (M)', fontsize=14)

for i, net in enumerate(backbone):
    plt.annotate(net,
                 (LP[i], ACC[i]),
                 xytext=(10, 10),
                 textcoords='offset points',
                 fontsize=10)

# Add grid for better readability
plt.grid(True, linestyle='--', alpha=0.7)
plt.ylim(97, 100)

plt.tight_layout()
plt.show()
