import numpy as np
from sklearn.preprocessing import StandardScaler

masks = np.load(
    r"C:\YaksiData\astrolglialAnalysis\tests\data\combined_mean_image_seg.npy",
    allow_pickle=True,
).item()["masks"]

selected_region = np.where(masks == 259)

selected_region = np.transpose(selected_region)
selected_region = np.flip(selected_region, axis=1)
selected_region = np.flip(selected_region, axis=0)
standart_scaler = StandardScaler()
selected_region = standart_scaler.fit_transform(selected_region)
# print(selected_region)
covariance = np.cov(selected_region.T)
pcs = np.linalg.eig(covariance)[1]
eigenvalues = np.linalg.eig(covariance)[0]
print("pcs: ", pcs)
print("eigenvalues: ", eigenvalues)
print(covariance)

covariance = 0
variancex = 0
variancey = 0

for data_point in selected_region:
    covariance += (data_point[0]) * (data_point[1])
    variancex += (data_point[0]) ** 2
    variancey += (data_point[1]) ** 2

covariance = covariance / len(selected_region)
variancex = variancex / len(selected_region)
variancey = variancey / len(selected_region)

print("covariance: ", covariance)
