import matplotlib.pyplot as plt
import numpy as np

GP_JUMPING_LPIPS = np.array([0.0097, 0.0096, 0.0108, 0.013, 0.0149, 0.0314])
HN_JUMPING_LPIPS = np.array([0, 0.0444, 0.0389, 0.0391, 0.0507, 0.0805])

GP_JUMPING_PSNR = np.array([38.12, 37.87, 36.95, 35.95, 35.4, 28.85])
HN_JUMPING_PSNR = np.array([0, 38.54, 37.35, 34.98, 34.07, 26.30])

GP_JUMPING_SSIM = np.array([0.9969, 0.9967, 0.9943, 0.9894, 0.9872, 0.9632])
HN_JUMPING_SSIM = np.array([0, 0.9952, 0.9939, 0.9898, 0.9881, 0.9506])

GP_BALLS_LPIPS = np.array([0.0081, 0.0095, 0.0091, 0.0103, 0.0132, 0.0319])
HN_BALLS_LPIPS = np.array([0, 0.0594, 0.0631, 0.0579, 0.0367, 0.0449])

GP_BALLS_PSNR = np.array([42.26, 40.8, 41.05, 39.28, 37.56, 28.88])
HN_BALLS_PSNR = np.array([0, 41.16, 41.82, 38.69, 39.38, 32.93])

GP_BALLS_SSIM = np.array([0.9971, 0.9963, 0.9964, 0.9945, 0.9915, 0.9737])
HN_BALLS_SSIM = np.array([0, 0.9913, 0.9946, 0.9880, 0.9909, 0.9757])

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
plt.show()