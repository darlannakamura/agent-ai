import numpy as np
import matplotlib.pyplot as plt 
import cv2
from skimage.restoration import denoise_wavelet, estimate_sigma
from skimage.metrics import peak_signal_noise_ratio, \
    structural_similarity as ssim

original = cv2.imread('images/Train/original/test_001.png', cv2.IMREAD_GRAYSCALE)

noisy = cv2.imread('images/Train/gaussian/test_001.png', cv2.IMREAD_GRAYSCALE)

print('PSNR Noisy: ', peak_signal_noise_ratio(original, noisy))
print('SSIM noisy: ', ssim(original, noisy, data_range=noisy.max() - noisy.min()))

# plt.imshow(noisy, cmap='gray')
# plt.show()

sigma_est = estimate_sigma(noisy, multichannel=False, average_sigmas=True)

denoise = denoise_wavelet(noisy, multichannel=False, convert2ycbcr=False,
                                method='VisuShrink', mode='soft',
                                sigma=sigma_est, rescale_sigma=True)

info = np.info(denoise) # Get the information of the incoming image type
denoise = denoise.astype(np.float64) / denoise.max()
denoise = 255 * denoise # Now scale by 255
denoise = denoise.astype(np.uint8)

print(denoise[:1])
print(denoise.min())
print(denoise.max())
print('PSNR Denoise: ', peak_signal_noise_ratio(original, denoise))
print('SSIM Denoise: ', ssim(original, denoise, data_range=denoise.max() - denoise.min()))

plt.imshow(denoise, cmap='gray', interpolation='nearest')
plt.show()