import os, cv2
import numpy as np
from agent import Agent, Environment
import matplotlib.pyplot as plt 
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage import img_as_float

def fn(noise):
    NOISE = noise
    if NOISE == 'gaussian':
        filename = '30-11-2019 16-32-26.npy'
    elif NOISE == 'gamma':
        filename = '30-11-2019 17-54-59.npy'
    elif NOISE == 'exponential':
        filename = '01-12-2019 18-07-26.npy'
    elif NOISE == 'rayleigh':
        filename = '01-12-2019 18-34-34.npy'

    window_size = (3,3)
    path = os.path.join('images', 'Test', NOISE)

    files = os.listdir(path)
    files = files[:1] # só o cameraman

    agent = Agent(noise=NOISE, lr=0.333, df=0.333, expl=0.333)
    agent.load_q_table_from_file(filename)

    images = []

    for f in files:
        print('FILE: ', f)

        env = Environment(NOISE, f, 
            noise_path=os.path.join('images', 'Test', NOISE, f), 
            window_size=window_size)

        denoise = np.zeros(env.img.shape)
        mean = np.zeros(env.img.shape)
        median = np.zeros(env.img.shape)
        
        img_float = img_as_float(env.img)
        sigma_est = np.mean(estimate_sigma(img_float, multichannel=False))
        print(f"estimated noise standard deviation = {sigma_est}")

        patch_kw = dict(patch_size=3,      # 3x3 patches
                        patch_distance=6,  # 13x13 search area
                        multichannel=False)

        # slow algorithm
        nlm = denoise_nl_means(img_float, h=1.15 * sigma_est, fast_mode=False,
                                **patch_kw)

        print(denoise.shape)    
        
        for (x,y, window) in env.sliding_window():
            # print(x,y)

            # plt.imshow(window, cmap='gray')
            # plt.show()

            agent.state = env.reset()
            # print('agent state: ', agent.state)
                    
            agent.set_environment(window) 
            agent.update_position(x,y)

            action = agent.get_action()
            # print('action: ', action)
            # Média
            avg = np.mean(window)
            mean_window = np.full(fill_value=int(avg), shape=window.shape)
            mean[y:y + window_size[1], x:x + window_size[0]] = mean_window

            # Mediana
            med = np.median(window)
            median_window = np.full(fill_value=int(med), shape=window.shape)
            median[y:y + window_size[1], x:x + window_size[0]] = median_window

            new_window = agent.get_env_after_action_index(action)

            # plt.imshow(new_window, cmap='gray')
            # plt.show()

            denoise[y:y + window_size[1], x:x + window_size[0]] = new_window

            # plt.imshow(denoise, cmap='gray')
            # plt.show()


        image = {}

        original = cv2.imread(os.path.join('images', 'Test', 'original', f), cv2.IMREAD_GRAYSCALE)
        image['original'] = original

        noisy = cv2.imread(os.path.join('images', 'Test', NOISE, f), cv2.IMREAD_GRAYSCALE)
        image['noisy'] = noisy

        image['mean'] = mean
        image['median'] = median

        image['agent'] = denoise

        # images.append(image)
        return image
    # fig, ax = plt.subplots(3,2, figsize=(15,10))

    # ax[0,0].imshow(original, cmap='gray', vmin=0, vmax=255)
    # ax[0,0].set_title(f'Original\nPSNR: {round(psnr(original, original),2)}, SSIM: {round(ssim(original, original), 2)}')
    # ax[0,0].axis('off')
    
    # ax[0,1].imshow(noisy, cmap='gray', vmin=0, vmax=255)
    # ax[0,1].set_title(f'Noisy\nPSNR: {round(psnr(original, noisy), 2)}, SSIM: {round(ssim(original, noisy), 2)}')
    # ax[0,1].axis('off')

    # ax[1,0].imshow(denoise, cmap='gray', interpolation='nearest', vmin=0, vmax=255)
    # ax[1,0].set_title(f'Agent\nPSNR: {round(psnr(original, denoise),2)}, SSIM: {round(ssim(original, denoise), 2)}')
    # ax[1,0].axis('off')

    # ax[1,1].imshow(mean, cmap='gray', interpolation='nearest', vmin=0, vmax=255)
    # ax[1,1].set_title(f'Mean\nPSNR: {round(psnr(original, mean), 2)}, SSIM: {round(ssim(original, mean), 2)}')
    # ax[1,1].axis('off')

    # ax[2,0].imshow(median, cmap='gray', interpolation='nearest', vmin=0, vmax=255)
    # ax[2,0].set_title(f'Median\nPSNR: {round(psnr(original, median),2)}, SSIM: {round(ssim(original, median), 2)}')
    # ax[2,0].axis('off')

    # ax[2,1].imshow(nlm, cmap='gray', interpolation='nearest')
    # ax[2,1].set_title(f'NLM\nPSNR: {round(psnr(img_as_float(original), nlm),2)}, SSIM: {round(ssim(img_as_float(original), nlm), 2)}')
    # ax[2,1].axis('off')
    


    # plt.show()

fig, ax = plt.subplots(4,4, figsize=(15,10))

noises = ['gaussian', 'gamma', 'exponential', 'rayleigh']

for i, noise in enumerate(noises):
    img = fn(noise)
    ax[i,0].imshow(img['noisy'], cmap='gray', vmin=0, vmax=255)
    ax[i,0].set_title(f"{noise}\nPSNR: {round(psnr(img['original'], img['noisy']),2)}, SSIM: {round(ssim(img['original'], img['noisy']), 2)}")
    ax[i,0].axis('off')

    ax[i,1].imshow(img['agent'], cmap='gray', vmin=0, vmax=255)
    ax[i,1].set_title(f"PSNR: {round(psnr(img['original'], img['agent']),2)}, SSIM: {round(ssim(img['original'], img['agent']), 2)}")
    ax[i,1].axis('off')

    ax[i,2].imshow(img['mean'], cmap='gray', vmin=0, vmax=255)
    ax[i,2].set_title(f"PSNR: {round(psnr(img['original'], img['mean']),2)}, SSIM: {round(ssim(img['original'], img['mean']), 2)}")
    ax[i,2].axis('off')

    ax[i,3].imshow(img['median'], cmap='gray', vmin=0, vmax=255)
    ax[i,3].set_title(f"PSNR: {round(psnr(img['original'], img['median']),2)}, SSIM: {round(ssim(img['original'], img['median']), 2)}")
    ax[i,3].axis('off')

plt.show()
