import math, os, cv2
import numpy as np

def img_save(filename, img):
    cv2.imwrite(filename, img)

def generate_noisy_image(img, noise='gaussian', mean=0.0, sigma=128, scale=0.5):
    if noise == 'gaussian':
        noise = np.random.normal(mean, sigma/255.0, img.shape)
    elif noise == 'gamma':
        noise = np.random.gamma(shape=0.2, scale=scale, size=img.shape)
    elif noise == 'exponential':
        noise = np.random.exponential(scale=scale, size=img.shape)
    elif noise == 'rayleigh':
        noise = np.random.rayleigh(scale=scale, size=img.shape)

    noisy = img + noise * sigma
    return noisy

def generate_data_train(train_path, directory):
    """
    Para cada arquivo vamos criar quatro arquivos e 
    jogar dentro das pastas gaussian, gamma, rayleigh e exponential.
    """

    path = os.path.join(train_path, directory)
    
    noises = ['gamma', 'rayleigh', 'exponential', 'gaussian']
    
    files = os.listdir(path)
    for f in files:
        img = cv2.imread(os.path.join(path, f), cv2.IMREAD_GRAYSCALE)
        
        for noise in noises:
            noisy = generate_noisy_image(img, noise=noise)
            img_save(os.path.join(train_path, noise, f'{f}'), noisy)

if __name__ == '__main__':
    #generate environments
    generate_data_train('images/Train', 'original')