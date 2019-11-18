import numpy as np 
import cv2
import scipy.stats as st
from numpy.random import randn
from sklearn.preprocessing import StandardScaler
import math


IMAGE = 'images/2007_11_16_0_2.png'
SEGMENTED_IMAGE = 'images/2007_11_16_0_2SEG4.png'

distributions = ['norm', 'expon', 'gamma', 'rayleigh'] #norm is gaussian distribution


"""
Retorna quantos segmentos a imagem segmentada possui. Os segmentos serão retornados
com a intensidade dos pixels.
Exemplo: suponha que você tenha uma imagem segmentada com os pixels brancos e pretos.
Essa função irá retornar um array da forma: [0, 255]
"""
def get_segments(image):
    segments = []

    #se a imagem tem mais de 1-dimensão:
    if len(image.shape) > 1: 
        image = image.flatten()

    for pixel in image:
        if not pixel in segments:
            segments.append(pixel)

    segments = np.sort(segments)
    return segments

"""
Usa teste de normalidade de Shapiro-Wilki com alpha = 5% para 
determinar se uma distribuição é ou não normal.
"""
def is_gaussian(data):
    stats, p_value = st.shapiro(data)
    #interpret
    alpha = 0.1
    if p_value > alpha:
        print('Se parece com Gaussiano.')
    else: 
        print('Não é gaussiano.')
"""
Retirado de: https://pythonhealthcare.org/2018/05/03/81-distribution-fitting-to-data/

Retirado de: https://stackoverflow.com/questions/51894150/python-chi-square-goodness-of-fit-test-to-get-the-best-distribution/51895468

Lembrando que nós queremos que o teste do qui-quadrado seja o menor possível e
o teste do P-valor do KS seja >0.05.

"""
def get_best_distribution(data): 
    histo, bin_edges = np.histogram(data, bins='auto', normed=False)
    number_of_bins = len(bin_edges) - 1
    observed_values = histo
    print('DADOS:')
    print(data)

    results = []
    for dist_name in distributions:
        dist = getattr(st, dist_name)
        param = dist.fit(data) #retorna média e desvio padrão
        arg = param[:-2] #média e desvio padrão
        loc = param[-2] #média
        scale = param[-1] #desvio padrão

        cdf = getattr(st, dist_name).cdf(bin_edges, loc=loc, scale=scale, *arg)
        print('Cumulative distribution function:')
        print(cdf)
        expected_values = len(data) * np.diff(cdf)
        print('Expected Values:')
        print(expected_values)


        print('Observed values:')
        print(observed_values)

        #ddof será quantidade de cores dos pixels nos dados
        colors = []
        for pixel in data:
            if not pixel in colors:
                colors.append(pixel)

        print(str(len(colors)), 'COLORS.')

        c , p = st.chisquare(observed_values, expected_values, ddof=len(colors))
        
        if (not math.isnan(c)) and (not math.isnan(p)):
            results.append({
                'name': dist_name,
                'chi_square': c,
                'p_value': p
            })

    return results


def flatten(image):
    #se a imagem tem mais de 1-dimensão:
    if len(image.shape) > 1: 
        return image.flatten()
    else:
        print('ERRO!!')

def group_region_pixels(original, segmented):
    original = flatten(original)
    segmented = flatten(segmented)

    segments_group = get_segments(segmented)

    segments = {} #dict

    for s in segments_group:
        segments[s] = []

    for i in range(len(original)):
        pixel = original[i]
        segment = segmented[i] # o segmento que aquele pixel pertence
        segments[segment].append(pixel)

    return segments

if __name__ == '__main__':
    original = cv2.imread(IMAGE)
    segmented = cv2.imread(SEGMENTED_IMAGE)
    
    groups = group_region_pixels(original, segmented)
    
    for k,v  in groups.items():
        print('GROUP: ', k)
        print('LENGTH', str(len(v)))
        results = get_best_distribution(v)
        print(results)

