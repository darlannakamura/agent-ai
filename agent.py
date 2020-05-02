import os, cv2, time, random, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio
from skimage.restoration import denoise_nl_means, denoise_bilateral, estimate_sigma
import seaborn as sns

IMG_PATH = os.path.join('images', 'Train')
BINS = 8

def argmax(arr):
    pos = 0

    all_negative = True
    for i in arr:
        if i >= 0:
            all_negative = False
    if all_negative:
        min = arr[0]
        for i, v in enumerate(arr):
            if v < min:
                min = v
                pos = i
    else:
        max = arr[0]
        for i, v in enumerate(arr):
            if v > max:
                max = v
                pos = i
    return pos

def scale(arr):
    """
    Equaliza de um intervalo para 0 a 255.
    """
    return ((arr - arr.min()) * (1/(arr.max() - arr.min()) * 255)).astype('uint8')

def is_str(*args):
    for inpt in args:
        assert type(inpt) == str

def is_float_between_0_1(*args):
    for inpt in args:
        assert type(inpt) == float and inpt <= 1.0 and inpt >= 0.0

"""
O agente irá atuar em cada patch de 3x3 da imagem ambiente.
O estado do agente será o histograma do seu respectivo patch.
Ele poderá realizar as seguintes ações: 
    - aplicar filtro non local means;
    - aplicar filtro da média;
    - aplicar filtro da mediana;
    - aumentar pixels +1;
    - diminuir pixels -1;
    - deixar todos os pixels 255 (brancos);
    - deixar todos os pixels 0 (pretos);
"""

class Agent:
    def __init__(self, noise, lr, df, expl):
        """
        - lr is Learning Rate (alpha)
        - df is Discount Factor (gamma)
        - expl is Exploration Factor (epsilon)
        """
        self.noise = noise
        
        self.lr = lr
        self.df = df
        self.expl = expl

        self.intuition = None

        self.all_negative = False # todos os elementos da Q-Table são negativos?

        self.action_history = []

        self.load_actions()
        self.load_states()
        self.load_q_table()

        self.position = [0,0]
        self.position_history = []

        is_str(self.noise)
        is_float_between_0_1(self.lr, 
            self.df, 
            self.expl)

        # self.print()

    def set_environment(self, environment):
        """
        O ambiente para o agente será a window em que ele está inserido.
        """
        self.environment = environment
    
    def load_q_table_from_file(self, filename):
        print('Loading q table from file:', filename)
        # data = pd.read_csv(os.path.join('q-tables', filename))
        # data = data.to_numpy()
        data = np.load(os.path.join('q-tables', filename))
        print(data.shape)
        self.q_table = data
        print('Done.')

    def load_q_table(self):
        """
        Tenho 5 ações e 
        (3x3)**6 = 531.441 estados, sendo 6 a quantidade de bins no histograma.
        totalizando 2.657.205 células possíveis.
        """
        self.q_table = np.zeros([self.states_size, len(self.actions)])

    def update_q_table(self, state, action, reward):
        self.q_table[state, action] = reward

    def load_actions(self):
        """
        0 - +42 nos pixels
        1 - -42 nos pixels
        2 - filtro da média
        3 - filtro da mediana
        4 - filtro preto
        """
        # self.actions = ['up', 'down', 'mean', 'median', 'black', 'white', 'nlm', 'troll']

        self.actions = ['mean', 'median', 'up', 'down', 'nothing', 'nlm'] # -> BOM RESULTADO
        # self.actions = ['mean', 'median', 'down', 'nothing', 'nlm']
    
        print(f'Actions: {len(self.actions)}')

    def load_states(self):
        # self.states_size = (3*3) ** BINS
        sz = ''
        for i in range(0,BINS):
            sz += '9' # 3x3 = 9
        
        self.states_size = int(sz) #999999
        print(f'Observed states: {self.states_size}')

    def get_action(self):
        """
        Get action from Q-Table.
        """
        # print(f'Q-Table no estado {self.state}')
        # print(self.q_table[self.state])
        all_zero = True
        for i in self.q_table[self.state]:
            if i != 0:
                all_zero = False
        
        if all_zero:
            self.all_zero = True
            # se ele tem uma intuição
            if self.intuition: 
                action = self.intuition
                self.random_action = False
                self.used_intuition = True
            else:
                action = random.randint(0, len(self.actions)-1)
                self.random_action = True
                self.used_intuition = False

        else:
            self.all_zero = False
            self.used_intuition = False

            self.all_negative = True
            for i in self.q_table[self.state]:
                if i >= 0:
                    self.all_negative = False
            
            action = argmax(self.q_table[self.state])

            self.random_action = False

        return action
    
    def get_env_after_action_index(self, action_index):
        act = self.actions[action_index]
        self.action_index = action_index
        self.action_history.append(act)

        return getattr(self, act)()

    def update_position(self, x,y):
        self.position_history.append(self.position)
        self.position = [x,y]


    def print(self):
        """
        Print agent's attributes.
        """
        print(f'AGENT {self.noise.lower()}:')
        print(f' - Learning Rate (LR):      {self.lr}')
        print(f' - Discount Factor (DF):    {self.df}')
        print(f' - Exploration Factor (EF): {self.expl}')
        print(f' - Position: ({self.position[0]}, {self.position[1]})')

    def print_environment(self):
        print(self.environment)

    def nlm(self):
        # print('Entrou no non local means')
        # estimate the noise standard deviation from the noisy image
        sigma_est = np.mean(estimate_sigma(self.environment, multichannel=False))
        # print(f"Estimated noise standard deviation = {sigma_est}")

        # slow algorithm
        # denoise = denoise_nl_means(self.environment, h=1.15 * sigma_est, fast_mode=False)
    
        denoise = denoise_bilateral(self.environment, win_size=3, sigma_spatial=sigma_est,
                multichannel=False)
        denoise = np.reshape(denoise, self.environment.shape)
        return scale(denoise)

    def up(self):
        arr = []
        for pixel in self.environment.flatten():
            pixel = pixel + 42
            if pixel > 255:
                pixel = 255

            arr.append(pixel)
        
        self.environment = np.array(arr, dtype='uint8').reshape(self.environment.shape)
        return self.environment
    
    def down(self):
        arr = []

        for pixel in self.environment.flatten():
            pixel = pixel - 42
            if pixel < 0:
                pixel = 0
            arr.append(pixel)
        
        self.environment = np.array(arr, dtype='uint8').reshape(self.environment.shape)

        return self.environment
    

    def mean(self):
        mean = np.mean(self.environment)
        return np.full(fill_value=int(mean), shape=self.environment.shape)

    def median(self):
        median = np.median(self.environment)
        return np.full(fill_value=int(median), shape=self.environment.shape)


    def white(self):
        return np.full(fill_value=255, shape=self.environment.shape)

    def black(self):
        return np.zeros(shape=self.environment.shape)

    def nothing(self):
        return self.environment

    def troll(self):
        return np.full(fill_value=-255, shape=self.environment.shape)

class Environment:
    def __init__(self, noise, filename, window_size=(3,3), noise_path=None, print=False):
        self.original_path = os.path.join(IMG_PATH, 'original', filename)
        if noise_path:
            self.noise_path = noise_path
        else:
            self.noise_path = os.path.join(IMG_PATH, noise, filename)

        self.action_index = 0

        self.load_ground_truth()
        self.load_environment()

        self.window_size = window_size

        self.history = []

        self.mse_history = []

        self.reward = None

    def load_ground_truth(self):
        self.original = cv2.imread(self.original_path, cv2.IMREAD_GRAYSCALE)

    def load_environment(self):
        self.img = cv2.imread(self.noise_path, cv2.IMREAD_GRAYSCALE)
        
    def convert_history_to_array(self, history):
        arr = []
        for data in history:
            d_arr = []
            for k,v in data.items():
                if type(v) is bool:
                    v = 'sim' if v == True else 'não'
                elif type(v) is float:
                    v = round(v,2)
                d_arr.append(v)
            arr.append(d_arr)
        return arr

    def render(self, agent, history, pause=False): 
        scale = 1
        clone = self.img.copy()
        cv2.rectangle(clone, (self.x, self.y), (x+self.window_size[0]*scale, y+self.window_size[1]*scale), (255, 0, 0), 2)
        
        fig, ax = plt.subplots(4,4, figsize=(15,10))
        
        ax[0,0].remove()
        ax[0,1].remove()
        ax[0,2].remove()
        gs = ax[0, 1].get_gridspec()

        axbig0 = fig.add_subplot(gs[0, :1])
        axbig0.axis('off')

        axbig0.set_title('Image')
        axbig0.imshow(clone, cmap='gray', interpolation='nearest', vmin=0, vmax=255)

        gs = ax[0,1].get_gridspec()
        axbig = fig.add_subplot(gs[0, 1:])
        axbig.axis('off')
        axbig.set_title('imagem original')
        axbig.imshow(self.original, cmap='gray', interpolation='nearest', vmin=0, vmax=255)
        
        rand_action = 'sim' if agent.random_action else 'não'

        text = f'Episódio: {self.epoch}\nEstado: {agent.state}\nPosição: ({agent.position[0]}, {agent.position[1]}) \nAção Escolhida: {agent.actions[action]}\nAção aleatória? { rand_action }\nReforço: {round(self.reward, 2)}\nPróximo estado: {agent.next_state}' 

        ax[0,3].text(0.5, 0.5, text , size=12, ha='center', va='center')
        ax[0,3].axis('off')

        ax[1,0].set_title('previous window')
        ax[1,0].axis('off')
        ax[1,0].imshow(self.previous_window, cmap='gray', interpolation='nearest', vmin=0, vmax=255)

        ax[1,1].set_title('window')
        ax[1,1].axis('off')
        ax[1,1].imshow(self.window, cmap='gray', interpolation='nearest', vmin=0, vmax=255)

        ax[1,2].set_title('original window')
        ax[1,2].axis('off')
        ax[1,2].imshow(self.original_window, cmap='gray', interpolation='nearest', vmin=0, vmax=255)

        ax[1,3].set_title('Histograma')
        ax[1,3].set_yticks(tuple(range(0,10, 1)))
        ax[1,3].set_xticks(self.bin_edges)
        ax[1,3].bar(self.bin_edges[:-1],self.hist, width=np.diff(self.bin_edges), ec='k', align='edge')
        
        ax[2,0].remove()
        ax[2,1].remove()
        ax[2,2].remove()
        ax[2,3].remove()
        gs = ax[2, 0].get_gridspec()

        axbig1 = fig.add_subplot(gs[2, :])
        axbig1.axis('off')
        plt.subplots_adjust(left=0.2, bottom=0.5, hspace=0.5)

        cell_text = self.convert_history_to_array(list(reversed(history))[:3])

        axbig1.set_title('Histórico')
        axbig1.table(cellText=cell_text,
            colLabels=tuple(history[0].keys()), loc='center')

        ax[3,0].remove()
        ax[3,1].remove()
        ax[3,2].remove()
        ax[3,3].remove()
        gs = ax[3, 0].get_gridspec()

        axbig2 = fig.add_subplot(gs[3, 0:])
        axbig2.axis('off')
        plt.subplots_adjust(left=0.2, bottom=0.5, hspace=0.5)

        cell_text = [
            list(agent.q_table[0]),
            list(agent.q_table[1]),
            list(agent.q_table[agent.state])
            ]
        
        axbig2.set_title('Q-Table')
        axbig2.table(cellText=cell_text, 
            rowLabels=['0', '1', str(agent.state)],
            colLabels=tuple(agent.actions), loc='center', 
            colWidths=[0.12 for i in range(0,len(agent.actions))]
        )

        fig.tight_layout()

        plt.show()
        
        time.sleep(0.025)

    def histogram(self):
        bins = tuple(i for i in range(0, 256, 256//BINS)) 
        bins = bins + (255,)
        hist, bin_edges = np.histogram(self.window.flatten(), bins=bins, density=False)
        
        self.hist = hist
        self.bin_edges = bin_edges
    
    def hist_to_index(self):
        return int(''.join(str(x) for x in self.hist))

    def reset(self):
        self.histogram()
        index = self.hist_to_index()
        return index

    def step(self, window, action):
        self.previous_window = self.window
        self.window = window
        self.action_index = action

        self.histogram()
        next_state = self.hist_to_index()

        reward = self.reward_function()

        self.prev_reward = self.reward

        if reward > 0: # se for uma recompensa, vamos multiplicar por 100
            self.reward = reward * 100
        else: 
            self.reward = reward

        done = self.is_done()

        if done:
            self.high_rewards[action] += 1 
            self.reward = 999999

        return next_state, reward, done

    def sliding_window(self, step_size=3):
        """
        Slide a window across the image
        From https://www.pyimagesearch.com/2015/03/23/sliding-windows-for-object-detection-with-python-and-opencv/    
        """

        window_size = self.window_size

        for y in range(0, self.img.shape[0], step_size):
            for x in range(0, self.img.shape[1], step_size):
                # yield the current window
                self.window = self.img[y:y + window_size[1], x:x + window_size[0]]
                
                self.x = x
                self.y = y
                yield (x, y, self.window)
    
    def is_done(self):
        return self.reward >= 30.0
           
    def reward_function(self):
        x, y, window_size = self.x, self.y, self.window_size
        original_window = self.original[y:y + window_size[1], x:x + window_size[0]]
        
        self.original_window = original_window
        self.window = self.window.astype(dtype=np.uint8)

        assert original_window.shape == self.window.shape
        assert original_window.dtype == self.window.dtype

        psnr = peak_signal_noise_ratio(original_window, self.window)
        
        if psnr < 20:
            psnr = - psnr

        return psnr
        
def grau_de_recompensa(x):
    if x >= 0 and x <= 10:
        return 'baixa'
    elif x > 10 and x <= 20:
        return 'médio'
    elif x > 20 and x <= 25:
        return 'alto'
    elif x > 25:
        return 'muito alto'

def show_percentages(percentages):
    percents = pd.DataFrame(percentages)
    sns.scatterplot(x="epoch", y="p", hue="file", data=percents)
    plt.show()


def show_results(history, agent):
    df = pd.DataFrame(history)
    df['grau_de_recompensa'] = df['reward'].apply(lambda x: grau_de_recompensa(x))
    df['positivo'] = df['reward'].apply(lambda x: True if x > 0 else False)

    sns.countplot(x='action', hue='grau_de_recompensa', data=df).set_title('Grau de Recompensa')
    plt.show()

    sns.countplot(x='action', data=df[df['positivo'] == True]).set_title('Recompensas')
    plt.show()

    actions = agent.actions

    for act in actions:
        df['is_'+act] = df['action'].apply(lambda x: 1 if x == act else 0)

    for act in actions:
        df[act+'_count'] = df['is_'+act].cumsum()

    MAX = df.groupby(['action']).count()['epoch'].max()

    if 'mean' in actions:
        ax = df.plot(x='epoch', y='mean_count', legend=False, ylim=(0, MAX), color="purple")

    if 'white' in actions:
        ax2 = ax.twinx()
        ax2.axis('off')
        df.plot(x='epoch', y='white_count', ax=ax2, legend=False, color="y", ylim=(0, MAX))

    if 'black' in actions:
        ax3 = ax.twinx()
        ax3.axis('off')
        df.plot(x='epoch', y='black_count', ax=ax3, legend=False, color="black", ylim=(0, MAX))
    
    if 'up' in actions:
        ax4 = ax.twinx()
        ax4.axis('off')
        df.plot(x='epoch', y='up_count', ax=ax4, legend=False, color="blue", ylim=(0, MAX))

    if 'down' in actions:
        ax5 = ax.twinx()
        ax5.axis('off')
        df.plot(x='epoch', y='down_count', ax=ax5, legend=False, color="red", ylim=(0, MAX))

    if 'median' in actions:
        ax6 = ax.twinx()
        ax6.axis('off')
        df.plot(x='epoch', y='median_count', ax=ax6, legend=False, color="green", ylim=(0, MAX))

    if 'nlm' in actions:
        ax7 = ax.twinx()
        ax7.axis('off')
        df.plot(x='epoch', y='nlm_count', ax=ax7, legend=False, ylim=(0, MAX))

    if 'troll' in actions:
        ax8 = ax.twinx()
        ax8.axis('off')
        df.plot(x='epoch', y='troll_count', ax=ax8, legend=False, color="pink", ylim=(0, MAX))


    ax.figure.legend()
    plt.show()
    
if __name__ == '__main__':

    NOISE = 'gaussian'
    print('NOISE: ', NOISE)
    start = time.time()

    LIMIT_DECAY = 0.01 # 1%

    EACH_IMAGE = 20 # estava 50
    EACH_PATCH = 5 # estava 5
    epochs, penalties = 0, 0
    percentages = [] # porcentagens de quantas vezes escolheu uma ação e ganhou uma recompensa

    # noises = ['gamma', 'rayleigh', 'exponential', 'gaussian']
    files = os.listdir(os.path.join('images', 'Train', 'original'))
    # files = files[:20]
    files = files[:10]
    print(files)
    # for noise in noises:

    history = []

    for f in files:
        print('FILE: ', f)
        each_img = 0
        env = Environment(NOISE, f, window_size=(3,3))
        agent = Agent(noise=NOISE, lr=0.333, df=0.333, expl=0.333)
        
        env.high_rewards = [0 for i in range(0, len(agent.actions))]
        
        alpha = agent.lr #learning rate
        gamma = agent.df #discount factor
        epsilon = agent.expl #exploration or exploitation
         
        for q in tqdm(range(0, EACH_IMAGE),):
            penalties = 0
            randoms = 0
            all_zeros = 0
            all_negatives = 0
            intuicoes = 0

            for (x,y, window) in env.sliding_window():
                #decay
                if (epochs//1000) > 0:
                    alpha = alpha / (epochs//1000)
                    gamma = gamma / (epochs//1000)
                    epsilon = epsilon / (epochs//1000)
                    
                    # k = 0.1
                    # alpha = alpha * math.exp(-k*epochs)   
                    # gamma = gamma * math.exp(-k*epochs)
                    # epsilon = epsilon * math.exp(-k*epochs)

                if alpha < LIMIT_DECAY:
                    alpha = LIMIT_DECAY

                if gamma < LIMIT_DECAY:
                    gamma = LIMIT_DECAY

                if epsilon < LIMIT_DECAY:
                    epsilon = LIMIT_DECAY

                epochs += 1
                done = False
                env.reward = None #reseta a recompensa, pq mudou de ambiente.
                i = 0 
                
                agent.state = env.reset()
                
                agent.set_environment(window)
                agent.update_position(x,y)
                
                while not done and i < EACH_PATCH:
                    env.epoch = i
                    if random.uniform(0, 1) < epsilon:
                        agent.all_zero = False
                        agent.used_intuition = False
                        agent.random_action = True
                        action = random.randint(0, len(agent.actions)-1)
                    else:
                        agent.intuition = argmax(env.high_rewards)
                        action = agent.get_action()
                    
                    new_window = agent.get_env_after_action_index(action)
                    agent.set_environment(new_window)

                    next_state, reward, done = env.step(new_window, action)
                    
                    if reward < 0:
                        penalties += 1

                    if agent.random_action:
                        randoms += 1

                    if agent.used_intuition:
                        intuicoes += 1
                    
                    if agent.all_zero:
                        all_zeros += 1

                    if agent.all_negative:
                        all_negatives += 1
                    
                    history.append({
                        'epoch': epochs,
                        'episode': i,
                        'position': agent.position,
                        'state': agent.state,
                        'action': agent.actions[action],
                        'random_action': agent.random_action,
                        'reward': reward,
                        'next_state': next_state,
                        'done': done
                    })

                    agent.state = next_state

                    agent.next_state = next_state
                    agent.update_q_table(next_state, action, reward)
                    old_value = agent.q_table[agent.state, action]
                    next_max = np.max(agent.q_table[next_state])
                    new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
                    agent.update_q_table(agent.state, action, new_value)
                    env.render(agent, history, pause=True)
                    i += 1

            # print('PENALTIES: ', penalties)
            # tqdm.write('Recompensas:' + str( round( ( 1 - (penalties/ (3600 * EACH_PATCH)) ) * 100 , 2) ) + ' %' + ' | Randoms: ' + str(randoms) + ' | Tudo Zero: ' + str(all_zeros) + ' | Intuições: ' + str(intuicoes) + ' | Tudo negativo: ' + str( round( ( 1 - (all_negatives/ (3600 * EACH_PATCH)) ) * 100 , 2)) + '%')
            percentages.append({
                'epoch': each_img,
                'p': round( ( 1 - (penalties/ (3600 * EACH_PATCH)) * 100), 2),
                'randoms': randoms,
                'file': f
            })
            
            each_img += 1
    #fora do while

    end = time.time()
    print('Tempo total: ', end - start)

    st_writing = time.time()

    df = pd.DataFrame(history)
    now = datetime.now()
    datestring = now.strftime('%d-%m-%Y %H-%M-%S')
    filename = datestring + '.csv'
    df.to_csv(os.path.join('histories', filename), index=False, header=True)
    print('Gravou o histórico com sucesso.')

    df = pd.DataFrame(percentages)
    df.to_csv(os.path.join('percentages', filename), header=True)
    print('Gravou as porcentagens com sucesso.')

    # df = pd.DataFrame(agent.q_table)
    # df.to_csv(os.path.join('q-tables', filename), index=False, header=True)
    np.save(os.path.join('q-tables', datestring + '.npy'), agent.q_table)
    print('Gravou a Q-Table com sucesso.')

    end_writing = time.time()

    print('Tempo de gravação: ', end_writing - st_writing, ' seconds')

    show_results(history, agent)
    show_percentages(percentages)
