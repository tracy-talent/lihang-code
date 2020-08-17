import numpy as np

class HiddenMarkov:
    def __init__(self, Q, V, A, B, PI):
        self.Q = Q  # 状态数组或者状态映射也可以，N
        self.V = np.array(V) # 观测集数组，包含数据集中所有可能的观测项，T
        self.A = np.array(A) # 状态转移概率分布矩阵，N*N
        self.B = np.array(B) # 观测概率分布矩阵，N*T
        self.PI = np.array(PI) # 初始状态概率分布数组，N
        
    def forward(self, O, logs=False):  # 使用前向算法，O为观察序列，logs用于控制是否输出计算过程
        N = len(self.Q)  #可能存在的状态数量
        M = len(O)  # 观测序列的大小
        self.alphas = np.zeros((N, M))  # 前向概率：alphas[i][j]表示t时刻部分观测序列为o1,o2,o3...,ot且状态为qi的概率
        T = M  # 有几个时刻，有几个观测序列，就有几个时刻
        for t in range(T):  # 遍历每一时刻，算出alpha值
            indexOfO = np.where(self.V == O[t])[0][0]  # 找出序列对应的索引
            for i in range(N):
                if t == 0:  # 计算初值
                    self.alphas[i][t] = self.PI[i] * self.B[i][indexOfO]  # P176（10.15）
                    if logs:
                        print('alpha1(%d)=p%db%db(o1)=%f' % (i, i, i, self.alphas[i][t]))
                else:
                    self.alphas[i][t] = np.dot(
                        [alpha[t - 1] for alpha in self.alphas],
                        [a[i] for a in self.A]) * self.B[i][indexOfO]  # 对应P176（10.16）
                    if logs:
                        print('alpha%d(%d)=sigma [alpha%d(i)ai%d]b%d(o%d)=%f' %
                              (t, i, t - 1, i, i, t, self.alphas[i][t]))
                        # print(alphas)
        P = np.sum([alpha[M - 1] for alpha in self.alphas])  # P176(10.17)
        if logs:
            print("P(O|lambda)=", end="")
            for i in range(N):
                print("%.3f+" % self.alphas[i][M - 1], end="")
            print("0=%.6f" % P)
        # alpha11 = pi[0][0] * B[0][0]    #代表a1(1)
        # alpha12 = pi[0][1] * B[1][0]    #代表a1(2)
        # alpha13 = pi[0][2] * B[2][0]    #代表a1(3)

    def backward(self, O, logs=False):  # 后向算法，O为观察序列,logs用于控制是否输出计算过程
        N = len(self.Q)  # 可能存在的状态数量
        M = len(O)  # 观测序列的大小
        self.betas = np.ones((N, M))  # 后向概率：时刻t状态为qi的条件下，从t+1到T的部分观测序列为ot+1,ot+2,...,oT的概率
        if logs:
            for i in range(N):
                print('beta%d(%d)=1' % (M, i))
        for t in range(M - 2, -1, -1):
            indexOfO = np.where(self.V == O[t + 1])[0][0]  # 找出序列对应的索引
            for i in range(N):
                self.betas[i][t] = np.dot(
                    np.multiply(self.A[i], [b[indexOfO] for b in self.B]),
                    [beta[t + 1] for beta in self.betas])
                realT = t + 1
                realI = i + 1
                if logs:
                    print(
                        'beta%d(%d)=sigma [a%djbj(o%d)beta%d(j)]=(' %
                        (realT, realI, realI, realT + 1, realT + 1),
                        end='')
                    for j in range(N):
                        print(
                            "%.3f*%.3f*%.3f+" % (self.A[i][j], self.B[j][indexOfO],
                                                 self.betas[j][t + 1]),
                            end='')
                    print("0)=%.6f" % self.betas[i][t])
        # print(betas)
        if logs:
            indexOfO = np.where(self.V == O[0])[0][0]
            P = np.dot(
                np.multiply(self.PI, [b[indexOfO] for b in self.B]),
                [beta[0] for beta in self.betas])
            print("P(O|lambda)=", end="")
            for i in range(N):
                print(
                    "%.3f*%.3f*%.3f+" % (self.PI[i], self.B[i][indexOfO], self.betas[i][0]),
                    end="")
            print("0=%.6f" % P)

    def viterbi(self, O, logs=False): # viterbi算法进行状态decode，O为观测序列，logs用于控制是否输出计算过程
        N = len(self.Q)  #可能存在的状态数量
        M = len(O)  # 观测序列的大小
        self.deltas = np.zeros((N, M)) # deltas[i][t]表示t时刻状态为qi的所有状态序列中的最大概率
        self.psis = np.zeros((N, M)) # psis[i][t]使t时刻状态为qi最大化的t-1时刻的状态
        I = np.zeros(M, dtype=np.int32)
        for t in range(M):
            realT = t + 1
            indexOfO = np.where(self.V == O[t])[0][0]  # 找出序列对应的索引
            for i in range(N):
                realI = i + 1
                if t == 0:
                    self.deltas[i][t] = self.PI[i] * self.B[i][indexOfO]
                    self.psis[i][t] = 0
                    if logs:
                        print('delta1(%d)=pi%d * b%d(o1)=%.3f * %.3f=%.6f' %
                              (realI, realI, realI, self.PI[i], self.B[i][indexOfO],
                               self.deltas[i][t]))
                        print('psis1(%d)=0' % (realI))
                else:
                    self.deltas[i][t] = np.max(
                        np.multiply([delta[t - 1] for delta in self.deltas],
                                    [a[i] for a in self.A])) * self.B[i][indexOfO]
                    if logs:
                        print(
                            'delta%d(%d)=max[delta%d(j)aj%d]b%d(o%d)=%.3f*%.3f=%.6f'
                            % (realT, realI, realT - 1, realI, realI, realT,
                               np.max(
                                   np.multiply([delta[t - 1] for delta in self.deltas],
                                               [a[i] for a in self.A])), self.B[i][indexOfO],
                               self.deltas[i][t]))
                    self.psis[i][t] = np.argmax(
                        np.multiply(
                            [delta[t - 1] for delta in self.deltas],
                            [a[i]
                             for a in self.A])) + 1  #由于其返回的是索引，因此应+1才能和正常的下标值相符合。
                    if logs:
                        print('psis%d(%d)=argmax[delta%d(j)aj%d]=%d' %
                              (realT, realI, realT - 1, realI, self.psis[i][t]))
        if logs:
            print(self.deltas)
            print(self.psis)
        I[M - 1] = np.argmax([delta[M - 1] for delta in self.deltas
                                 ]) + 1  #由于其返回的是索引，因此应+1才能和正常的下标值相符合。
        print('i%d=argmax[deltaT(i)]=%d' % (M, I[M - 1]))
        for t in range(M - 2, -1, -1):
            I[t] = self.psis[int(I[t + 1]) - 1][t + 1]
            print('i%d=psis%d(i%d)=%d' % (t + 1, t + 2, t + 2, I[t]))
        print("状态序列I：", I)
        return I
    
    def train(self, O, criterion=0.05, logs=False): 
        '''
        Baum-Welch无监督参数学习，EM算法进行训练，训练之前必须已计算前向forward和后向backward概率
        O为观察序列
        criterion为前后两次训练参数相差允许的最小值，用于控制迭代次数
        logs为真则会打印forward和backward详细的计算过程
        '''
        N = len(self.Q)
        M = len(O)
        xi = np.zeros((M - 1, N, N))
        gamma = np.zeros((M, N))
        done = False
        O_index = np.zeros(M, dtype=np.int32)
        for t in range(M):
            O_index[t] = np.where(self.V == O[t])[0][0]
        while not done:
            # 计算更新参数后的前向概率alphas和后向概率betas
            self.forward(O,logs=logs)
            self.backward(O,logs=logs)
            # EM算法的E step
            # 计算xi
            for t in range(M - 1):
                indexofO = O_index[t + 1]
                xi_divisor = np.dot([alpha[t] for alpha in self.alphas], 
                                    [np.dot(np.multiply(self.A[i], [b[indexofO] for b in self.B]), 
                                            [beta[t + 1] for beta in self.betas]) for i in range(N)])
                xi_dividend = np.array([self.alphas[i][t] * np.multiply(np.multiply(self.A[i], [b[indexofO] for b in self.B]), 
                                                  [beta[t + 1] for beta in self.betas]) for i in range(N)])
                xi[t] = xi_dividend / xi_divisor
            # 计算gamma
            for t in range(M):
                gamma[t] = np.multiply([alpha[t] for alpha in self.alphas], [beta[t] for beta in self.betas]) / np.dot(
                    [alpha[t] for alpha in self.alphas], [beta[t] for beta in self.betas])
            # EM算法的M step
            # 更新状态转移概率分布矩阵A
            new_A = np.zeros(self.A.shape)
            for i in range(N):
                new_A_divisor = np.sum([g[i] for g in gamma]) - gamma[M - 1][i]
                for j in range(N):
                    new_A[i][j] = np.sum([xit[i, j] for xit in xi]) / new_A_divisor
            # 更新观测概率分布矩阵B
            new_B = np.zeros(self.B.shape)
            for j in range(new_B.shape[0]):
                new_B_divisor = np.sum([g[j] for g in gamma])
                for k in range(new_B.shape[1]):
                    new_B[j][k] = np.sum([gamma[t][j] for t in range(M) if O_index[t] == k]) / new_B_divisor
            # 更新初始状态概率分布数组PI
            new_PI = np.zeros(self.PI.shape)
            new_PI = gamma[0]
            # 对比前后两次更新幅度
            if (np.max(np.abs(new_A - self.A)) < criterion 
                and np.max(np.abs(new_B - self.B)) < criterion 
                and np.max(np.abs(new_PI - self.PI)) < criterion):
                done = True
            self.A[:,:], self.B[:,:], self.PI[:] = new_A, new_B, new_PI

if __name__ == '__main__':
    Q = [1, 2, 3]
    V = ['红', '白']
    A = [[0.5, 0.2, 0.3], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]]
    B = [[0.5, 0.5], [0.4, 0.6], [0.7, 0.3]]
    O = ['红', '白', '红', '红', '白', '红', '白', '白']
    PI = [0.2, 0.3, 0.5]
    HMM = HiddenMarkov(Q, V, A, B, PI)
    print('forward:')
    HMM.forward(O, logs=True)
    print('\nbackward:')
    HMM.backward(O, logs=True)
    print('\nviterbi状态预测before train:')
    HMM.viterbi(O)
    print('\nBaum-Welch train:')
    HMM.train(O) 
    print('\nviterbi状态预测after train:')
    HMM.viterbi(O)