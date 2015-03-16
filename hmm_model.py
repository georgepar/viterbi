__author__ = 'george'

import numpy as np


class HmmModel:
    def __init__(self, obs_symbols, pi, b, T):
        self.__obs_symbols = obs_symbols
        self.__nstates = len(pi)
        self.__nsymbols = len(obs_symbols)
        self.__init_probs = np.array(pi)
        self.__trans_table = np.array(T)
        self.__emmission_probs = {obs_symbols[i]: np.array(b[i][:])
            for i in xrange(len(obs_symbols))}

    def log_probs(self):
        log_t = np.log10(self.trans_table)
        log_b = {o: np.log10(b_o) for o,b_o in self.emmission_probs.iteritems()}
        log_pi = np.log10(self.init_probs)
        return log_t, log_b, log_pi

    def viterbi_decode(self, o):
        N = len(o)
        a, b, pi = self.log_probs()
        a = np.transpose(a)
        delta = np.zeros((N, self.nstates))
        psi = np.zeros((N, self.nstates))
        q_star = np.zeros(N)
        delta[0, :] = pi[:] + b[o[0]][:]
        for t in xrange(1, N):
            delta_a = delta[(t-1), :] + a
            delta[t, :] = np.amax(delta_a, axis=1) + b[o[t]][:]
            psi[t, :] = np.argmax(delta_a, axis=1)
        p_star = np.amax(delta[(N-1), :])
        q_star[N-1] = np.argmax(delta[(N-1), :])
        for t in xrange(N-2, -1, -1):
            q_star[t] = psi[t+1, q_star[t+1]]
        delta, p_star, q_star = 10**delta, 10**p_star, q_star+1
        return delta, p_star, q_star

    @property
    def obs_symbols(self):
        return self.__obs_symbols

    @property
    def nstates(self):
        return self.__nstates

    @property
    def nsymbols(self):
        return self.__nsymbols

    @property
    def init_probs(self):
        return self.__init_probs

    @property
    def emmission_probs(self):
        return self.__emmission_probs

    @property
    def trans_table(self):
        return self.__trans_table


def main():
    b = [[0.5, 0.8, 0.25, 0.2],
         [0.5, 0.2, 0.75, 0.8]]
    T = [[0.25, 0.2, 0.3, 0.25],
         [0.2, 0.25, 0.3, 0.25],
         [0.4, 0.2, 0.2, 0.2],
         [0.25, 0.3, 0.2, 0.25]]
    pi = [0.25, 0.25, 0.25, 0.25]
    V, U = 'V', 'U'
    obs_syms = [V, U]
    o = [U, V, U, V, V, V, U, U, V, U]
    hmm = HmmModel(obs_syms, pi, b, T)
    delta, p_star, q_star = hmm.viterbi_decode(o)
    print delta
    print p_star
    print q_star

if __name__ == "__main__":
    main()
