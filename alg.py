import numpy as np
from scipy.optimize import minimize

class SP():

    def __init__(self, lamb, dim, alpha=1.0):

        self.lamb = lamb
        self.alpha = alpha

        self.cov = np.eye(dim) * self.lamb
        self.d = dim
        self.theta_hat = np.zeros(dim)
        
        self.last_cov = self.cov.copy()
        inv = np.linalg.inv(self.cov)
        self.invs = [inv]
        self.inv = inv
        self.X = []
        self.y = []
        self.X_offline = []
        self.updates = 0

    def pi(self, cs):
        index = np.random.choice(np.arange(len(self.invs)))
        inv = self.invs[index]

        values = []
        for c in cs:
            value = c.dot(inv).dot(c)
            values.append(value)

        return np.argmax(values)

    def update_offline(self, c):
        self.cov = self.cov + self.alpha * np.outer(c, c)
        logdet = np.linalg.slogdet(self.cov)[1]
        last_logdet = np.linalg.slogdet(self.last_cov)[1]
        log2 = np.log(2)

        if logdet >= log2 + last_logdet:
            self.last_cov = self.cov.copy()
            inv = np.linalg.inv(self.cov)
            self.invs.append(inv)
            self.updates += 1
        else:
            self.invs.append(self.invs[-1])

    def pi_offline(self, cs):
        cs = np.array(cs)
        inv = self.invs[-1]
        values = []
        for c in cs:
            value = c.dot(inv).dot(c)
            values.append(value)

        return np.argmax(values), values
    
    def update_sampled_data(self, c, r):
        self.X.append(c)
        self.y.append(r)

class SP_cost():

    def __init__(self, lamb, dim, cost_power=1, alpha=1.0):

        self.lamb = lamb
        self.alpha = alpha
        self.cost_power = cost_power

        self.cov = np.eye(dim) * self.lamb
        self.d = dim
        self.theta_hat = np.zeros(dim)
        
        self.last_cov = self.cov.copy()
        inv = np.linalg.inv(self.cov)
        self.invs = [inv]
        self.inv = inv
        self.X = []
        self.y = []
        self.X_offline = []
        self.updates = 0

    def pi(self, cs, cost_list):
        cs = np.array(cs)
        cost_list = np.array(cost_list)
        adjusted_cost_list = np.power(cost_list, self.cost_power)

        index = np.random.choice(np.arange(len(self.invs)))
        inv = self.invs[index]
        values = []
        values_over_cost = []
        for i in range(len(cs)):
            value = (cs[i].dot(inv).dot(cs[i])) ** 2
            value_over_cost = value/adjusted_cost_list[i]
            values.append(value)
            values_over_cost.append(value_over_cost)

        return np.argmax(values_over_cost)

    def update_offline(self, c):
        self.cov = self.cov + self.alpha * np.outer(c, c)
        logdet = np.linalg.slogdet(self.cov)[1]
        last_logdet = np.linalg.slogdet(self.last_cov)[1]
        log2 = np.log(2)

        if logdet >= log2 + last_logdet:
            self.last_cov = self.cov.copy()
            inv = np.linalg.inv(self.cov)
            self.invs.append(inv)
            self.updates += 1
        else:
            self.invs.append(self.invs[-1])

    def pi_offline(self, cs, cost_list):
        cs = np.array(cs)
        if cost_list is None:
            inv = self.invs[-1]
            values = []
            for c in cs:
                value = (c.dot(inv).dot(c)) ** 2
                values.append(value)
            return np.argmax(values), values, None
        else:
            cost_list = np.array(cost_list)
            adjusted_cost_list = np.power(cost_list, self.cost_power)

            inv = self.invs[-1]
            values = []
            values_over_cost = []
            for i in range(len(cs)):
                value = (cs[i].dot(inv).dot(cs[i])) ** 2
                value_over_cost = value/adjusted_cost_list[i]
                values.append(value)
                values_over_cost.append(value_over_cost)
            return np.argmax(values_over_cost), values, cost_list

    def update_sampled_data(self, c, r):
        self.X.append(c)
        self.y.append(r)
        
class HATCH():

    def __init__(self, lamb, dim, cost_power=1, alpha=1.0):

        self.lamb = lamb
        self.alpha = alpha
        self.cost_power = cost_power

        self.cov = np.eye(dim) * self.lamb
        self.d = dim
        self.theta_hat = np.zeros(dim)
        
        self.last_cov = self.cov.copy()
        inv = np.linalg.inv(self.cov)
        self.invs = [inv]
        self.inv = inv
        self.X = []
        self.y = []
        self.X_offline = []
        self.updates = 0

    def pi(self, cs, cost_list, remaining_budget, total_budget):
        cs = np.array(cs)
        cost_list = np.array(cost_list)
        adjusted_cost_list = np.power(cost_list, self.cost_power)

        index = np.random.choice(np.arange(len(self.invs)))
        inv = self.invs[index]
        values = []
        values_over_cost = []
        for i in range(len(cs)):
            value = (cs[i].dot(inv).dot(cs[i])) ** 2
            value_over_cost = value/adjusted_cost_list[i]
            values.append(value)
            values_over_cost.append(value_over_cost)

        max_index = np.argmax(values_over_cost)
        min_cost_index = np.argmin(cost_list)
        max_cost = cost_list[max_index]

        probability = (remaining_budget - max_cost) / total_budget

        if np.random.rand() < probability:
            return max_index
        else:
            return min_cost_index

    def update_offline(self, c):
        self.cov = self.cov + self.alpha * np.outer(c, c)
        logdet = np.linalg.slogdet(self.cov)[1]
        last_logdet = np.linalg.slogdet(self.last_cov)[1]
        log2 = np.log(2)

        if logdet >= log2 + last_logdet:
            self.last_cov = self.cov.copy()
            inv = np.linalg.inv(self.cov)
            self.invs.append(inv)
            self.updates += 1
        else:
            self.invs.append(self.invs[-1])

    def pi_offline(self, cs, cost_list, remaining_budget, total_budget):
        cs = np.array(cs)
        if cost_list is None:
            inv = self.invs[-1]
            values = []
            for c in cs:
                value = (c.dot(inv).dot(c)) ** 2
                values.append(value)
            return np.argmax(values), values, None
        else:
            cost_list = np.array(cost_list)
            adjusted_cost_list = np.power(cost_list, self.cost_power)

            inv = self.invs[-1]
            values = []
            values_over_cost = []
            for i in range(len(cs)):
                value = (cs[i].dot(inv).dot(cs[i])) ** 2
                value_over_cost = value/adjusted_cost_list[i]
                values.append(value)
                values_over_cost.append(value_over_cost)
        max_index = np.argmax(values_over_cost)
        min_cost_index = np.argmin(cost_list)
        max_cost = cost_list[max_index]

        probability = (remaining_budget - max_cost) / total_budget

        if np.random.rand() < probability:
            return max_index, values, cost_list
        else:
            return min_cost_index, values, cost_list

    def update_sampled_data(self, c, r):
        self.X.append(c)
        self.y.append(r)

class LinUCB(SP):

    def pi(self, cs):
        cs = np.array(cs)
        inv = self.inv
        rewards_hat = cs.dot(self.theta_hat)
        uncertainty_values = []
        for c in cs:
            value = c.dot(inv).dot(c)
            uncertainty_values.append(value)
        ub_list = np.array(rewards_hat) + np.array(uncertainty_values)
        lb_list = np.array(rewards_hat) - np.array(uncertainty_values)
        opt_lb = np.max(lb_list)
        final_value_list = uncertainty_values*(ub_list > opt_lb).astype(float)

        return np.argmax(final_value_list)

    def update(self, c, r):
        self.X.append(c)
        self.y.append(r)
        X = np.array(self.X)
        y = np.array(self.y)
        
        self.cov = self.cov + self.alpha * np.outer(c, c)
        logdet = np.linalg.slogdet(self.cov)[1]
        last_logdet = np.linalg.slogdet(self.last_cov)[1]
        log2 = np.log(2)

        if logdet >= log2 + last_logdet:
            self.last_cov = self.cov.copy()
            self.inv = np.linalg.inv(self.cov)
            b = X.T.dot(y)
            self.theta_hat = self.inv.dot(b)
            self.updates += 1

class LinUCB_cost(SP_cost):

    def pi(self, cs, cost_list):
        cs = np.array(cs)
        cost_list = np.array(cost_list)        
        inv = self.inv
        rewards_hat = cs.dot(self.theta_hat)
        uncertainty_values = []
        for c in cs:
            value = c.dot(inv).dot(c)
            uncertainty_values.append(value)
        ub_list = np.array(rewards_hat) + np.array(uncertainty_values)
        lb_list = np.array(rewards_hat) - np.array(uncertainty_values)
        opt_lb = np.max(lb_list)
        final_value_list = (uncertainty_values/cost_list)*(ub_list > opt_lb).astype(float)

        return np.argmax(final_value_list)

    def update(self, c, r):
        self.X.append(c)
        self.y.append(r)
        X = np.array(self.X)
        y = np.array(self.y)
        
        self.cov = self.cov + self.alpha * np.outer(c, c)
        logdet = np.linalg.slogdet(self.cov)[1]
        last_logdet = np.linalg.slogdet(self.last_cov)[1]
        log2 = np.log(2)

        if logdet >= log2 + last_logdet:
            self.last_cov = self.cov.copy()
            self.inv = np.linalg.inv(self.cov)
            b = X.T.dot(y)
            self.theta_hat = self.inv.dot(b)
            self.updates += 1


class LogisticTS:
    
    def __init__(self, lambda_, alpha, n_dim):
        
        self.lambda_ = lambda_; self.alpha = alpha
                
        self.n_dim = n_dim, 
        self.m = np.zeros(self.n_dim)
        self.q = np.ones(self.n_dim) * self.lambda_
        
        self.w = np.random.normal(self.m, self.alpha * (self.q)**(-1.0), size = self.n_dim)

    def loss(self, w, *args):
        X, y = args
        return 0.5 * (self.q * (w - self.m)).dot(w - self.m) + np.sum([np.log(1 + np.exp(-y[j] * w.dot(X[j]))) for j in range(y.shape[0])])

    def grad(self, w, *args):
        X, y = args
        return self.q * (w - self.m) + (-1) * np.array([y[j] *  X[j] / (1. + np.exp(y[j] * w.dot(X[j]))) for j in range(y.shape[0])]).sum(axis=0)

    def get_weights(self):
        return np.random.normal(self.m, self.alpha * (self.q)**(-1.0), size = self.n_dim)

    def fit(self, X, y):

        self.w = minimize(self.loss, self.w, args=(X, y), jac=self.grad, method="L-BFGS-B", options={'maxiter': 20, 'disp':False}).x
        self.m = self.w

        P = (1 + np.exp(1 - X.dot(self.m))) ** (-1)
        self.q = self.q + (P*(1-P)).dot(X ** 2)

    def predict_proba(self, X, mode):
        self.w = self.get_weights()
        
        if mode == 'sample':
            w = self.w # weights are samples of posteriors
        elif mode == 'expected':
            w = self.m # weights are expected values of posteriors
        else:
            raise Exception('mode not recognized!')
        
        proba = 1 / (1 + np.exp(-1 * X.dot(w)))
        return np.array([1-proba , proba]).T

    def pi(self, cs, mode):
        values = []
        for c in cs:
            value = self.predict_proba(c, mode)[1]
            values.append(value)

        return np.argmax(values)