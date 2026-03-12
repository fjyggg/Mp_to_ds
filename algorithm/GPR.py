import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import value_and_grad, grad
from scipy.optimize import minimize
from autograd.misc.optimizers import adam
import matplotlib.pyplot as plt
import autograd.scipy.stats.multivariate_normal as mvn
from autograd.numpy.linalg import solve

np.random.seed(5)


class sgpr:
    def __init__(self, X, y, Type, likelihood_noise=0.1, restart=1):
        self.X = X
        self.y = y
        self.init_param = []
        self.param = []
        self.input_dim = np.shape(self.X)[1]
        self.input_num = np.shape(self.X)[0]
        self.likelihood_noise = likelihood_noise
        self.restart = restart
        self.type = Type
        self.cov_y_y = None
        self.beta = None

    def init_random_param(self):
        kern_length_scale = 0.01 * np.random.normal(size=self.input_dim) + 2
        if self.type == 'Sshape':
            kern_noise = 0.1 * np.random.normal(size=1)
        elif self.type == 'WShape':
            kern_noise = 0.02 * np.random.normal(size=1)
        elif self.type == 'Assembly_data1' or self.type == 'Assembly_data2' or self.type == 'Assembly_data':
            kern_noise = 0.1 * np.random.normal(size=1)
        else:
            kern_noise = 0.1 * np.random.normal(size=1)
        self.init_param = np.hstack((kern_noise, kern_length_scale))
        self.param = self.init_param.copy()
        # print("self.init_param is", self.init_param)

    def set_param(self, param):
        self.param = param.copy()
        self.cov_y_y = self.rbf(self.X, self.X, self.param) + self.likelihood_noise ** 2 * np.eye(self.input_num)
        self.beta = solve(self.cov_y_y, self.y)
        self.inv_cov_y_y = solve(self.cov_y_y, np.eye(self.input_num))

    def save_param(self, direction):
        np.savetxt(direction, self.param)

    def set_XY(self, X, y):
        self.X = X
        self.y = y
        self.input_dim = np.shape(self.X)[1]
        self.input_num = np.shape(self.X)[0]

    def build_objective(self, param):
        cov_y_y = self.rbf(self.X, self.X, param)
        cov_y_y = cov_y_y + self.likelihood_noise**2 * np.eye(self.input_num)
        out = - mvn.logpdf(self.y, np.zeros(self.input_num), cov_y_y)
        return out

    def train(self):
        max_logpdf = -1e20
        # cons = con((0.001, 10))
        for i in range(self.restart):
            self.init_random_param()
            result = minimize(value_and_grad(self.build_objective), self.init_param, jac=True, method='L-BFGS-B', tol=0.01)
            logpdf = -result.fun
            param = result.x
            if logpdf > max_logpdf:
                self.param = param
                max_logpdf = logpdf
        print(max_logpdf, self.param)
        # 提前计算，做预测时可用
        self.cov_y_y = self.rbf(self.X, self.X, self.param) + self.likelihood_noise**2 * np.eye(self.input_num)
        self.beta = solve(self.cov_y_y, self.y)
        self.inv_cov_y_y = solve(self.cov_y_y, np.eye(self.input_num))

    def rbf(self, x, x_, param):
        kern_noise = param[0]
        sqrt_kern_length_scale = param[1:]
        diffs = np.expand_dims(x / sqrt_kern_length_scale, 1) - np.expand_dims(x_ / sqrt_kern_length_scale, 0)
        return kern_noise**2 * np.exp(-0.5 * np.sum(diffs ** 2, axis=2))

    def predict_determined_input(self, inputs):  # 单维GP预测
        # inputs 是矩阵
        cov_y_f = self.rbf(self.X, inputs, self.param)
        mean_outputs = np.dot(cov_y_f.T, self.beta.reshape((-1, 1)))
        var = (self.param[0]**2 - np.diag(np.dot(np.dot(cov_y_f.T, self.inv_cov_y_y), cov_y_f))).reshape(-1, 1)
        return mean_outputs, var

    def print_params(self):
        print('final param is', self.param)

    def callback(self, param):
        # ToDo: add something you want to know about the training process
        pass

    def gradient2input(self, input):
        sqrt_kern_length_scale = self.param[1:]
        temp1 = np.dot(self.X - input, np.diag(1 / (sqrt_kern_length_scale**2)))
        cov_y_f = self.rbf(self.X, input.reshape(1, -1), self.param).reshape(-1)
        temp2 = (temp1.T * cov_y_f).T
        gradient = np.dot(self.beta.T, temp2)
        return gradient.reshape(-1)


# multi GP
class mgpr():
    def __init__(self, X, Y, Type, likelihood_noise=0.1, restart=1):
        '''
        :param X: 训练集中的X
        :param Y: 训练集中的Y,行数代表训练集样本个数，列数为标签维度，下面的例子标签为2维
        :param likelihood_noise: 见sgpr中的定义
        :param restart: 不用管
        '''
        self.X = X
        self.Y = Y
        self.type = Type
        self.param = []
        self.input_dim = np.shape(X)[1]
        self.input_num = np.shape(X)[0]
        self.output_dim = np.shape(Y)[1]
        self.likelihood_noise = np.zeros(self.output_dim) + likelihood_noise
        self.restart = restart

    def set_XY(self, X, Y):
        self.X = X
        self.Y = Y
        self.input_num = np.shape(X)[0]
        for i in range(self.output_dim):
            self.models[i].set_XY(self.X, self.Y[i])

    def set_param(self, param):
        self.create_models()
        self.param = param.copy()
        for i in range(self.output_dim):
            self.models[i].set_param(self.param[i])

    def save_param(self, direction):
        np.savetxt(direction, self.param)

    def create_models(self):
        self.models = []
        for i in range(self.output_dim):
            self.models.append(sgpr(self.X, self.Y[:, i], self.type ,likelihood_noise=self.likelihood_noise[i], restart=self.restart))

    def init_random_param(self):
        for i in range(self.output_dim):
            self.models[i].init_random_param()

    def train(self):
        self.create_models()
        self.init_random_param()
        for i in range(self.output_dim):
            print('training model ', i, '...')
            self.models[i].train()
            if i == 0:
                self.param = self.models[i].param.copy()
            else:
                self.param = np.vstack((self.param, self.models[i].param.copy()))

    def print_params(self):
        print('final param is', self.param)

    def predict_determined_input_robot(self, inputs):
        # mean_outputs0, var0 = self.models[0].predict_determined_input(inputs)
        # mean_outputs1, var1 = self.models[1].predict_determined_input(inputs)
        for i in range(self.output_dim):
            mean_outputi, vari = self.models[i].predict_determined_input(inputs)
            if i == 0:
                mean_outputs = mean_outputi
                vars = vari
            else:
                mean_outputs = np.hstack((mean_outputs, mean_outputi))
                vars = np.hstack((vars, vari))
        return mean_outputs, vars

    def predict_determined_input(self, inputs):
        mean_outputs0, var0 = self.models[0].predict_determined_input(inputs)
        mean_outputs1, var1 = self.models[1].predict_determined_input(inputs)
        mean_outputs = np.hstack((mean_outputs0, mean_outputs1))
        vars = np.hstack((var0, var1))
        input_dim = np.shape(inputs)[0]
        # print('input_dim is:', input_dim)
        if input_dim == 1:
            mean_outputs = mean_outputs.reshape(-1)
            vars = vars.reshape(-1)
        return mean_outputs, vars

    def inference(self, test_t, plt_handle=None):
        mean_outputs, Sigma = self.predict_determined_input(test_t.reshape(-1, 1))
        print(Sigma.shape)

        if plt_handle is not None:
            fig = plt_handle.figure(figsize=(13, 3))
            plt_handle.subplots_adjust(left=0.05, right=0.99, wspace=0.2, hspace=0.25, bottom=0.1, top=0.99)
            plt5 = fig.add_subplot(131)
            plt6 = fig.add_subplot(132)
            plt7 = fig.add_subplot(133)

            plt5.plot(test_t, mean_outputs[:,0], c='red', ls='--')
            plt5.scatter(self.X.ravel(), self.Y[:, 0], color='grey', s=25, alpha=0.7, label='训练数据')

            plt6.plot(test_t, mean_outputs[:,1], c='red', ls='--')
            plt6.scatter(self.X,self.Y[:,1], color='grey', s=25, alpha=0.7, label='训练数据')

            plt7.plot(mean_outputs[:,0], mean_outputs[:,1], c='red')
            plt7.scatter(self.Y[:, 0], self.Y[:,1], color='grey', s=25, alpha=0.7, label='训练数据')

            plt5.fill_between(test_t, mean_outputs[:, 0] - 2 * np.sqrt(Sigma[:,0]),
                              mean_outputs[:, 0] + 2 * np.sqrt(Sigma[:,0]), color='red', alpha=0.5)
            plt6.fill_between(test_t, mean_outputs[:, 1] - 2 * np.sqrt(Sigma[:, 1]),
                              mean_outputs[:, 1] + 2 * np.sqrt(Sigma[:, 1]), color='red', alpha=0.5)
            # for i in range(self.tras.shape[0]):
            #     plt7.plot(self.tras[i,1,:],self.tras[i,2,:],'grey',ls='--')

            # x_nearest_point_test = np.array([0.2,1.0])
            # x_nearest_point_computr = np.array([0.23716892, 0.94768688])
            #
            # plt.scatter(x_nearest_point_test[0],x_nearest_point_test[1])
            # plt.scatter(x_nearest_point_computr[0],x_nearest_point_computr[1])

        plt.show()

#     def find_nearest_point(self, xi):
#         """全局网格搜索"""
#         self.t_grid = np.arange(0.0, 1.0, 1.0/100)  # 这个到时候可以和外面的相统一
#         distances = []
#         mean_set = []
#         for t in self.t_grid:
#             dist, mean_outputs = self.mahalanobis_distance(xi, t)
#             # print(dist)
#             distances.append(dist)
#             mean_set.append(mean_outputs)
#
#         min_idx = np.argmin(distances)
#         min_mean_value = mean_set[min_idx]
#         dist_value = distances[min_idx]
#
#         return self.t_grid[min_idx], min_mean_value, dist_value
#
#     def mahalanobis_distance(self, xi, t):
#         """计算点xi在时间t处的马氏距离平方"""
#         mean_outputs, Sigma = selfpredict_determined_input(t.reshape(-1, 1), self.param)
#
#         diff = (xi - mean_outputs).reshape(1,-1)
#         cov = Sigma Omega + np.eye(2) * 1e-6  #(1,2,2)
#
#         # 计算马氏距离: (x-μ)^T Σ^{-1} (x-μ)
#         try:
#             inv_cov = np.linalg.inv(cov)
#             # dist = diff @ inv_cov @ diff.T
#             dist = diff @ diff.T
#         except np.linalg.LinAlgError:
#             # 如果矩阵奇异，使用伪逆
#             inv_cov = np.linalg.pinv(cov)
#             # dist = diff @ inv_cov @ diff.T
#             dist = diff @ diff.T
#         return dist, mean_outputs
# #
#     def compute_energy_field(self):
#         # 网格参数
#         x_min, x_max = -0.2, 1.2
#         y_min, y_max = -0.2, 1.2
#         resolution = 50  # 网格分辨率
#
#         # 生成网格
#         x = np.linspace(x_min, x_max, resolution)
#         y = np.linspace(y_min, y_max, resolution)
#         X, Y = np.meshgrid(x, y)
#
#         # 初始化能量矩阵
#         energy = np.zeros_like(X)
#         # 初始化能量矩阵
#         for i in range(resolution):
#             for j in range(resolution):
#                 point = np.array([X[i,j],Y[i,j]])
#
#                 _, _, dist_value = self.find_nearest_point(point)
#
#                 energy[i, j] = dist_value
#
#         # 绘制能量场
#         self._plot_energy_field(X, Y, energy)
#
#
#     def _plot_energy_field(sel,X, Y, energy):
#         # 创建图形
#         fig, axes = plt.subplots(1, 2, figsize=(12, 6))
#
#         # 1. 等高线图
#         ax1 = axes[0]
#         contour = ax1.contourf(X, Y, energy, levels=10)
#         # contour = ax1.contourf(X, Y, energy, levels=50, cmap='viridis')
#         # fig.colorbar(contour, ax=ax1, label='Energy Value')
#         ax1.clabel(contour, fontsize=8)   # 显示等高线的值
