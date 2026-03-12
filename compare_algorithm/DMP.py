import numpy as np
import pyLasaDataset as lasa
import matplotlib.pyplot as plt
from MP_to_DS.algorithm.GPR import mgpr
from MP_to_DS.algorithm.GMR.mfc_gpr import middle_field_construct
import time
from trajectory_metrics import *



class SimpleDMP:
    """修正版DMP，确保收敛到目标"""

    def __init__(self, n_bfs=50, dt=0.001, alpha_z=25.0):
        self.n_bfs = n_bfs
        self.dt = dt
        self.alpha_z = alpha_z
        self.beta_z = alpha_z / 4.0
        self.alpha_x = 1.0

    def train(self, y_demo):
        """从演示轨迹中学习"""
        self.y0 = y_demo[0]
        self.goal = y_demo[-1]
        self.tau = len(y_demo) * self.dt

        # 基函数中心（在相位域均匀分布）
        self.c = np.exp(-self.alpha_x * np.linspace(0, self.tau, self.n_bfs) / self.tau)
        self.h = self.n_bfs / (self.c + 1e-10)

        # 计算演示的速度和加速度
        dy = np.gradient(y_demo, self.dt)
        ddy = np.gradient(dy, self.dt)

        # 计算目标强迫项 - 关键修正：除以 (goal - y0) 只在训练时使用
        f_target = np.zeros(len(y_demo))
        time_vals = np.arange(len(y_demo)) * self.dt

        for i, t in enumerate(time_vals):
            x = np.exp(-self.alpha_x * t / self.tau)
            # 标准DMP公式
            f_target[i] = (self.tau ** 2 * ddy[i] -
                           self.alpha_z * (self.beta_z * (self.goal - y_demo[i]) -
                                           self.tau * dy[i])) / (self.goal - self.y0 + 1e-10)

        # 学习权重 - 使用局部加权回归
        self.w = np.zeros(self.n_bfs)
        x_vals = np.exp(-self.alpha_x * time_vals / self.tau)

        for j in range(self.n_bfs):
            psi = np.exp(-self.h[j] * (x_vals - self.c[j]) ** 2)
            # 局部加权回归
            denom = np.sum(psi * x_vals ** 2) + 1e-10
            self.w[j] = np.sum(psi * x_vals * f_target) / denom

    def generate(self, goal=None, y0=None, tau=None):
        """生成运动 - 修正版确保收敛"""
        goal = goal if goal is not None else self.goal
        y0 = y0 if y0 is not None else self.y0
        tau = tau if tau is not None else self.tau

        n_steps = int(tau / self.dt) + 1  # +1确保包含终点
        y = np.zeros(n_steps)
        z = np.zeros(n_steps)
        y[0] = y0

        # 记录每步的强迫项用于调试
        f_history = []

        for i in range(n_steps - 1):
            t = i * self.dt
            x = np.exp(-self.alpha_x * t / tau)

            # 计算强迫项 - 使用学习到的权重
            psi = np.exp(-self.h * (x - self.c) ** 2)
            psi_sum = np.sum(psi)
            if psi_sum > 1e-10:
                psi = psi / psi_sum

            # 强迫项随相位衰减
            f = np.dot(self.w, psi) * x
            f_history.append(f)

            # 变换系统 - 关键：使用 (goal - y0) 缩放强迫项
            # 但当接近目标时，强迫项自然衰减（因为x -> 0）
            coupling_term = (goal - y0) * f

            # 加速度
            dz_dt = self.alpha_z * (self.beta_z * (goal - y[i]) - z[i]) + coupling_term
            dz = dz_dt / tau

            # 速度
            dy = z[i] / tau

            # 欧拉积分
            y[i + 1] = y[i] + dy * self.dt
            z[i + 1] = z[i] + dz * self.dt

        return np.linspace(0, tau, n_steps), y


# 使用示例
if __name__ == "__main__":
    # 导入时间序列轨迹N
    # Type = 'Sshape'
    # Type = 'WShape'
    Type = 'heee'

    data = getattr(lasa.DataSet, Type)
    demos = data.demos  # （7，）

    # 时间序列轨迹导入完成
    new_demo = []
    max_pos = demos[0].pos.max()
    min_pos = demos[0].pos.min()
    Demo_x = np.array([])
    Demo_y = np.array([])
    for i in range(len(demos)):
        pos = demos[i].pos  # shape: (2, n_points)
        t = demos[i].t  # 假设 shape: (1, n_points) 或 (n_points,)

        # 归一化
        t_normalized = (t - t.min()) / (t.max() - t.min())
        pos_normalized = (pos - min_pos) / (max_pos - min_pos)

        # 组合数据
        demo_data = np.vstack([t_normalized, pos_normalized])

        # 均匀采样但保留末端点
        n_samples = demo_data.shape[1]
        step = 10
        indices = list(range(0, n_samples - 1, step)) + [n_samples - 1]  # 确保包含最后一个点
        if i == 0:
            new_demo = demo_data[:, indices]
            Demo_x = new_demo[0, :].reshape(1, -1)
            Demo_y = new_demo[1:, :]

        else:
            new_demo = demo_data[:, indices]
            Demo_x = np.hstack((Demo_x, new_demo[0, :].reshape(1, -1)))
            Demo_y = np.hstack((Demo_y, new_demo[1:, :]))
    print(Demo_y.shape)
    Mgpr = mgpr(Demo_x.T, Demo_y.T, Type, likelihood_noise=0.01, restart=1)
    Mgpr.train()
    Mgpr.save_param('D:\computer document\essex_project\MP_to_DS\para/' + 'gp_para_'+ Type + '.txt')

    gp_para_org = np.loadtxt('D:\computer document\essex_project\MP_to_DS\para\gp_para_' + Type + '.txt')


    Mgpr.set_param(gp_para_org)

    t_grid = np.arange(0.0, 1.0, 1.0 / 1000)
    mu_grid, _ = Mgpr.predict_determined_input(t_grid.reshape(-1, 1))  # (100, 2)
    # print(mu_grid.shape)
    # print(t_grid[-1])

    n_bfs = 500
    dt = 0.001
    alpha_z = 50
    dmp1 = SimpleDMP(n_bfs, dt, alpha_z)
    dmp2 = SimpleDMP(n_bfs, dt, alpha_z)
    time1 = time.time()
    dmp1.train(mu_grid[:,0].reshape(-1))
    dmp2.train(mu_grid[:,1].reshape(-1))
    time2 = time.time()
    print('train time is:', time2-time1)

    print(mu_grid.shape)
    # 可视化
    plt.figure(figsize=(10, 5))


    # 存储每条轨迹的误差
    sea_errors = []
    dtw_errors = []
    dtw_norm_errors = []


    for i in range(7):
        y_init = Demo_y[:,0 + 101*i]

        t_gen1, y_gen1 = dmp1.generate(goal=mu_grid[-1,0], y0=y_init[0], tau=0.1)

        t_gen2, y_gen2 = dmp2.generate(goal=mu_grid[-1,1], y0=y_init[1], tau=0.1)


        # 组合成2D轨迹
        generated_traj = np.column_stack([y_gen1, y_gen2])
        demo_segment = Demo_y[:, 101*i+0:101*i+101].T  # 使用完整的演示轨迹作为参考

        print('generated_traj is:', generated_traj.shape)
        # 计算误差
        errors = compute_comprehensive_errors(generated_traj, demo_segment)

        sea_errors.append(errors['sea_error'])
        dtw_errors.append(errors['dtw_distance'])



        plt.plot(mu_grid[:,0], mu_grid[:,1], 'r--', label='demos traj.')
        plt.plot(y_gen1, y_gen2, 'black', label='gene traj.')
        plt.scatter(y_init[0], y_init[1],color='black')

    # 打印汇总统计
    print("SEA误差:", np.sum(sea_errors))
    print("DTW距离:", np.sum(dtw_errors))


    plt.scatter(mu_grid[-1,0], mu_grid[-1,1],color='black',marker='x',s=100,linewidths=2)
    # plt.legend()
    plt.grid(True)
    plt.title('DMP motion')
    plt.show()