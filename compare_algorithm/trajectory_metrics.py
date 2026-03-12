import numpy as np
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean


def compute_sea_error(generated_trajectory, demo_trajectory, num_scans=20):
    """
    计算SEA (Spatial Error Aggregation) 扫描误差

    参数:
        generated_trajectory: 生成的轨迹，形状 (n_points, 2)
        demo_trajectory: 演示轨迹，形状 (n_points, 2)
        num_scans: 扫描线数量

    返回:
        sea_error: SEA误差值
        scan_errors: 每个扫描线的误差
    """
    # 确定扫描范围
    x_min = min(demo_trajectory[:, 0].min(), generated_trajectory[:, 0].min()) - 0.05
    x_max = max(demo_trajectory[:, 0].max(), generated_trajectory[:, 0].max()) + 0.05
    y_min = min(demo_trajectory[:, 1].min(), generated_trajectory[:, 1].min()) - 0.05
    y_max = max(demo_trajectory[:, 1].max(), generated_trajectory[:, 1].max()) + 0.05

    # 生成扫描线（水平方向）
    scan_lines = np.linspace(y_min, y_max, num_scans)

    scan_errors = []

    for y_level in scan_lines:
        # 在y_level处找到演示轨迹上的交点
        demo_points_at_y = []
        for i in range(len(demo_trajectory) - 1):
            p1, p2 = demo_trajectory[i], demo_trajectory[i + 1]
            if (p1[1] <= y_level <= p2[1]) or (p2[1] <= y_level <= p1[1]):
                if abs(p2[1] - p1[1]) > 1e-10:
                    t = (y_level - p1[1]) / (p2[1] - p1[1])
                    x_interp = p1[0] + t * (p2[0] - p1[0])
                    demo_points_at_y.append(x_interp)

        # 在y_level处找到生成轨迹上的交点
        gen_points_at_y = []
        for i in range(len(generated_trajectory) - 1):
            p1, p2 = generated_trajectory[i], generated_trajectory[i + 1]
            if (p1[1] <= y_level <= p2[1]) or (p2[1] <= y_level <= p1[1]):
                if abs(p2[1] - p1[1]) > 1e-10:
                    t = (y_level - p1[1]) / (p2[1] - p1[1])
                    x_interp = p1[0] + t * (p2[0] - p1[0])
                    gen_points_at_y.append(x_interp)

        # 计算该扫描线的误差
        if demo_points_at_y and gen_points_at_y:
            demo_points_at_y = np.sort(demo_points_at_y)
            gen_points_at_y = np.sort(gen_points_at_y)

            # 使用最近邻匹配
            line_error = 0
            for x_demo in demo_points_at_y:
                min_dist = np.min(np.abs(np.array(gen_points_at_y) - x_demo))
                line_error += min_dist

            for x_gen in gen_points_at_y:
                min_dist = np.min(np.abs(np.array(demo_points_at_y) - x_gen))
                line_error += min_dist

            line_error /= (len(demo_points_at_y) + len(gen_points_at_y))
            scan_errors.append(line_error)
        elif demo_points_at_y:
            # 只有演示轨迹有交点
            scan_errors.append(np.mean(np.abs(demo_points_at_y)))
        elif gen_points_at_y:
            # 只有生成轨迹有交点
            scan_errors.append(np.mean(np.abs(gen_points_at_y)))
        else:
            scan_errors.append(0)

    sea_error = np.mean(scan_errors)
    return sea_error, scan_errors


def compute_dtw_error(trajectory1, trajectory2):
    """
    计算DTW (Dynamic Time Warping) 距离误差

    参数:
        trajectory1: 第一条轨迹，形状 (n1, 2)
        trajectory2: 第二条轨迹，形状 (n2, 2)

    返回:
        dtw_distance: DTW距离
        normalized_distance: 归一化DTW距离
        path: 对齐路径
    """
    # 使用fastdtw计算DTW距离
    distance, path = fastdtw(trajectory1, trajectory2, dist=euclidean)

    # 计算归一化DTW距离
    normalized_distance = distance / max(len(trajectory1), len(trajectory2))

    return distance, normalized_distance, path


def compute_comprehensive_errors(generated_trajectory, demo_trajectory):
    """
    计算全面的误差指标

    参数:
        generated_trajectory: 生成的轨迹，形状 (n_points, 2)
        demo_trajectory: 演示轨迹，形状 (n_points, 2)

    返回:
        errors_dict: 包含各种误差的字典
    """
    errors = {}

    # 1. SEA误差
    errors['sea_error'], errors['scan_errors'] = compute_sea_error(
        generated_trajectory, demo_trajectory, num_scans=20
    )

    # 2. DTW误差
    dtw_dist, dtw_norm, dtw_path = compute_dtw_error(
        demo_trajectory, generated_trajectory
    )
    errors['dtw_distance'] = dtw_dist
    errors['dtw_normalized'] = dtw_norm
    errors['dtw_path'] = dtw_path

    # 3. 均方根误差 (RMSE) - 使用DTW路径对齐
    aligned_demo = []
    aligned_gen = []

    for i, j in dtw_path:
        aligned_demo.append(demo_trajectory[i])
        aligned_gen.append(generated_trajectory[j])

    aligned_demo = np.array(aligned_demo)
    aligned_gen = np.array(aligned_gen)

    squared_diff = np.sum((aligned_demo - aligned_gen) ** 2, axis=1)
    rmse = np.sqrt(np.mean(squared_diff))
    errors['rmse'] = rmse

    # 4. 终点误差
    end_point_error = np.linalg.norm(
        demo_trajectory[-1] - generated_trajectory[-1]
    )
    errors['end_point_error'] = end_point_error

    # 5. 平均轮廓距离
    demo_tree = KDTree(demo_trajectory)
    gen_tree = KDTree(generated_trajectory)

    distances_demo_to_gen, _ = gen_tree.query(demo_trajectory)
    distances_gen_to_demo, _ = demo_tree.query(generated_trajectory)

    avg_contour_distance = (np.mean(distances_demo_to_gen) +
                            np.mean(distances_gen_to_demo)) / 2
    errors['avg_contour_distance'] = avg_contour_distance

    return errors


def plot_error_analysis(generated_trajectory, demo_trajectory, save_path=None):
    """
    绘制误差分析图
    """
    import matplotlib.pyplot as plt

    errors = compute_comprehensive_errors(generated_trajectory, demo_trajectory)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 1. 轨迹对比图
    ax = axes[0, 0]
    ax.plot(demo_trajectory[:, 0], demo_trajectory[:, 1],
            'b-', linewidth=2, label='Demonstration', alpha=0.7)
    ax.plot(generated_trajectory[:, 0], generated_trajectory[:, 1],
            'r--', linewidth=2, label='Generated', alpha=0.7)
    ax.scatter(demo_trajectory[0, 0], demo_trajectory[0, 1],
               color='green', s=100, marker='o', label='Start')
    ax.scatter(demo_trajectory[-1, 0], demo_trajectory[-1, 1],
               color='red', s=100, marker='s', label='Goal')
    ax.set_title(f'Trajectory Comparison\nEnd Error: {errors["end_point_error"] * 100:.2f}cm')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. SEA扫描误差图
    ax = axes[0, 1]
    ax.bar(range(len(errors['scan_errors'])), np.array(errors['scan_errors']) * 100)
    ax.set_xlabel('Scan Line Index')
    ax.set_ylabel('Error (cm)')
    ax.set_title(f'SEA Scan Errors\nMean SEA: {errors["sea_error"] * 100:.2f}cm')
    ax.grid(True, alpha=0.3)

    # 3. DTW对齐路径
    ax = axes[0, 2]
    dtw_path = np.array(errors['dtw_path'])
    ax.plot(dtw_path[:, 0], dtw_path[:, 1], 'b-', linewidth=1, alpha=0.5)
    ax.scatter(dtw_path[:, 0], dtw_path[:, 1], c='red', s=10)
    ax.set_xlabel('Demo Point Index')
    ax.set_ylabel('Generated Point Index')
    ax.set_title(f'DTW Alignment\nDTW: {errors["dtw_distance"]:.2f} (Norm: {errors["dtw_normalized"] * 100:.2f}cm)')
    ax.grid(True, alpha=0.3)

    # 4. 最近距离分布
    ax = axes[1, 0]
    demo_tree = KDTree(demo_trajectory)
    distances, _ = demo_tree.query(generated_trajectory)
    ax.hist(distances * 100, bins=30, alpha=0.7, edgecolor='black')
    ax.axvline(errors['avg_contour_distance'] * 100, color='red',
               linestyle='--', linewidth=2, label=f'Mean: {errors["avg_contour_distance"] * 100:.2f}cm')
    ax.set_xlabel('Distance to Demo Trajectory (cm)')
    ax.set_ylabel('Frequency')
    ax.set_title('Distance Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5. 点对点误差
    ax = axes[1, 1]
    _, _, dtw_path = compute_dtw_error(demo_trajectory, generated_trajectory)
    aligned_demo = demo_trajectory[[i for i, _ in dtw_path]]
    aligned_gen = generated_trajectory[[j for _, j in dtw_path]]

    pointwise_errors = np.linalg.norm(aligned_demo - aligned_gen, axis=1) * 100
    ax.plot(pointwise_errors, 'b-', linewidth=2)
    ax.fill_between(range(len(pointwise_errors)), pointwise_errors, alpha=0.3)
    ax.axhline(errors['rmse'] * 100, color='red', linestyle='--',
               linewidth=2, label=f'RMSE: {errors["rmse"] * 100:.2f}cm')
    ax.set_xlabel('Aligned Point Index')
    ax.set_ylabel('Error (cm)')
    ax.set_title('Point-wise Error Along Trajectory')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6. 误差指标汇总
    ax = axes[1, 2]
    ax.axis('off')
    metrics_text = f"""
    Error Metrics Summary:
    =======================
    SEA Error: {errors['sea_error'] * 100:.2f} cm
    DTW Distance: {errors['dtw_distance']:.2f}
    DTW Normalized: {errors['dtw_normalized'] * 100:.2f} cm
    RMSE: {errors['rmse'] * 100:.2f} cm
    End Point Error: {errors['end_point_error'] * 100:.2f} cm
    Avg Contour Distance: {errors['avg_contour_distance'] * 100:.2f} cm
    """
    ax.text(0.1, 0.5, metrics_text, fontsize=12, verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

    return errors