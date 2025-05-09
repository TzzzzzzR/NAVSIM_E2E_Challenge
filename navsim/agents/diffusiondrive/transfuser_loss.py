from typing import Dict
from scipy.optimize import linear_sum_assignment

import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig
from navsim.agents.diffusiondrive.transfuser_features import BoundingBox2DIndex


#多任务损失聚合
# GT信息存在target里面，输出信息存在prediction里面
def transfuser_loss(
    targets: Dict[str, torch.Tensor], predictions: Dict[str, torch.Tensor], config: TransfuserConfig
):
    """
    Helper function calculating complete loss of Transfuser
    :param targets: dictionary of name tensor pairings
    :param predictions: dictionary of name tensor pairings
    :param config: global Transfuser config
    :return: combined loss value
    """
    # transfuser：
    # trajectory_loss = F.l1_loss(predictions["trajectory"], targets["trajectory"])
    # agent_class_loss, agent_box_loss = _agent_loss(targets, predictions, config)
    # bev_semantic_loss = F.cross_entropy(predictions["bev_semantic_map"], targets["bev_semantic_map"].long())
    # loss = (
    #     config.trajectory_weight * trajectory_loss
    #     + config.agent_class_weight * agent_class_loss
    #     + config.agent_box_weight * agent_box_loss
    #     + config.bev_semantic_weight * bev_semantic_loss
    # )
    # return loss

    # diffusionDrive：
    # diffusionDrive在forward_train过程中计算了loss，所predict的trajectory里面包含键trajectory_loss
    # 这里的predictions就是V2TransfuserModel的output
    # predictions包含：{bev_semantic_map, trajectory, trajectory_loss, trajectory_loss_dict, agent_states, agent_labels}
    if "trajectory_loss" in predictions:
        trajectory_loss = predictions["trajectory_loss"]
    else:
        trajectory_loss = F.l1_loss(predictions["trajectory"], targets["trajectory"])   # diffuison直接用forward时计算的多轨迹损失替代原来的l1损失
    agent_class_loss, agent_box_loss = _agent_loss(targets, predictions, config)    # 车辆检测、BEV语义分割等辅助任务的loss
    bev_semantic_loss = F.cross_entropy(
        predictions["bev_semantic_map"], targets["bev_semantic_map"].long()
    )
    # 同上在forward_train过程中计算了diffusion_loss
    if 'diffusion_loss' in predictions:
        diffusion_loss = predictions['diffusion_loss']
    else:
        diffusion_loss = 0

    # lane_deviation_loss = _compute_lane_deviation_loss(predictions, targets, config)

    # loss聚合
    loss = (
        config.trajectory_weight * trajectory_loss
        + config.diff_loss_weight * diffusion_loss
        + config.agent_class_weight * agent_class_loss
        + config.agent_box_weight * agent_box_loss
        + config.bev_semantic_weight * bev_semantic_loss
        # + config.lane_deviation_weight * lane_deviation_loss
    )
    loss_dict = {
        'loss': loss,
        'trajectory_loss': config.trajectory_weight*trajectory_loss,
        'diffusion_loss': config.diff_loss_weight*diffusion_loss,
        'agent_class_loss': config.agent_class_weight*agent_class_loss,
        'agent_box_loss': config.agent_box_weight*agent_box_loss,
        'bev_semantic_loss': config.bev_semantic_weight*bev_semantic_loss,
        # 'lane_deviation_loss': config.lane_deviation_weight*lane_deviation_loss
    }
    print(loss_dict)
    if "trajectory_loss_dict" in predictions:
        trajectory_loss_dict = predictions["trajectory_loss_dict"]
        loss_dict.update(trajectory_loss_dict)
    # import ipdb; ipdb.set_trace()

    # loss_dict={loss:loss, trajectory_loss, diffuison_loss, agent_class_loss, agent_box_loss, bev_semantic_loss, trajectory_loss_0, trajectory_loss_1}
    return loss_dict



#-----------
# def _agent_future_traj_loss(
#     predictions: Dict[str, torch.Tensor], 
#     targets: Dict[str, torch.Tensor], 
#     config: TransfuserConfig
# ): # -> torch.Tensor:


#-----------


#-----------
def _compute_lane_deviation_loss(
    predictions: Dict[str, torch.Tensor], 
    targets: Dict[str, torch.Tensor], 
    config: TransfuserConfig
): # -> torch.Tensor:
    """
    计算轨迹预测与车道中心线的偏离损失
    :param predictions: 模型输出，包含预测轨迹 ["trajectory"]
    :param targets: 包含BEV语义地图 ["bev_semantic_map"]
    :param config: 配置参数（如阈值、惩罚系数）
    :return: 车道偏离损失（标量张量）
    """
    # eval_config:
    # lane_keeping_deviation_limit: float = 0.5  # [m] (lane keeping) (hydraMDP++)
    # lane_keeping_horizon_window: float = 2.0  # [s] (lane keeping) (hydraMDP++)

    # train_config:
    deviation_threshold: float = 1.0    # 偏离阈值（米）
    time_window: int = 5                # 连续检测窗口大小（轨迹点数）

    # 获取预测轨迹
    pred_trajectory = predictions["trajectory"][..., :2]  # 仅取XY坐标
    batch_size, num_poses, _ = pred_trajectory.shape
    lane_deviation_weight = 1

    print("pred_trajectory:", pred_trajectory)
    print("pred_trajectory shape:", pred_trajectory.shape)  # ([64, 8, 2]) 64个batch，8个时间步，2维度坐标
    
    # 从config文件中获取BEV参数
    bev_width = config.bev_pixel_width         # 256 
    bev_height = config.bev_pixel_height        
    resolution = config.bev_pixel_size          # 0.25米/像素
    
    # 生成BEV物理坐标网格 [H, W, 2]
    # x_coords表示BEV图像的X轴坐标，用0-256表示从 -32.0米 到 +31.75米的X轴坐标，Y轴同理
    x_coords = torch.linspace(start=-(bev_width // 2) * resolution, end=(bev_width // 2 - 1) * resolution, steps=bev_width, device=pred_trajectory.device)
    y_coords = torch.linspace(start=(bev_height // 2 - 1) * resolution,end=-(bev_height // 2) * resolution, steps=bev_height, device=pred_trajectory.device)
    xx, yy = torch.meshgrid(x_coords, y_coords, indexing='xy')  # xx:每个点的x坐标，yy:每个点的y坐标（例如：xx[0,0] = -32.0, yy[0,0] = +15.75 ，则指左上角点坐标）
    # 堆叠坐标形成每个点的(x,y)坐标，bev_coords[i,j] = [x_coords[j], y_coords[i]]（i，j ：0~256）
    bev_coords = torch.stack([xx, yy], dim=-1)  # bev_coords形状为 [H, W, 2]，表示每个像素的 (x, y) 物理坐标
    
    # 提取车道线掩码 (标签3)
    bev_semantic_map = targets["bev_semantic_map"]  # [batch_size, H, W]
    lane_mask = (bev_semantic_map == 3)             # [batch_size, H, W]
    
    # print("bev_semantic_map:", bev_semantic_map)
    # print("bev_semantic_map shape:", bev_semantic_map.shape)    # [64, 128, 256] 64个batch，垂直分辨率，水平分辨率，数值表示语义分割的类别
    # print("lane_mask:", lane_mask)
    # print("lane_mask shape:", lane_mask.shape)                  # [64, 128, 256] ,数值为0/1

    # #   BEV lane_mask可视化
    sample_map = bev_semantic_map[0].detach()
    sample_lane_mask = (sample_map == 3).to(torch.uint8)
    sample_map_np = sample_map.cpu().numpy()
    lane_mask_np = sample_lane_mask.cpu().numpy()
    plt.figure(figsize=(12, 6))
    # 显示原始语义地图
    plt.subplot(1, 2, 1)
    plt.imshow(sample_map_np, cmap='viridis', vmin=1, vmax=6)  # 类别范围1-6
    plt.title("BEV Semantic Map (All Classes)")
    plt.colorbar(ticks=[1, 2, 3, 4, 5, 6])
    # 显示车道线掩码
    plt.subplot(1, 2, 2)
    plt.imshow(lane_mask_np, cmap='gray')
    plt.title("Lane Mask (Class 3)")
    plt.tight_layout()
    plt.savefig("bev_visualization.png")
    plt.close()  # 关闭图像释放内存

    # 收集车道线点坐标 [batch, num_points, 2]
    lane_points = []
    max_points = 0
    for batch_idx in range(batch_size):
        # 当前样本的所有车道线点坐标
        points = bev_coords[lane_mask[batch_idx]]  # [num_points, 2] num_points个车道线点每个点是个2维坐标
        lane_points.append(points)
        num_points = points.shape[0]  # 当前样本的车道线点数
        max_points = max(max_points, num_points)   # 所有样本中的最大点数，以便统一尺寸
    
    # 填充为统一尺寸 [batch, max_points, 2]
    padded_lane_points = torch.zeros(
        (batch_size, max_points, 2), 
        device=pred_trajectory.device
    )
    for batch_idx, points in enumerate(lane_points):
        num_points = points.shape[0]
        padded_lane_points[batch_idx, :num_points] = points
    
    # 计算轨迹点到最近车道线点的距离 [batch, num_poses]
    dists = torch.cdist(pred_trajectory, padded_lane_points, p=2)  # [batch, num_poses, max_points]
    min_dists, _ = dists.min(dim=-1)  # [batch, num_poses]
    # cdist计算欧氏距离，可微
    
    # 计算超限部分的距离,每个轨迹点到最近车道线点的超限距离,小于0时设为0
    exceed_dist = F.relu(min_dists - deviation_threshold)  # [batch, num_poses]
    # relu函数，可微

    # 二进制超限掩码,每个轨迹点的超限情况（1表示超限，0表示未超限）
    binary_mask = (min_dists > deviation_threshold).float()  # [batch, num_poses]

    # 滑动窗口卷积检测连续超限
    kernel = torch.ones(time_window, device=binary_mask.device)  # 全1卷积核
    padded = F.pad(binary_mask, (time_window-1, 0), value=0)  # 左填充，保证卷积能覆盖起始点
    convolved = F.conv1d(padded.unsqueeze(1), kernel.view(1, 1, -1), groups=1).squeeze()  
    # 输出[batch, num_poses],每个位置的值表示该点及前time_window-1个点的超限次数

    # 连续超限掩码,每个轨迹点连续超限的情况（1表示超限，0表示未超限）
    continuous_mask = (convolved >= time_window).float() #  [batch, num_poses]
    # 滑动检测和连续超限中间计算不可导，但是这里也不需要反向传播

    # 计算总损失，只计算连续超限轨迹点的偏离
    # exceed_dist是可微的，continuous_mask不可微分但只是作为0/1掩码权重，不破坏反向传播路径
    loss = (exceed_dist * continuous_mask).mean() * lane_deviation_weight

    return loss
#------------------------

def _agent_loss(
    targets: Dict[str, torch.Tensor], predictions: Dict[str, torch.Tensor], config: TransfuserConfig
):
    """
    Hungarian matching loss for agent detection
    :param targets: dictionary of name tensor pairings
    :param predictions: dictionary of name tensor pairings
    :param config: global Transfuser config
    :return: detection loss
    """

    gt_states, gt_valid = targets["agent_states"], targets["agent_labels"]
    pred_states, pred_logits = predictions["agent_states"], predictions["agent_labels"]

    if config.latent:
        rad_to_ego = torch.arctan2(
            gt_states[..., BoundingBox2DIndex.Y],
            gt_states[..., BoundingBox2DIndex.X],
        )

        in_latent_rad_thresh = torch.logical_and(
            -config.latent_rad_thresh <= rad_to_ego,
            rad_to_ego <= config.latent_rad_thresh,
        )
        gt_valid = torch.logical_and(in_latent_rad_thresh, gt_valid)

    # save constants
    batch_dim, num_instances = pred_states.shape[:2]
    num_gt_instances = gt_valid.sum()
    num_gt_instances = num_gt_instances if num_gt_instances > 0 else num_gt_instances + 1

    ce_cost = _get_ce_cost(gt_valid, pred_logits)   #分类loss
    l1_cost = _get_l1_cost(gt_states, pred_states, gt_valid)    #回归loss

    cost = config.agent_class_weight * ce_cost + config.agent_box_weight * l1_cost
    cost = cost.cpu()

    indices = [linear_sum_assignment(c) for i, c in enumerate(cost)]
    matching = [
        (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
        for i, j in indices
    ]
    idx = _get_src_permutation_idx(matching)

    pred_states_idx = pred_states[idx]
    gt_states_idx = torch.cat([t[i] for t, (_, i) in zip(gt_states, indices)], dim=0)

    pred_valid_idx = pred_logits[idx]
    gt_valid_idx = torch.cat([t[i] for t, (_, i) in zip(gt_valid, indices)], dim=0).float()

    l1_loss = F.l1_loss(pred_states_idx, gt_states_idx, reduction="none")
    l1_loss = l1_loss.sum(-1) * gt_valid_idx
    l1_loss = l1_loss.view(batch_dim, -1).sum() / num_gt_instances

    ce_loss = F.binary_cross_entropy_with_logits(pred_valid_idx, gt_valid_idx, reduction="none")
    ce_loss = ce_loss.view(batch_dim, -1).mean()

    return ce_loss, l1_loss


@torch.no_grad()
def _get_ce_cost(gt_valid: torch.Tensor, pred_logits: torch.Tensor) -> torch.Tensor:
    """
    Function to calculate cross-entropy cost for cost matrix.
    :param gt_valid: tensor of binary ground-truth labels
    :param pred_logits: tensor of predicted logits of neural net
    :return: bce cost matrix as tensor
    """

    # NOTE: numerically stable BCE with logits
    # https://github.com/pytorch/pytorch/blob/c64e006fc399d528bb812ae589789d0365f3daf4/aten/src/ATen/native/Loss.cpp#L214
    gt_valid_expanded = gt_valid[:, :, None].detach().float()  # (b, n, 1)
    pred_logits_expanded = pred_logits[:, None, :].detach()  # (b, 1, n)

    max_val = torch.relu(-pred_logits_expanded)
    helper_term = max_val + torch.log(
        torch.exp(-max_val) + torch.exp(-pred_logits_expanded - max_val)
    )
    ce_cost = (1 - gt_valid_expanded) * pred_logits_expanded + helper_term  # (b, n, n)
    ce_cost = ce_cost.permute(0, 2, 1)

    return ce_cost


@torch.no_grad()
def _get_l1_cost(
    gt_states: torch.Tensor, pred_states: torch.Tensor, gt_valid: torch.Tensor
) -> torch.Tensor:
    """
    Function to calculate L1 cost for cost matrix.
    :param gt_states: tensor of ground-truth bounding boxes
    :param pred_states: tensor of predicted bounding boxes
    :param gt_valid: mask of binary ground-truth labels
    :return: l1 cost matrix as tensor
    """

    gt_states_expanded = gt_states[:, :, None, :2].detach()  # (b, n, 1, 2)
    pred_states_expanded = pred_states[:, None, :, :2].detach()  # (b, 1, n, 2)
    l1_cost = gt_valid[..., None].float() * (gt_states_expanded - pred_states_expanded).abs().sum(
        dim=-1
    )
    l1_cost = l1_cost.permute(0, 2, 1)
    return l1_cost


def _get_src_permutation_idx(indices):
    """
    Helper function to align indices after matching
    :param indices: matched indices
    :return: permuted indices
    """
    # permute predictions following indices
    batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
    src_idx = torch.cat([src for (src, _) in indices])
    return batch_idx, src_idx


# 车道保持损失
def _lane_keeping_loss(
    predictions: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    config: TransfuserConfig
):
    """
    1. 计算预测轨迹到车道线的横向距离
    2. 动态累积连续超限时间步的惩罚
    3. 应用交叉口区域掩码
    4. 返回可微分损失值
    
    输入要求：
    (B=批大小,T=时间步,N=车道线点数）
    predictions["trajectory"]: [batch_size, time_steps, 2] (x,y坐标)
    targets["centerline"]: [batch_size, num_points, 2] 车道线坐标
    targets["intersection_mask"]: [batch_size, time_steps] bool类型交叉口掩码
    """
    pred_traj = predictions["trajectory"]  # [B, T, 2]
    centerline = targets["centerline"]     # [B, N, 2]
    intersection_mask = targets["intersection_mask"].float()  # [B, T]
    
    B, T, _ = pred_traj.shape
    _, N, _ = centerline.shape

    # ===== 1. 计算点到车道线的最短距离 =====
    # 分解车道线段
    line_start = centerline[:, :-1, :]  # [B, N-1, 2]
    line_end = centerline[:, 1:, :]     # [B, N-1, 2]
    
    # 扩展维度用于广播计算
    points = pred_traj.unsqueeze(2)     # [B, T, 1, 2]
    starts = line_start.unsqueeze(1)    # [B, 1, N-1, 2]
    ends = line_end.unsqueeze(1)        # [B, 1, N-1, 2]
    
    # 计算投影参数
    vec = ends - starts  # 线段方向向量 [B, 1, N-1, 2]
    t_numerator = torch.sum((points - starts) * vec, dim=-1)  # [B, T, N-1]
    t_denominator = torch.sum(vec ** 2, dim=-1) + 1e-6  # [B, 1, N-1]
    t = t_numerator / t_denominator  # [B, T, N-1]
    
    # 限制投影在线段范围内
    t = torch.clamp(t, 0.0, 1.0).unsqueeze(-1)  # [B, T, N-1, 1]
    
    # 计算最近点坐标
    projections = starts + t * vec  # [B, T, N-1, 2]
    
    # 计算最小距离
    dists = torch.norm(points - projections, dim=-1)  # [B, T, N-1]
    min_dists, _ = torch.min(dists, dim=2)  # [B, T]
    
    # ===== 2. 连续超限权重计算 =====
    exceed_mask = (min_dists > config.lane_keeping_threshold).float()  # [B, T]
    exceed_mask *= (1 - intersection_mask)  # 应用交叉口掩码
    
    # 时序权重累积（可微分实现）
    weights = torch.zeros_like(exceed_mask)
    for t_step in range(1, T):
        weights[:, t_step] = config.lk_decay_factor * weights[:, t_step-1] + exceed_mask[:, t_step]
    
    # ===== 3. 最终损失计算 =====
    penalty = (min_dists - config.lane_keeping_threshold).clamp(min=0) * weights
    return penalty.sum(dim=1).mean()