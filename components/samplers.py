import torch
from sklearn.cluster import DBSCAN


def farthest_point_sample(points, npoint): # From PointNetV2
    """
    Identify a well-spread subset P2 ⊂ P1 with the requisite cardinality
    Input:
        points: pointcloud data
        npoint: number of samples
    """
    device = points.device
    batch_size, N, _ = points.shape
    centroids = torch.zeros(batch_size, npoint, dtype=torch.long).to(device)
    distance = torch.ones(batch_size, N).to(device) * 1e10
    farthest = torch.randint(0, N, (batch_size,), dtype=torch.long).to(device)
    batch_indices = torch.arange(batch_size, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = points[batch_indices, farthest, :].view(batch_size, 1, 3)
        dist = torch.sum((points - centroid) ** 2, -1)
        distance = torch.min(distance, dist)
        farthest = torch.max(distance, -1)[1]
    return centroids.to(device)


def density_based_sampling(points, num_samples):
    """
    Perform density-based sampling using DBSCAN.
    
    Args:
        points (torch.Tensor): Point cloud data (batch_size, N, 3).
        num_samples (int): Number of samples to select.

    Returns:
        torch.Tensor: Indices of sampled points (batch_size, num_samples).

    Note:
        This function applies DBSCAN clustering to the input point cloud to identify clusters
        and then samples points from each cluster to ensure a diverse selection of points.
    """
    sampled_indices = []
    for point_cloud in points:
        dbscan = DBSCAN(eps=0.1, min_samples=10)  
        labels = dbscan.fit_predict(point_cloud.cpu().numpy())
        unique_labels, counts = torch.unique(torch.tensor(labels), return_counts=True)
        if len(unique_labels) == 0:
            continue  # Skip if no clusters found
        sampled_labels = unique_labels[torch.multinomial(counts.float(), min(num_samples, len(unique_labels)), replacement=True)]
        for label in sampled_labels:
            # Filter out-of-bounds indices
            valid_indices = torch.nonzero(torch.tensor(labels) == label).flatten()
            sampled_indices.extend(valid_indices.tolist())

    num_samples = min(num_samples, len(sampled_indices))
    sampled_indices = sampled_indices[:num_samples]
    return torch.tensor(sampled_indices).unsqueeze(0).expand(points.size(0), -1).to(points.device)

