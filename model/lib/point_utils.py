import torch
import torch.nn as nn
from torch.autograd import Function
import pointnet2_cuda

class KNN(nn.Module):
    def __init__(self, neighbors, transpose_mode=True):
        super(KNN, self).__init__()
        self.neighbors = neighbors

    @torch.no_grad()
    def forward(self, support, query):
        """
        Args:
            support ([tensor]): [B, N, C]
            query ([tensor]): [B, M, C]
        Returns:
            [int]: neighbor idx. [B, M, K]
        """
        dist = torch.cdist(support, query)
        k_dist = dist.topk(k=self.neighbors, dim=1, largest=False)
        return k_dist.values, k_dist.indices.transpose(1, 2).contiguous().int()


class GroupingOperation(Function):

    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, features: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """
        :param ctx:
        :param features: (B, C, N) tensor of features to group
        :param idx: (B, npoint, nsample) tensor containing the indicies of features to group with
        :return:
            output: (B, C, npoint, nsample) tensor
        """
        assert features.is_contiguous()
        assert idx.is_contiguous()

        B, nfeatures, nsample = idx.size()
        _, C, N = features.size()
        output = torch.cuda.FloatTensor(B, C, nfeatures, nsample, device=features.device)

        pointnet2_cuda.group_points_wrapper(B, C, N, nfeatures, nsample, features, idx, output)

        ctx.for_backwards = (idx, N)
        return output

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        """
        :param ctx:
        :param grad_out: (B, C, npoint, nsample) tensor of the gradients of the output from forward
        :return:
            grad_features: (B, C, N) gradient of the features
        """
        idx, N = ctx.for_backwards

        B, C, npoint, nsample = grad_out.size()
        grad_features = torch.zeros([B, C, N], dtype=torch.float, device=grad_out.device, requires_grad=True)
        grad_out_data = grad_out.data.contiguous()
        pointnet2_cuda.group_points_grad_wrapper(B, C, N, npoint, nsample, grad_out_data, idx, grad_features.data)
        return grad_features, None

grouping_operation = GroupingOperation.apply


class KNNGroup(nn.Module):
    def __init__(self, nsample: int,
                 relative_xyz=True,
                 normalize_dp=False,
                 return_only_idx=False,
                 **kwargs
                 ):
        """[summary]

        Args:
            nsample (int): maximum number of features to gather in the ball
            use_xyz (bool, optional): concate xyz. Defaults to True.
            ret_grouped_xyz (bool, optional): [description]. Defaults to False.
            normalize_dp (bool, optional): [description]. Defaults to False.
        """
        super().__init__()
        self.nsample = nsample
        self.knn = KNN(nsample, transpose_mode=True)
        self.relative_xyz = relative_xyz
        self.normalize_dp = normalize_dp
        self.return_only_idx = return_only_idx

    def forward(self, query_xyz: torch.Tensor, support_xyz: torch.Tensor, features: torch.Tensor = None):
        """
        :param query_xyz: (B, N, 3) xyz coordinates of the features
        :param support_xyz: (B, npoint, 3) centroids
        :param features: (B, C, N) descriptors of the features
        :return:
            new_features: (B, 3 + C, npoint, nsample)
        """
        _, idx = self.knn(support_xyz, query_xyz)
        if self.return_only_idx:
            return idx
        idx = idx.int()
        xyz_trans = support_xyz.transpose(1, 2).contiguous()
        grouped_xyz = grouping_operation(xyz_trans, idx)  # (B, 3, npoint, nsample)
        if self.relative_xyz:
            grouped_xyz -= query_xyz.transpose(1, 2).unsqueeze(-1)  # relative position
        if self.normalize_dp:
            grouped_xyz /= torch.amax(torch.sqrt(torch.sum(grouped_xyz**2, dim=1)), dim=(1, 2)).view(-1, 1, 1, 1)
        if features is not None:
            grouped_features = grouping_operation(features, idx)
            return grouped_xyz, grouped_features
        else:
            return grouped_xyz, None


class FurthestPointSampling(Function):
    @staticmethod
    def forward(ctx, xyz: torch.Tensor, npoint: int) -> torch.Tensor:
        """
        Uses iterative furthest point sampling to select a set of npoint features that have the largest
        minimum distance
        :param ctx:
        :param xyz: (B, N, 3) where N > npoint
        :param npoint: int, number of features in the sampled set
        :return:
             output: (B, npoint) tensor containing the set (idx)
        """
        assert xyz.is_contiguous()

        B, N, _ = xyz.size()
        # output = torch.cuda.IntTensor(B, npoint, device=xyz.device)
        # temp = torch.cuda.FloatTensor(B, N, device=xyz.device).fill_(1e10)
        output = torch.cuda.IntTensor(B, npoint)
        temp = torch.cuda.FloatTensor(B, N).fill_(1e10)

        pointnet2_cuda.furthest_point_sampling_wrapper(
            B, N, npoint, xyz, temp, output)
        return output

    @staticmethod
    def backward(xyz, a=None):
        return None, None

furthest_point_sample = FurthestPointSampling.apply


class PointPatchEmbed(nn.Module):

    def __init__(self,
                 sample_ratio=0.0625,
                 sample_number=1024,
                 group_size=32,
                 in_channels=6,
                 channels=1024,
                 kernel_size=1,
                 stride=1,
                 normalize_dp=False,
                 relative_xyz=True,
                 ):
        super().__init__()
        self.sample_ratio = sample_ratio
        self.sample_number = sample_number
        self.group_size = group_size

        self.sample_fn = furthest_point_sample
        self.grouper = KNNGroup(self.group_size, relative_xyz=relative_xyz, normalize_dp=normalize_dp)
        
        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=kernel_size, stride=stride)


    def forward(self, x):
        # coordinates
        p = x[:, :, 3:].contiguous()

        B, N, _ = p.shape[:3]
        # idx = self.sample_fn(p, int(N * self.sample_ratio)).long()
        idx = self.sample_fn(p, self.sample_number).long()
        center_p = torch.gather(p, 1, idx.unsqueeze(-1).expand(-1, -1, 3))
        # query neighbors.
        _, fj = self.grouper(center_p, p, x.permute(0, 2, 1).contiguous()) # [B, N, 6] -> [B, 6, N] -> [B, 6, 1024, 32]

        # [B, 6, 1024] -> [B, channels, 1024, 1]
        fj = self.conv1(fj).max(dim=-1, keepdim=True)[0]

        return fj


if __name__ == '__main__':
    model = PointPatchEmbed(channels=256).cuda()
    input = torch.rand(4, 16384, 6).cuda()
    ou = model(input)
    import pdb;pdb.set_trace()