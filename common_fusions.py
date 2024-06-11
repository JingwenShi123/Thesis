import torch
from torch import nn
from torch.autograd import Variable

class EMIFusion(nn.Module):

    def __init__(self, input_dims, output_dim, rank, flatten=True):
        """
        初始化EMIFusion对象。

        :param input_dims: 输入模态的维度的列表或元组
        :param output_dim: 输出维度
        :param rank: LRTF的一个超参数。详细信息请参见上面的链接
        :param flatten: 布尔值，指示是否应该对输出进行扁平化。默认值：True

        """
        super(EMIFusion, self).__init__()

        # 维度按照音频、视频和文本的顺序指定
        self.input_dims = input_dims
        self.output_dim = output_dim
        self.rank = rank
        self.flatten = flatten

        self.factors = []
        for input_dim in input_dims:
            factor = nn.Parameter(torch.Tensor(
                self.rank, input_dim + 1, self.output_dim)).to(
                torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
            nn.init.xavier_normal_(factor)
            self.factors.append(factor)

        self.fusion_weights = nn.Parameter(torch.Tensor(1, self.rank)).to(
            torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        self.fusion_bias = nn.Parameter(
            torch.Tensor(1, self.output_dim)).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        # 初始化融合权重
        nn.init.xavier_normal_(self.fusion_weights)
        self.fusion_bias.data.fill_(0)

    def forward(self, modalities):

        batch_size = modalities[0].shape[0]
        fused_tensor = 1
        for (modality, factor) in zip(modalities, self.factors):
            ones = Variable(torch.ones(batch_size, 1).type(
                modality.dtype), requires_grad=False).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
            if self.flatten:
                modality_withones = torch.cat(
                    (ones, torch.flatten(modality, start_dim=1)), dim=1)
            else:
                modality_withones = torch.cat((ones, modality), dim=1)
            modality_factor = torch.matmul(modality_withones, factor)
            fused_tensor = fused_tensor * modality_factor

        output = torch.matmul(self.fusion_weights, fused_tensor.permute(
            1, 0, 2)).squeeze() + self.fusion_bias
        output = output.view(-1, self.output_dim)
        return output
