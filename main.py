import torch
import sys
import os
import warnings
warnings.filterwarnings("ignore")
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import matplotlib.pyplot as plt
from common_models import GRUWithLinear, MLP  # 加载基本模型GRU和MLP
from get_data import get_dataloader  # 获取数据
from Supervised_Learning import train, test  # 模型训练和测试
from common_fusions import EMIFusion  # 模态融合

def init_argparse():
    import argparse
    parser = argparse.ArgumentParser(description="多模态情感分析")
    parser.add_argument("--path", type=str, default='C:/Users/User/Desktop/Code1/Code1/data/SIMS/sims_raw.pkl', help="数据路径")
    parser.add_argument("--epochs", type=int, default=5, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=2048, help="训练时的批大小")
    parser.add_argument("--num_workers", type=int, default=0, help="子进程数量")
    parser.add_argument("--data_type", type=str, default='mosi', help="要加载的数据类型")
    parser.add_argument("--max_seq_len", type=int, default=50, help="最大序列长度")
    parser.add_argument("--optimizer", type=str, default='AdamW', help="优化器")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="权重衰减")
    parser.add_argument("--save", type=str, default='best.pt', help="权重保存名字")
    parser.add_argument("--seed", type=int, default=1234, help="随机种子")
    return parser

def main(args):
    if args.optimizer == 'AdamW':
        optimtype = torch.optim.AdamW
    elif args.optimizer == 'Adam':
        optimtype = torch.optim.Adam
    elif args.optimizer == 'SGD':
        optimtype = torch.optim.SGD
    else:
        raise ValueError("Invalid optimizer name")
    sys.path.append(os.getcwd())
    sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
    traindata, validdata, testdata = get_dataloader(args.path, batch_size=args.batch_size, num_workers=args.num_workers,
                                                    robust_test=False, max_pad=False, data_type=args.data_type,
                                                    max_seq_len=args.max_seq_len)
    sys.path.append(os.getcwd())
    sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
    encoders = [GRUWithLinear(35, 64, 32, dropout=True, has_padding=True).cuda(),
                GRUWithLinear(74, 128, 32, dropout=True, has_padding=True).cuda(),
                GRUWithLinear(300, 512, 128, dropout=True, has_padding=True).cuda()]
    head = MLP(128, 512, 1).cuda()

    fusion = EMIFusion([32, 32, 128], 128, 32).cuda()
    trainlosses, vallosses = train(encoders, fusion, head, traindata, validdata, args.epochs, optimtype=optimtype,
          is_packed=True, lr=args.lr, save=args.save, weight_decay=args.weight_decay, objective=torch.nn.L1Loss())
    model = torch.load('best.pt').cuda()
    # 绘制训练损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(trainlosses) + 1), trainlosses, color='blue')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()

    # 绘制验证损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(vallosses) + 1), vallosses, color='red')
    plt.title('Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()

    test(model, testdata, 'affect', is_packed=True, criterion=torch.nn.L1Loss(), no_robust=True)

if __name__ == "__main__":
    parser = init_argparse()
    args = parser.parse_args()
    main(args)
