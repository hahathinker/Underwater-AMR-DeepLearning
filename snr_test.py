import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from model import UnderwaterCNN
from OFDM_dataset import Dataset
from utils import Averager, count_acc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====================== 超参数 ======================
BATCH_SIZE = 256
MODEL_PATH = "./model/underwater_cnn.pth"
SNRS = [-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10]
# ====================================================

# 加载数据集
dataset = Dataset()
total_len = len(dataset)
train_len = int(total_len * 0.6)
val_len = int(total_len * 0.3)
test_len = total_len - train_len - val_len
_, _, test_set = torch.utils.data.random_split(dataset, [train_len, val_len, test_len])

# 加载模型
model = UnderwaterCNN(num_class=5).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# 存储结果（用于画图）
snr_acc_results = []

print("\n========== 开始按信噪比测试 ==========\n")
with torch.no_grad():
    for snr in SNRS:
        data_list = []
        label_list = []

        # 筛选当前 SNR 的所有测试样本
        for idx in range(len(test_set)):
            x, y, s = test_set[idx]
            if s == snr:
                data_list.append(x)
                label_list.append(y)

        if len(data_list) == 0:
            print(f"SNR {snr} 无数据")
            continue

        # 构建数据集
        data_tensor = torch.stack(data_list)
        label_tensor = torch.tensor(label_list, dtype=torch.long)
        test_ds = TensorDataset(data_tensor, label_tensor)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

        acc_avg = Averager()
        for data, label in test_loader:
            data = data.to(device).unsqueeze(1)
            label = label.to(device)
            logits = model(data)
            acc = count_acc(logits, label)
            acc_avg.add(acc)

        acc = acc_avg.item()
        snr_acc_results.append((snr, acc))
        print(f"snr = {snr:2d},  acc = {acc:.4f}")

# ====================== 最终输出（可直接写论文） ======================
print("\n========== SNR - Accuracy 曲线数据 ==========")
for snr, acc in snr_acc_results:
    print(f"{snr}    {acc:.4f}")