from torch import nn, optim
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import random
from metric.mask_metric import masked_mae, masked_mape, masked_rmse
from TIP.missPredNew import MissPredNew   # VMPredictor
import time
import datetime
from utils.adj_matrix import *
import copy
from data_solve import load_adj

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

seed = 42
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# GPU
gpu_id = 3
print("GPU:", gpu_id)

def Inverse_normalization(x, max, min):
    return x * (max - min) + min

### PEMS-BAY、METR-LA、PEMS04、PEMS08
data_name = 'PEMS08'
data_file = "./data/" + data_name + "/dataNew.npz"
raw_data = np.load(data_file, allow_pickle=True)
adj_path = './data/PEMS08/PEMS08.csv'
print(data_file)

### graph
batch_size = 48
epoch = 200
IF_mask = 0.9
print("MissingRate:" + str(int(IF_mask * 100)) + "%")
lr_rate = 0.001


now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
model_path = './result/' + data_name + '/' + data_name + str(int(100*IF_mask)) + str(now) + '.pt'

### Hyperparameter
input_len = 12
num_id = 170
out_len = 12
in_size = 4  
emb_size = 24  
print("emb_s:", emb_size)
graph_size = 20 
dropout = 0.15

adj_mx0 = get_adjacent_matrix(adj_path, num_id, graph_type="connect")    
adj_matrix = torch.from_numpy(adj_mx0)  # [N,N]
adj_mx = process_graph(adj_matrix)
max_norm = 5  # Gradient pruning
max_num = 100

###learning rate
num_lr = 5
gamme = 0.5
milestone = [1, 15, 40, 70, 90]

### Train_data
if IF_mask == 0.25:
    train_data = torch.cat([torch.tensor(raw_data["train_x_mask_25"]), torch.tensor(raw_data["train_y"])], dim=-1).to(torch.float32)
elif IF_mask == 0.5:
    train_data = torch.cat([torch.tensor(raw_data["train_x_mask_50"]), torch.tensor(raw_data["train_y"])], dim=-1).to(
        torch.float32)
elif IF_mask == 0.75:
    train_data = torch.cat([torch.tensor(raw_data["train_x_mask_75"]), torch.tensor(raw_data["train_y"])], dim=-1).to(
        torch.float32)
elif IF_mask == 0.9:
    train_data = torch.cat([torch.tensor(raw_data["train_x_mask_90"]), torch.tensor(raw_data["train_y"])], dim=-1).to(
        torch.float32)
else:
    train_data = torch.cat([torch.tensor(raw_data["train_x_raw"]), torch.tensor(raw_data["train_y"])], dim=-1).to(
        torch.float32)

train_data = DataLoader(train_data,batch_size=batch_size,shuffle=True)


### Valid_data
if IF_mask == 0.25:
    valid_data = torch.cat([torch.tensor(raw_data["vail_x_mask_25"]), torch.tensor(raw_data["vail_y"])], dim=-1).to(torch.float32)
elif IF_mask == 0.5:
    valid_data = torch.cat([torch.tensor(raw_data["vail_x_mask_50"]), torch.tensor(raw_data["vail_y"])], dim=-1).to(torch.float32)
elif IF_mask == 0.75:
    valid_data = torch.cat([torch.tensor(raw_data["vail_x_mask_75"]), torch.tensor(raw_data["vail_y"])], dim=-1).to(torch.float32)
elif IF_mask == 0.9:
    valid_data = torch.cat([torch.tensor(raw_data["vail_x_mask_90"]), torch.tensor(raw_data["vail_y"])], dim=-1).to(torch.float32)
else:
    valid_data = torch.cat([torch.tensor(raw_data["vail_x_raw"]), torch.tensor(raw_data["vail_y"])], dim=-1).to(torch.float32)

valid_data = DataLoader(valid_data,batch_size=batch_size,shuffle=False)

### test_data
if IF_mask == 0.25:
    test_data = torch.cat([torch.tensor(raw_data["test_x_mask_25"]), torch.tensor(raw_data["test_y"])], dim=-1).to(torch.float32)
elif IF_mask == 0.5:
    test_data = torch.cat([torch.tensor(raw_data["test_x_mask_50"]), torch.tensor(raw_data["test_y"])], dim=-1).to(torch.float32)
elif IF_mask == 0.75:
    test_data = torch.cat([torch.tensor(raw_data["test_x_mask_75"]), torch.tensor(raw_data["test_y"])], dim=-1).to(torch.float32)
elif IF_mask == 0.9:
    test_data = torch.cat([torch.tensor(raw_data["test_x_mask_90"]), torch.tensor(raw_data["test_y"])], dim=-1).to(torch.float32)
else:
    test_data = torch.cat([torch.tensor(raw_data["test_x_raw"]), torch.tensor(raw_data["test_y"])], dim=-1).to(torch.float32)

test_data = DataLoader(test_data,batch_size=batch_size,shuffle=False)


max_min = raw_data['max_min']
max_data, min_data = max_min[0], max_min[1]

device = torch.device(f"cuda:{gpu_id}")

if not torch.cuda.is_available() or gpu_id >= torch.cuda.device_count():
    raise RuntimeError(f"GPU {gpu_id} is not available.")

device2 = torch.device(f"cuda:{gpu_id}")
adj_mx = adj_mx.to(device2)  


my_net = MissPredNew(input_len, num_id, out_len, emb_size, adj_mx, graph_size)    

print(f"Model name: {my_net.__class__.__name__}")


trainable_params = sum(p.numel() for p in my_net.parameters() if p.requires_grad)
print("Param:", trainable_params)

my_net = my_net.to(device)
optimizer = optim.Adam(params=my_net.parameters(), lr=lr_rate)
num_vail = 0

min_vaild_loss = float("inf")
best_state_dict = None
best_epoch = 0

### train
for i in range(epoch):
    num = 0
    loss_out = 0.0
    my_net.train()
    start = time.time()
    for data in train_data:
        my_net.zero_grad()

        train_feature = data[:, :, :, 0:in_size].to(device)
        train_target = data[:, :, :, -1].to(device)
        train_pre = my_net(train_feature)
        loss_data = masked_mae(train_pre, train_target, 0.0)

        num += 1
        loss_data.backward()

        if max_norm > 0 and i < max_num:
            nn.utils.clip_grad_norm_(my_net.parameters(), max_norm=max_norm)
        else:
            pass
        num += 1

        optimizer.step()
        loss_out += loss_data
    loss_out = loss_out / num
    end = time.time()

    num_va = 0
    loss_vaild = 0.0
    my_net.eval()
    with torch.no_grad():
        for data in valid_data:
            valid_x = data[:, :, :, 0:in_size].to(device)
            valid_y = data[:, :, :, -1].to(device)
            valid_pre = my_net(valid_x)
            loss_data = masked_mae(valid_pre, valid_y, 0.0)

            num_va += 1
            loss_vaild += loss_data
        loss_vaild = loss_vaild / num_va

    print(
        'Loss of the {} epoch of the training set: {:02.4f}, Loss of the validation set Loss:{:02.4f}, training time: {:02.4f}:'.format(
            i + 1, loss_out, loss_vaild, end - start))
    if loss_vaild < min_vaild_loss:
        min_vaild_loss = loss_vaild
        best_state_dict = copy.deepcopy(my_net.state_dict())
        print("save model..........")
        torch.save(my_net.state_dict(), model_path)
        best_epoch = 0
    else:
        best_epoch += 1
        print("best_epoch", best_epoch)

    if best_epoch >= 15:
        break

my_net.load_state_dict(best_state_dict)
my_net.eval()
my_net = my_net.to(device2)
with torch.no_grad():
    all_pre = 0.0
    all_true = 0.0
    num = 0
    for data in test_data:
        test_feature = data[:, :, :, 0:in_size].to(device2)
        test_target = data[:, :, :, -1].to(device2)
        test_pre = my_net(test_feature)
        if num == 0:
            all_pre = test_pre
            all_true = test_target
        else:
            all_pre = torch.cat([all_pre, test_pre], dim=0)
            all_true = torch.cat([all_true, test_target], dim=0)
        num += 1

final_pred = Inverse_normalization(all_pre, max_data, min_data)
final_target = Inverse_normalization(all_true, max_data, min_data)

mae, mape, rmse = masked_mae(final_pred, final_target, 0.0), \
                  masked_mape(final_pred, final_target, 0.0) * 100, masked_rmse(final_pred, final_target, 0.0)
print('RMSE: {}, MAPE: {}, MAE: {}'.format(rmse, mape, mae))



