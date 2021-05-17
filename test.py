import os
import time

import torch
import torch.nn as nn
import torch.quantization

from model.LSTM import LSTMModel
from util.data_manager import Corpus

model_data_filepath = 'data/'
corpus = Corpus(model_data_filepath + 'wikitext-2')
ntokens = len(corpus.dictionary)

### 构造模型 模型初始化
model = LSTMModel(
    ntoken = ntokens,
    ninp = 512,
    nhid = 256,
    nlayers = 5,
)
### 找不到预训练模型，暂时使用默认初始化模型参数
# model.load_state_dict(
#     torch.load(
#         model_data_filepath + 'word_language_model_quantize.pth',
#         map_location=torch.device('cpu')
#         )
#     )

model.eval()
print("====== show details of model ======")
print(model)


bptt = 25
criterion = nn.CrossEntropyLoss()
eval_batch_size = 1

# create test data set
### 构造测试集
def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    return data.view(bsz, -1).t().contiguous()

# 得到测试集
test_data = batchify(corpus.test, eval_batch_size)

# Evaluation functions
def get_batch(source, i):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].reshape(-1)
    return data, target

def repackage_hidden(h):
  """Wraps hidden states in new Tensors, to detach them from their history."""

  if isinstance(h, torch.Tensor):
      return h.detach()
  else:
      return tuple(repackage_hidden(v) for v in h)

### 测试用的主函数
def evaluate(model_, data_source):
    # Turn on evaluation mode which disables dropout.
    model_.eval()
    total_loss = 0.
    hidden = model_.init_hidden(eval_batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, bptt):
            data, targets = get_batch(data_source, i)
            output, hidden = model_(data, hidden)
            hidden = repackage_hidden(hidden)
            output_flat = output.view(-1, ntokens)
            ### 计算损失
            total_loss += len(data) * criterion(output_flat, targets).item()
    return total_loss / (len(data_source) - 1)

# 得到quantized_model
quantized_model = torch.quantization.quantize_dynamic(
    model, {nn.LSTM, nn.Linear}, dtype=torch.qint8
)
print("====== show details of quantized_model ======")
print(quantized_model)

def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')

print("====== print size of model ======")
print_size_of_model(model)
print("====== print size of quantized_model ======")
print_size_of_model(quantized_model)


torch.set_num_threads(1)

### 进行测试
# 测试函数 主体函数为evaluate(model, test_data)
def time_model_evaluation(model, test_data):
    s = time.time()
    loss = evaluate(model, test_data)# 调用测试evaluate
    elapsed = time.time() - s
    print('''loss: {0:.3f}\nelapsed time (seconds): {1:.1f}'''.format(loss, elapsed))

print("===>开始测试")
print("====== 原模型 ======")
time_model_evaluation(model, test_data)
print("====== quantized model ======")
time_model_evaluation(quantized_model, test_data)


