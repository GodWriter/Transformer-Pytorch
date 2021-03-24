import torch.nn as nn
import torch.optim as optim

from dataloader import *
from model import Transformer


def train():
    inputs, src_vocab_size, tgt_vocab_size, idx2word = create_data()

    enc_inputs, dec_inputs, dec_outputs = make_data(*inputs)
    data_loader = Data.DataLoader(dataset=MyDataSet(enc_inputs, dec_inputs, dec_outputs),
                                  batch_size=2,
                                  shuffle=True)

    model = Transformer(src_vocab_size, tgt_vocab_size).cuda()
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # PAD本身无意义，单词索引为0，设置ignore_index=0，可避免计算PAD的损失
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.09)

    for epoch in range(30):
        for enc_inputs, dec_inputs, dec_outputs in data_loader:
            """
            enc_inputs: [batch_size, src_len]
            dec_inputs: [batch_size, tgt_len]
            dec_outputs: [batch_size, tgt_len]
            """

            enc_inputs, dec_inputs, dec_outputs = enc_inputs.cuda(), dec_inputs.cuda(), dec_outputs.cuda()

            outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
            loss = criterion(outputs, dec_outputs.view(-1))

            print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


if __name__ == '__main__':
    train()
