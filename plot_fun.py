import pickle
import matplotlib.pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser(description="CIFAR-10 demo")
parser.add_argument('--filename', default='alexnet_1.pkl', type=str, metavar='N',
        help='filename')

def plot_fun():
    filename = 'log/' + args.filename
    fp = open(filename, 'rb')

    record = pickle.load(fp)

    train_loss = record['train loss']
    train_acc = record['train acc']
    test_loss = record['test loss']
    test_acc = record['test acc']
    print(max(record['test acc']))

    epochs = len(train_loss)

    epoch = np.arange(1., epochs + 1, 1)
    fig = plt.figure()
    plt.xlabel('#Epoch')

    ax1 = fig.add_subplot(111)
    l1 = ax1.plot(epoch, train_loss, 'r', label='Training Loss')
    l2 = ax1.plot(epoch, test_loss, 'b', label='Testing Loss')
    ax1.set_ylabel('Loss')
    ax1.set_title('CIFAR-10')
    # ax1.legend(loc=0, bbox_to_anchor=(1, 0.7))

    ax2 = ax1.twinx()
    l3 = ax2.plot(epoch, train_acc, 'r--', label='Training Accuracy')
    l4 = ax2.plot(epoch, test_acc, 'b--', label='Testing Accuracy')
    ax2.set_ylabel('Accuracy')
    # ax2.legend(loc=0, bbox_to_anchor=(1, 0.3))
    ls = l1+l2+l3+l4
    les = [l.get_label() for l in ls]
    ax1.legend(ls, les, loc=0, bbox_to_anchor=(0.6, 0.2))

    plt.savefig('result/'+args.filename+'.pdf')
    plt.show()

if __name__=="__main__":
    global args
    args = parser.parse_args()
    plot_fun()






