import argparse
import logging
import torch
import torch.optim as optim

from src.network.policy_value_resnet import PolicyValueNetwork
from src.dataloader import HcpeDataLoader


parser = argparse.ArgumentParser(description="Train policy value network")
parser.add_argument("train_data", type=str, nargs="+", help="training data file")
parser.add_argument("test_data", type=str, help="test data file")
parser.add_argument("--gpu", "-g", type=int, help="GPU ID")
parser.add_argument("--epoch", "-e", type=int, default=1, help="Number of epoch times")
parser.add_argument("--batchsize", "-b", type=int, default=1024,
                    help="Number of positions in each mini-batch")
parser.add_argument("--testbatch_size", type=int, default=1024,
                    help="Number of positions in each test mini-batch")
parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
parser.add_argument("--checkpoint", default="checkpoints/checkpoint-{epoch:03}.pth",
                    help="checkpoint file name")
parser.add_argument("--resume", "-r", default="", help="resume from snapshot")
parser.add_argument("--eval_interval", type=int, default=100, help="evaluation interval")
parser.add_argument("--log", type=None, help="log file path")
args = parser.parse_args()


logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    filename=args.log if args.log else "train.log",
    level=logging.DEBUG
)
logging.info(f"batchsize={args.batchsize}")
logging.info(f"learning_rate={args.lr}")

# device
if args.gpu >= 0:
    device = torch.device(f"cuda:{args.gpu}")
else:
    device = torch.device("cpu")

# model
model = PolicyValueNetwork()
model.to(device)

# optimizer
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0001)

# loss function
cross_entropy_loss = torch.nn.CrossEntropyLoss()
bce_with_logits_loss = torch.nn.BCEWithLogitsLoss()  # BCE: binary cross-entropy

# load checkpoint
if args.resume:
    logging.info(f"Loading the checkpoint from {args.resume}")
    checkpoint = torch.load(args.resume, map_location=device)
    epoch = checkpoint["epoch"]
    t = checkpoint["t"]  # total steps
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # change learning rate to argument value
    optimizer.param_groups[0]["lr"] = args.lr
else:
    epoch = 0
    t = 0  # total steps


# read train & test data
logging.info("Reading training data")
train_dataloader = HcpeDataLoader(args.train_data, args.batchsize, device, shuffle=True)
logging.info("Reading test data")
test_dataloader = HcpeDataLoader(args.test_data, args.testbatchsize, device)

logging.info(f"train position num = {len(train_dataloader)}")
logging.info(f"test position num = {len(test_dataloader)}")


def accuracy(y, t):
    """calculate policy accuracy"""
    return (torch.max(y, 1)[1] == t).sum().item() / len(t)

def binary_accuracy(y, t):
    """calculate value accuracy"""
    pred = (y >= 0)
    truth = (t >= 0.5)
    return pred.eq(truth).sum().item() / len(t)

def save_checkpotint():
    path = args.checkpoint.format(**{"epoch": epoch, "step": t})
    logging.info(f"Saving the checkpoint to {path}")
    checkpoint = {
        "epoch": epoch,
        "t": t,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict()
    }
    torch.save(checkpoint, path)


# train loop
for e in range(args.epoch):
    epoch += 1
    steps_interval = 0
    sum_loss_policy_interval = 0
    sum_loss_value_interval = 0
    steps_epoch = 0
    sum_loss_policy_epoch = 0
    sum_loss_value_epoch = 0
    for x, move_label, result in train_dataloader:
        model.train()

        # forward
        y1, y2 = model(x)
        # calc loss
        loss_policy = cross_entropy_loss(y1, move_label)
        loss_value = bce_with_logits_loss(y2, result)
        loss = loss_policy + loss_value
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # add 1 to total step
        t += 1

        # add 1 to step counter, add losses to total loss for evaluation
        # for interval
        steps_interval += 1
        sum_loss_policy_interval += loss_policy.item()
        sum_loss_value_interval += loss_value.item()
        # for epoch
        steps_epoch += 1
        sum_loss_policy_epoch += loss_policy.item()
        sum_loss_value_epoch += loss_value.item()

        # display training loss and test loss and accuracy
        # for each evaluation interval
        if t % args.eval_interval == 0:
            model.eval()

            x, move_label, result = test_dataloader.sample()
            with torch.no_grad():
                # predict
                y1, y2 = model(x)
                # calc loss
                test_loss_policy = cross_entropy_loss(y1, move_label).item()
                test_loss_value = bce_with_logits_loss(y2, result).item()
                # calc accuracies
                test_accuracy_policy = accuracy(y1, move_label)
                test_accuracy_value = binary_accuracy(y2, result)

                logging.info(
                    f"epoch = {epoch}, step = {t}, train loss = "
                    f"{sum_loss_policy_interval / steps_interval:.5f}, "
                    f"{sum_loss_value_interval / steps_interval:.5f}, "
                    f"{(sum_loss_policy_interval + sum_loss_value_interval) / steps_interval:.5f},"
                    f" test loss = {test_loss_policy:.5f}, {test_loss_value:.5f}, "
                    f"{test_loss_policy + test_loss_value:.5f}, "
                    f"test accuracy = {test_accuracy_policy:.5f}, {test_loss_value:.5f}"
                )

            steps_interval = 0
            sum_loss_policy_interval = 0
            sum_loss_value_interval = 0

    # evaluate using all of the data at the end of the epoch
    test_steps = 0
    sum_test_loss_policy = 0
    sum_test_loss_value = 0
    sum_test_accuracy_policy = 0
    sum_test_accuracy_value = 0

    model.eval()
    with torch.no_grad():
        for x, move_label, result in test_dataloader:
            y1, y2 = model(x)

            test_steps += 1
            sum_test_loss_policy += cross_entropy_loss(y1, move_label).item()
            sum_test_loss_value += bce_with_logits_loss(y2, result).item()
            sum_test_accuracy_policy += accuracy(y1, move_label)
            sum_test_accuracy_value += binary_accuracy(y2, result)

    logging.info(
        f"epoch = {epoch}, step = {t}, "
        f"train loss = "
        f"{sum_loss_policy_epoch / steps_epoch:.5f}, "
        f"{sum_loss_value_epoch / steps_epoch:.5f}, "
        f"{(sum_loss_policy_epoch + sum_loss_value_epoch) / steps_epoch:.5f},"
        f" test loss = "
        f"{sum_test_loss_policy / test_steps:.5f}, {sum_test_loss_value / test_steps:.5f}, "
        f"{(sum_test_loss_policy + sum_test_loss_value) / test_steps:.5f}, "
        f"test accuracy = "
        f"{sum_test_accuracy_policy / test_steps:.5f}, {sum_test_loss_value / test_steps:.5f}"
    )

    if args.checkpoint:
        save_checkpotint()
