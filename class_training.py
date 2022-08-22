import os

import monai
import torch
from timm.utils import AverageMeter
# from models.MobilenetV2 import get_model
from get_class_data_loaders import get_class_data_loaders
# from get_training_components import get_wandb
from TrainConfig import config
from tqdm import trange
from torch.cuda.amp import autocast, GradScaler

from monai.networks.nets import  EfficientNet
import  timm
from get_training_components import get_wandb
def classification(config):
    if config.use_wandb:
        mywandb = get_wandb(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(device)

    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), config.learning_rate)

    val_interval = 2
    best_metric = -1
    for fold in range(5):
        if config.resume:
            resume_path = os.path.join("saved_models","fold_{}classification_model.pth".format(fold))
            print("resume from {}".format(resume_path))
            model.load_state_dict(torch.load(resume_path), strict=True)
        train_loader, val_loader = get_class_data_loaders(config,fold)
        print(" train loader length {} val length {}".format(len(train_loader),len(val_loader)))
        epochs  =200
        softmax_l = torch.nn.Softmax(dim=1)
        with trange(epochs) as t:
            for epoch in t:
                train_info ={}
                t.set_description('Fold %i Epoch %i' % (fold,epoch))
                loss_one_epoch,train_acc = train_one_epoch(device,  loss_function, model, optimizer, softmax_l, train_loader,epoch)
                if epoch % val_interval == 0:
                    valid_metric_epoch = valid_one_epoch(device,  model, softmax_l, val_loader)
                    if config.developing:
                        if valid_metric_epoch > best_metric:
                            if not os.path.exists("saved_models"):
                                os.mkdir("saved_models")
                            best_metric = valid_metric_epoch
                            torch.save(model.state_dict(), os.path.join("saved_models", "fold_{}classification_model.pth".format(fold)))
                            print("saved new best metric model")
                train_info["Fold{} classification train loss ".format(fold)] = loss_one_epoch
                train_info["Fold{} classification train acc ".format(fold)] = train_acc
                train_info["Fold{} classification valid acc ".format(fold)] = valid_metric_epoch
                if config.use_wandb:
                    mywandb.upload_wandb_info(train_info)
                # if config.use_wandb:
                #     mywandb.upload_wandb_info(train_info)
                # print("epoch {} matrics {}".format(epoch,train_info))
                t.set_postfix(train_info)


def get_model(device):
    # model = monai.networks.nets.Densenet(spatial_dims=2, in_channels=3, out_channels=3).to(device)
    # model_name = "mobilenetv3_small_075"
    model_name = "tf_mobilenetv3_small_100"
    config.run_name = "model {}".format(model_name)
    model = timm.create_model(model_name,num_classes=3,in_chans=3,pretrained=True,
                              drop_rate=0.3, drop_path_rate=0.3).to(device)
    return model


def valid_one_epoch( device,  model, sigmoid_l, val_loader):
    model.eval()
    with torch.no_grad():
        num_correct = 0.0
        metric_count = 0
        for val_data in val_loader:
            val_images, val_labels = val_data["image"].to(device), val_data["label"].long().to(
                device)
            val_outputs = sigmoid_l(model(val_images))
            # print(val_outputs,val_outputs.argmax(dim=1),val_labels)
            value = torch.eq(val_outputs.argmax(dim=1), val_labels.squeeze(1).squeeze(1))
            metric_count += len(value)
            num_correct += value.sum().item()
        metric = num_correct / metric_count


    return  metric


def train_one_epoch(device,  loss_function, model, optimizer, softmax_l, train_loader,epoch):
    model.train()
    losses = AverageMeter()
    num_correct = 0.0
    metric_count = 0
    scaler = GradScaler()
    for idx , batch_data in enumerate(train_loader):
        if epoch == 0 :
            print(idx)
        inputs, labels = batch_data["image"].to(device), batch_data["label"].long().to(device)
        optimizer.zero_grad()
        with autocast():
            outputs = softmax_l(model(inputs))
            # outputs = sigmoid_l(model(inputs))

            loss = loss_function(outputs, labels.squeeze(1).squeeze(1))
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # optimizer.step()
        with torch.no_grad():
            # print(val_outputs,val_outputs.argmax(dim=1),val_labels)
            value = torch.eq(outputs.argmax(dim=1), labels.squeeze(1).squeeze(1))
            metric_count += len(value)
            num_correct += value.sum().item()
        losses.update(loss.item(),inputs.shape[0])
    train_acc = num_correct / metric_count

    return losses.avg,train_acc

if __name__ == "__main__":
    config.run_name = "chache class training kfold"
    classification(config)
