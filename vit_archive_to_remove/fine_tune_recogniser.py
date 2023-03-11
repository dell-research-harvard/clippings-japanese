import torch
import torch.nn as nn
from pytorch_metric_learning import losses, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from pytorch_metric_learning.utils.inference import InferenceModel, FaissKNN
import logging
import faiss
import os
from torchvision import transforms as T
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW, SGD, LBFGS, RMSprop
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau, CosineAnnealingLR,ExponentialLR,StepLR, CyclicLR

logging.getLogger().setLevel(logging.INFO)
from transformers import get_cosine_schedule_with_warmup
import wandb
import argparse

from models.encoders import *
# from models.classifiers import *
from datasets.recognizer_datasets import * # make sure Huggingface datasets is not installed...
from utils.datasets_utils import INV_NORMALIZE, create_random_doc_transform, LIGHT_AUG_BASE
import glob
import scripts.infer_for_synth_match as test_lang_images

import torch.nn.functional as F
from utils.gen_synthetic_segments import LIGHT_AUG

import infer_for_synth_match_japanese_all_tk 
import time

from utils.gpu_temp_utils import manage_temp 

##Misc functions
def get_last_digit(x):
    return (str(x)[-1])


class DifferentialEntropyRegularization(torch.nn.Module):

    def __init__(self, eps=1e-8):
        super(DifferentialEntropyRegularization, self).__init__()
        self.eps = eps
        self.pdist = torch.nn.PairwiseDistance(2)

    def forward(self, x):

        with torch.no_grad():
            dots = torch.mm(x, x.t())
            n = x.shape[0]
            dots.view(-1)[::(n + 1)].fill_(-1)  # trick to fill diagonal with -1
            _, I = torch.max(dots, 1)  # max inner prod -> min distance

        rho = self.pdist(x, x[I])

        # dist_matrix = torch.norm(x.unsqueeze(1) - x.unsqueeze(0), p=2, dim=-1)
        # rho = dist_matrix.topk(k=2, largest=False)[0][:, 1]

        loss = -torch.log(rho + self.eps).mean()

        return loss

def diff_size_collate(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    target = torch.LongTensor(target)
    return [data, target]


def get_class_image_name(image_name):
    return image_name.split("-font-")[1].split("-ori-")[0]

def infer_hardneg(query_paths, ref_dataset, model, index_path, transform, inf_save_path,inf_check_save_path, k=8,render=False):

    knn_func = FaissKNN(index_init_fn=faiss.IndexFlatL2, reset_before=False, reset_after=False) ###IndexFlatIP was used previously
    infm = InferenceModel(model, knn_func=knn_func)
    infm.load_knn_func(index_path)
    
    all_nns = []
    all_nns_char=[]
    for query_path in tqdm(query_paths):
        im = Image.open(query_path).convert("RGB")
        query = transform(im).unsqueeze(0)
        _, indices = infm.get_nearest_neighbors(query, k=k)
        if render:
            nn_words_list = [os.path.basename(ref_dataset.data[i][0]).split("-font-")[1].split("-ori-")[0].split("_") for i in indices[0]]
            ##Paste the chars together
            nn_words= ["_".join(word) for word in nn_words_list]
            all_nns.append("|".join(nn_words))
            nn_words_char=[[chr(int(char)) for char in word] for word in nn_words_list]
            nn_words_char=["".join(word) for word in nn_words_char]
            all_nns_char.append("|".join(nn_words_char))
        
        else:
            nn_words = [os.path.basename(ref_dataset.data[i][0]).split("-var-")[0] for i in indices[0]]
            nn_words = [word.split(".")[0] for word in nn_words]
            all_nns.append("|".join(nn_words))

            

    with open(inf_save_path, 'w') as f:
        f.write("\n".join(all_nns))
    



def save_ref_index(ref_dataset, model, save_path,render=False):

    os.makedirs("indices", exist_ok=True)
    knn_func = FaissKNN(index_init_fn=faiss.IndexFlatIP, reset_before=False, reset_after=False)
    infm = InferenceModel(model, knn_func=knn_func)
    infm.train_knn(ref_dataset)
    infm.save_knn_func(os.path.join(save_path, "ref.index"))

    # ref_data_file_names = [chr(int(os.path.basename(x[0]).split("_")[0], base=16)) if \
    #     os.path.basename(x[0]).startswith("0x") else os.path.basename(x[0])[0] for x in ref_dataset.data]
    if render:
        ref_data_file_names = [os.path.basename(x[0]).split("-font-")[1].split("-ori-")[0]  for x in ref_dataset.data]
    else:
        ref_data_file_names =[os.path.basename(x[0]).split(".")[0] for x in ref_dataset.data ]
    with open(os.path.join(save_path, "ref.txt"), "w") as f:
        f.write("\n".join(ref_data_file_names))


def save_model(model_folder, enc, epoch, datapara):

    if not os.path.exists(model_folder): os.makedirs(model_folder)

    if datapara:
        torch.save(enc.module.state_dict(), os.path.join(model_folder, f"enc_{epoch}.pth"))
    else:
        torch.save(enc.state_dict(), os.path.join(model_folder, f"enc_{epoch}.pth"))


def get_all_embeddings(dataset, model, batch_size=128):

    tester = testers.BaseTester(batch_size=batch_size)
    return tester.get_all_embeddings(dataset, model)

def get_all_embedding_diff_size(dataset, model, batch_size=128):
    
    print("Embedding...")
    image_embeddings = []

    for images, _ in tqdm(dataset):
        images = images.to(device)
        with torch.no_grad(): #Get last hidden state
            embedding=model.forward(images[None, ...]).cpu().numpy()
            ##Normalize the embeddings
            embedding = embedding / np.linalg.norm(embedding)
            image_embeddings.append(embedding)
    
    image_class_name=[get_class_image_name(os.path.basename(x[0])) for x in dataset.data]
    # image_embeddings = np.concatenate(image_embeddings)


    return image_embeddings, image_class_name
    

    


def tester_knn(test_set, ref_set, model, accuracy_calculator, split, log=True):

    model.eval()

  
    test_embeddings, test_labels = get_all_embeddings(test_set, model)
    test_labels = test_labels.squeeze(1)
    ref_embeddings, ref_labels = get_all_embeddings(ref_set, model)
    ref_labels = ref_labels.squeeze(1)
    print("Computing accuracy...")
    # accuracies = accuracy_calculator.get_accuracy(test_embeddings, 
    #     ref_embeddings,
    #     test_labels,
    #     ref_labels,
    #     embeddings_come_from_same_source=False)

    # prec_1 = accuracies["precision_at_1"]

    ###Custom accuracy using faiss
    ###Get all paths
    ###Test base path 
    test_base_path=[os.path.basename(x[0]).split("-var-")[0] for x in test_set.data]
    ###Remove .png
    test_base_path = [x.split(".")[0] for x in test_base_path]
    
    

    ##Ref base path
    ref_base_path=[os.path.basename(x[0]).split("-var-")[0] for x in ref_set.data]
    ###Remove .png
    ref_base_path = [x.split(".")[0] for x in ref_base_path]

    ##Convert tensors to numpy arrays
    test_embeddings = test_embeddings.cpu().numpy()
    ref_embeddings = ref_embeddings.cpu().numpy()
    test_labels = test_labels.cpu().numpy()
    ref_labels = ref_labels.cpu().numpy()

    ###Make index of ref embeddings
    index = faiss.IndexFlatIP(test_embeddings[0].shape[0])
    index.add(ref_embeddings)

    ###Get the labels of the nearest neighbor
    D, I = index.search(test_embeddings, 1)
    ##Get base path of the nearest neighbors
    ref_base_path = np.array(ref_base_path)
    ref_base_path = ref_base_path[I]

    ref_base_path = ref_base_path.squeeze(1)
    print(ref_base_path[0])
    print(test_base_path[0])
    ###Check if base path is the same as the test base path
    correct = np.sum(ref_base_path == test_base_path)
    prec_1 = correct / len(test_base_path)

    ###print some incorrect examples
    incorrect = np.where(ref_base_path != test_base_path)[0]
    print("Incorrect examples")
    for i in incorrect:
        print(f"Test: {test_base_path[i]} Ref: {ref_base_path[i]}")


    print(f"Accuracy on {split} set (Precision@1) = {prec_1}")

    if log:
        wandb.log({f"{split}/accuracy": prec_1})
        print(f"Accuracy on {split} set (Precision@1) = {prec_1}")



    return prec_1

def tester_knn_diff_sizes(test_set, ref_set, model, accuracy_calculator, split, log=True):

    model.eval()

    test_embeddings, test_labels = get_all_embedding_diff_size(test_set, model)
    ref_embeddings, ref_labels = get_all_embedding_diff_size(ref_set, model)
    print("Computing accuracy...")

    ##Convert tensors to numpy arrays
    test_embeddings = np.concatenate(test_embeddings)
    ref_embeddings = np.concatenate(ref_embeddings)


    ###Search for the nearest neighbor using faiss of each test image
    index = faiss.IndexFlatIP(test_embeddings[0].shape[0])
    index.add(ref_embeddings)

    ###Get the labels of the nearest neighbor
    D, I = index.search(test_embeddings, 1)
    ref_labels = np.array(ref_labels)

    ###Compute the accuracy
    correct = 0
    for i in range(len(test_labels)):
        if test_labels[i] == ref_labels[I[i][0]]:
            correct += 1
    prec_1 = correct / len(test_labels)

    if log:
        wandb.log({f"{split}/accuracy": prec_1})
    print(f"Accuracy on {split} set (Precision@1) = {prec_1}")

    index.reset()

    return prec_1


def tester_ffnn(model, val_dataset, val_loader, device, split):

    model.eval()

    corr_preds = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            labels = labels.to(device)
            inputs = inputs.to(device)
            outputs = model(inputs)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            predictions = logits.argmax(-1)
            corr_preds += torch.sum(predictions == labels).item()

    acc = corr_preds / len(val_dataset)
    wandb.log({f"{split}/accuracy": acc})
    print(f"{split} set accuracy = {acc}")

    return acc



def trainer_knn(model, loss_func, device, train_loader, optimizer, epoch, epochviz=None, diff_sizes=False, scheduler=None,regularization=None,lambda_reg=0.1,alpha=None,loss_func_supcon=None,loss_func_infonce=None):

    model.train()

    for batch_idx, (data, labels) in enumerate(tqdm(train_loader)):

        labels = labels.to(device)
        data = [datum.to(device) for datum in data] if diff_sizes else data.to(device)
        # print("Data shape",data.shape)


        optimizer.zero_grad()

        if diff_sizes:
            out_emb = []
            for datum in data:
                emb = model(datum.unsqueeze(0)).squeeze(0)
                out_emb.append(emb)
            embeddings = torch.stack(out_emb, dim=0)
        else:
            embeddings = model(data)
            ##Normalize the embeddings
            embeddings = embeddings / torch.norm(embeddings, dim=1, keepdim=True)

        if alpha is None:
            ##work in progress
            loss = loss_func(embeddings, labels)
        else :
            loss_supcon = loss_func_supcon(embeddings, labels)
            

            ##Iterate over the labels in the batch
            n=3
            batch_infonce_loss_list = []
            ###Convert the data tensor to list by unbinding on the first dimension
            data = data.unbind(0)
            unique_labels = torch.unique(labels)
            ###iterate over each class
            for i in (unique_labels):
                index_class = torch.where(labels==i)[0]
                class_loss_list = []
                for j in index_class:
                    with torch.no_grad():
                    # print("##Make n augmentations of the datum")
                        augmented_datum_copies = [LIGHT_AUG(data[j]) for _ in range(n)]
                    # print(len(augmented_datum_copies))
                    # print("Make the other datum list - all elements other than j")
                    other_datum = [data[k] for k in index_class if k != j]
                    # [print(datum.shape) for datum in other_datum]
                    labels_other_datum=[0 for _ in other_datum]
                    # print("###Combine the augmented data with the other data")
                    augmented_datum_copies.extend(other_datum)
                    # print("###Combine the labels")
                    label_same_data = [1 for _ in range(n)] + labels_other_datum
                    label_same_data = torch.tensor(label_same_data, device=device)
                    ##Send cpu data to gpu
                    augmented_datum_copies=[datum.to(device) for datum in augmented_datum_copies]
                    ###Add an extra dimension to the data and stack
                    augmented_datum_copies=torch.stack(augmented_datum_copies,dim=0)
                    ##Convert the labels to tensor
                    # with torch.no_grad(): - dont do this!
                    augmented_datum_copies_embeds=model(augmented_datum_copies.detach())
                    augmented_datum_copies_embeds = augmented_datum_copies_embeds / torch.norm(augmented_datum_copies_embeds, dim=1, keepdim=True)
                    del augmented_datum_copies
                    class_loss_infonce = loss_func_infonce(augmented_datum_copies_embeds, label_same_data)
                    class_loss_list.append(class_loss_infonce)
                    ##Garbage collection
                    del augmented_datum_copies_embeds
                    del class_loss_infonce
                    del label_same_data
                    torch.cuda.empty_cache()
                class_mean_loss_infonce = torch.stack(class_loss_list).sum() ##Thanks lukas
                del class_loss_list
                batch_infonce_loss_list.append(class_mean_loss_infonce)
                del class_mean_loss_infonce
                torch.cuda.empty_cache()
            loss_infonce = torch.stack(batch_infonce_loss_list).sum() ##Thanks lukas
            loss=(1-alpha)*loss_supcon+alpha*loss_infonce
                    
        if regularization is not None:
            embeddings=F.normalize(embeddings, dim=1)
            loss_koleo=regularization(embeddings) 
            loss=loss + loss_koleo * lambda_reg

        loss.backward()
        optimizer.step()
        ##For ReduceLROnPlateau scheduler, we need to pass the loss value
        if scheduler!=None:
            scheduler.step()
            # scheduler.step(loss.item())
            if batch_idx % 50 == 0:
                print("Current LR: {}".format(scheduler.get_lr()[0]))
            wandb.log({"train/lr": scheduler.get_lr()[0]})
        wandb.log({"train/loss": loss.item()})

        if batch_idx % 50 == 0:
            print("Epoch {} Iteration {}: Loss = {}".format(
                str(epoch).zfill(3), str(batch_idx).zfill(4), loss))
            if not epochviz is None:
                for i in range(10):
                    image = T.ToPILImage()(INV_NORMALIZE(data[i].cpu()))
                    # image = T.ToPILImage()((data[i].cpu()))
                    # print("Saving image to {} with shape {}".format(os.path.join(epochviz, f"train_sample_{epoch}_{i}.png"), image.size))
                    image.save(os.path.join(epochviz, f"train_sample_{epoch}_{i}.png"))


def trainer_ffnn(model, loss_func, device, train_loader, optimizer, epoch, epochviz=False, diff_sizes=False):

    model.train()

    for batch_idx, (inputs, labels) in enumerate(train_loader):

        optimizer.zero_grad()
        labels = labels.to(device)
        inputs = inputs.to(device)
        outputs = model(inputs)
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs

        loss = loss_func(logits, labels)
        loss.backward()
        optimizer.step()

        wandb.log({"train/loss": loss.item()})
        if batch_idx % 50 == 0:
            print("Epoch {} Iteration {}: Loss = {}".format(
                str(epoch).zfill(3), str(batch_idx).zfill(4), loss))

                    
if __name__ == "__main__":

    # args
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir_path", type=str, required=True,
        help="Root image directory path, with character class subfolders")
    parser.add_argument("--train_ann_path", type=str, required=True,
        help="Path to COCO-style annotation file that localizer was trained on")
    parser.add_argument("--val_ann_path", type=str, required=True,
        help="Path to COCO-style annotation file that localizer was validated on")
    parser.add_argument("--test_ann_path", type=str, required=True,
        help="Path to COCO-style annotation file that localizer was tested on")
    parser.add_argument("--run_name", type=str, required=True,
        help="Name of run for W&B logging purposes")
    parser.add_argument('--batch_size', type=int, default=128,
        help="Batch size")
    parser.add_argument('--encoder',
        choices=['vit', 'xcit', 'beit', 'swin', 'bit', "convnext", "mae"], type=str, default="vit",
        help='DEPRECATED: specify type of encoder architecture (see `auto_model_X` args)')
    parser.add_argument('--lr', type=float, default=2e-6,
        help="LR for AdamW")
    parser.add_argument('--weight_decay', type=float, default=5e-4,
        help="Weight decay for AdamW")
    parser.add_argument('--num_epochs', type=int, default=5,
        help="Number of epochs")
    parser.add_argument('--temp', type=float, default=0.1,
        help="Temperature for InfoNCE loss")
    parser.add_argument('--start_epoch', type=int, default=1,
        help="Starting epoch")
    parser.add_argument('--m', type=int, default=4,
        help="m for m in m-class sampling")
    parser.add_argument('--imsize', type=int, default=224,
        help="Size of image for encoder")
    parser.add_argument("--hns_txt_path", type=str, default=None,
        help="Path to text file of mined hard negatives")
    parser.add_argument("--checkpoint", type=str, default=None,
        help="Load checkpoint before training")
    parser.add_argument("--lang", type=str, default="jp", choices=["jp", "en"],
        help="Language of characters being recognized")
    parser.add_argument('--finetune', action='store_true', default=False,
        help="Train just on target character crops")
    parser.add_argument('--pretrain', action='store_true', default=False,
        help="Train just on render character crops")
    parser.add_argument('--high_blur', action='store_true', default=False,
        help="Increase intensity of the blurring data augmentation for renders")
    parser.add_argument('--diff_sizes', action='store_true', default=False,
        help="DEPRECATED: allow different sizes for training crops")
    parser.add_argument('--epoch_viz_dir', type=str, default=None,
        help="Visualize and save some training samples by batch to this directory")
    parser.add_argument('--infer_hardneg_k', type=int, default=8,
        help="Infer k-NN hard negatives for each training sample, and save to a text file")
    parser.add_argument('--N_classes', type=int, default=None,
        help="Triggers use of FFNN classifier head with N classes")
    parser.add_argument('--test_at_end', action='store_true', default=False,
        help="Inference on test set at end of training with best val checkpoint")
    parser.add_argument("--auto_model_hf", type=str, default=None,
        help="Use model from HF by specifying model name")
    parser.add_argument("--auto_model_timm", type=str, default=None,
        help="Use model from timm by specifying model name")
    parser.add_argument("--num_passes", type=int, default=1,
        help="Defines epoch as number of passes of N_chars * M")
    parser.add_argument('--resize', action='store_true',
        help="Resize the image for encoder")
    parser.add_argument('--trans_epoch',action='store_true',default=False,
        help="Transform data from train loader on every epoch to vary the images slightly")
    parser.add_argument('--lambda_reg', type=float, default=0.1, help="Lambda for reg")
    
    args = parser.parse_args()

 

    print("Resizing: ", args.resize)

    # setup

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    os.makedirs(args.run_name, exist_ok=True)
    with open(os.path.join(args.run_name, "args_log.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # load encoder

    if args.auto_model_hf is None and args.auto_model_timm is None:
        if args.encoder == "vit" and args.N_classes is None:
            encoder = VitEncoder
        elif args.encoder == "xcit" and args.N_classes is None:
            encoder = XcitEncoder
        elif args.encoder == "beit":
            encoder = BeitEncoder
        elif args.encoder == "swin":
            encoder = SwinEncoder
        elif args.encoder == "convnext":
            encoder = ConvNextEncoder
        elif args.encoder == "bit":
            encoder = BitEncoder
        elif args.encoder == "mae":
            encoder = MaeEncoder
        elif args.encoder == "vit" and not args.N_classes is None:
            encoder = VitClassifier
        elif args.encoder == "xcit" and not args.N_classes is None:
            encoder = XcitClassifier
        else:
            raise NotImplementedError
    elif not args.auto_model_timm is None:
        encoder = AutoEncoderFactory("timm", args.auto_model_timm)
    elif not args.auto_model_hf is None:
        encoder = AutoEncoderFactory("hf", args.auto_model_hf)
    elif not args.auto_model_timm is None and not args.N_classes is None:
        encoder = AutoClassifierFactory("timm", args.auto_model_timm)
    elif not args.auto_model_hf is None and not args.N_classes is None:
        encoder = AutoClassifierFactory("hf", args.auto_model_hf)

    # init encoder

    if args.checkpoint is None and args.N_classes is None:
        if not args.auto_model_timm is None:
            enc = encoder(args.auto_model_timm)
        elif not args.auto_model_hf is None:
            enc = encoder(args.auto_model_hf)
        else:
            enc = encoder()
    elif args.checkpoint is None and not args.N_classes is None:
        if not args.auto_model_timm is None:
            enc = encoder(args.auto_model_timm, n_classes=args.N_classes)
        elif not args.auto_model_hf is None:
            enc = encoder(args.auto_model_hf, n_classes=args.N_classes)
        else:
            enc = encoder(n_classes=args.N_classes)
    elif not args.checkpoint is None and not args.N_classes is None:
        enc = encoder.load(args.checkpoint, n_classes=args.N_classes)
    else:
        enc = encoder.load(args.checkpoint)

    # data parallelism

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        datapara = True
        enc = nn.DataParallel(enc)
    else:
        datapara = False
    
        
 
    # set trainer

    trainer = trainer_knn if args.N_classes is None else trainer_ffnn # kNN vs. FFNN

    # warm start training
    def main():
        # load data
        # create dataset
        wandb.init(project="visual_record_linkage", name=args.run_name)

        train_dataset, val_dataset, test_dataset, \
            train_loader, val_loader, test_loader = create_dataset(
                args.root_dir_path, 
                args.train_ann_path,
                args.val_ann_path, 
                args.test_ann_path, 
                wandb.config.batch_size,
                hardmined_txt=args.hns_txt_path, 
                m=wandb.config.m,
                finetune=args.finetune,
                pretrain=args.pretrain,
                high_blur=args.high_blur,
                # lang=args.lang,
                knn=args.N_classes is None,
                diff_sizes=args.diff_sizes,
                imsize=args.imsize,
                num_passes=wandb.config.num_passes,
                resize=args.resize,
                renders=False,
                trans_epoch=args.trans_epoch
            )


        


        render_dataset = create_render_dataset(
            args.root_dir_path
        )

        # render_dataset_tk=create_render_dataset_only_tk(args.root_dir_path,"tk")
        # print(list(enumerate(render_dataset_tk.data)))

        train_image_folder=args.root_dir_path.split("/")[:-2]
        train_image_folder="/".join(train_image_folder)
        train_image_folder=train_image_folder+"/splits/train_images"
        # train_image_folder='/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/PaddleOCR_testing/Paddle_test_images/japan_label_df/splits/train_images'
        
        render_dataset_training = create_render_dataset(
            train_image_folder
        )

        

        # optimizer and loss


        # set tester

        if args.N_classes is None: # kNN classification
            if args.diff_sizes:
                tester = tester_knn_diff_sizes
            else:
                tester = tester_knn
            accuracy_calculator = AccuracyCalculator(include = ("precision_at_1",), k = 1)
        else:                      # FFNN classification
            tester = tester_ffnn
            idx_to_class = {v: chr(int(k)) for k, v in val_dataset.class_to_idx.items()}
            with open(os.path.join(args.run_name, "class_map.json"), "w") as f:
                json.dump(idx_to_class, f, indent=2)
            assert len(idx_to_class.keys()) == args.N_classes, \
                f"WARNING: specified number of classes {args.N_classes} disagrees with number of classes in dataset {len(idx_to_class.keys())}"

        val_dataset_only_partner=convert_dataset_to_partner_only(val_dataset)

        print("Training...")
           # get zero-shot accuracy

        print("Zero-shot accuracy:")
        if args.N_classes is None:
            print(val_dataset.data[0])


            best_acc = tester(val_dataset, render_dataset, enc, accuracy_calculator, "val", log=True)
        else:
            best_acc = tester(enc, val_dataset, val_loader, device, "val")

        # best_acc=0

        #USe ADAMW with LR scheduler

        

        optimizer = AdamW(enc.parameters(), lr=wandb.config.lr, weight_decay=wandb.config.weight_decay)
        # optimizer = SGD(params=enc.parameters(), lr=wandb.config.lr, weight_decay=wandb.config.weight_decay,momentum=wandb.config.momentum,)
        # optimizer=LBFGS(enc.parameters(), lr=wandb.config.lr, max_iter=20, max_eval=None, tolerance_grad=1e-05, tolerance_change=1e-09, history_size=100, line_search_fn=None)
        # optimizer=RMSprop(enc.parameters(), lr=wandb.config.lr, alpha=0.99, eps=1e-08, weight_decay=wandb.config.weight_decay, momentum=wandb.config.momentum, centered=False)
        # scheduler=CosineAnnealingLR(optimizer, 1)
        # scheduler=ExponentialLR(optimizer, gamma=0.5)
        # scheduler=None
        scheduler = CosineAnnealingWarmRestarts(optimizer, 1, 2)
        # scheduler = CyclicLR(optimizer, base_lr=0.000001, max_lr=0.01, step_size_up=20, step_size_down=None, mode='triangular', gamma=1.0, scale_fn=None, scale_mode='cycle', last_epoch=-1)
        # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
        loss_func=losses.SupConLoss(temperature = wandb.config.temp) if args.N_classes is None else CrossEntropyLoss()
        # loss_func=losses.CentroidTripletLoss(margin=wandb.config.margin) if args.N_classes is None else CrossEntropyLoss()
        # loss_func=losses.ArcFaceLoss(num_classes=len(list(set(train_dataset.targets))),embedding_size=768,margin=28.6,scale=64)
        loss_func_supcon = losses.SupConLoss(temperature = wandb.config.temp) if args.N_classes is None else CrossEntropyLoss()
        loss_func_infonce=losses.NTXentLoss(temperature = wandb.config.temp_nce) if args.N_classes is None else CrossEntropyLoss()


        if wandb.config.lambda_reg==0:
            regularization=None
        else:
            regularization=DifferentialEntropyRegularization()

        # # prototyping=True
        # train_dataset=torch.utils.data.Subset(train_dataset,range(0,300))
        # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=wandb.config.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
        # val_dataset=torch.utils.data.Subset(val_dataset,range(0,300))
        # val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=wandb.config.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
        # test_dataset=torch.utils.data.Subset(test_dataset,range(0,300))
        # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=wandb.config.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

        if not args.epoch_viz_dir is None: os.makedirs(args.epoch_viz_dir, exist_ok=True)
        for epoch in range(args.start_epoch, wandb.config.num_epochs+args.start_epoch):
            print("Training epoch", epoch)
            # manage_temp(10,wandb_log=True)
            ###Take a break after every 20 epochs
            # if epoch%20==0:
            #     print("Taking a break for 4 minutes")
            #     time.sleep(240)
            ##Inevery epoch, augment the training data if trans_epoch=True
            if args.trans_epoch:
            ##Transform trainloader images using the transform - create_random_doc_transform()
                print("Re-transforming images before training")
                train_loader_transformed=TransformLoader(train_loader,  AUG_NORMALIZE)
                trainer(enc, loss_func, device, train_loader_transformed, optimizer, epoch, args.epoch_viz_dir, args.diff_sizes,scheduler=scheduler,regularization=regularization,lambda_reg=wandb.config.lambda_reg,alpha=wandb.config.alpha,loss_func_supcon=loss_func_supcon,loss_func_infonce=loss_func_infonce)

            else:
                trainer(enc, loss_func, device, train_loader, optimizer, epoch, args.epoch_viz_dir, args.diff_sizes,scheduler=scheduler,regularization=regularization,lambda_reg=wandb.config.lambda_reg,alpha=wandb.config.alpha,loss_func_supcon=loss_func_supcon,loss_func_infonce=loss_func_infonce)
            
            
            if args.N_classes is None:
                if epoch > 10:
                    acc = tester(val_dataset, render_dataset, enc, accuracy_calculator, "val")
                if epoch % 5 == 0:
                    # train_acc=tester(train_dataset, train_dataset, enc, accuracy_calculator, "train")
                    # print("Train accuracy:",train_acc)
                    pass



            # else:
            #     acc = tester(enc, val_dataset, val_loader, device, "val")
                if epoch > 10:

                    if acc >= best_acc:
                        best_acc = acc
                        save_model(args.run_name, enc, f"best_{get_last_digit(wandb.config.alpha)}", datapara)
                        print("Best model saved")
                if epoch in  [30,60,90,140,180,210,240]:
                    model="vit_base_patch16_224.dino"
                    trained_model_path=os.path.join(args.run_name, f"enc_best_{get_last_digit(wandb.config.alpha)}.pth")
                    lang_code="TK"
                    root_folder="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/PaddleOCR_testing/Paddle_test_images/japan_vit"
                    top1_accu=infer_for_synth_match_japanese_all_tk.main(root_folder, model, trained_model_path , "TK", transform_type="custom",xcit=False,remove_train=True,recopy=True)
                    ###log top1_accu
                    wandb.log({"top1_accu":top1_accu})  

                



                    

        ##Del enc/ Save index
        if args.N_classes is None:
            best_enc = encoder.load(os.path.join(args.run_name, f"enc_best_{get_last_digit(wandb.config.alpha)}.pth"))
            
        else:
            best_enc = encoder.load(os.path.join(args.run_name, f"enc_best_{get_last_digit(wandb.config.alpha)}.pth"), n_classes=args.N_classes)

            # optionally test at end...
        print("Testing on all of TK")


        model="vit_base_patch16_224.dino"
        trained_model_path=os.path.join(args.run_name, f"enc_best_{get_last_digit(wandb.config.alpha)}.pth")
        lang_code="TK"
        root_folder="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/PaddleOCR_testing/Paddle_test_images/japan_vit"
        
        top1_accu=infer_for_synth_match_japanese_all_tk.main(root_folder, model, trained_model_path , lang_code, transform_type="custom",xcit=False,remove_train=True,recopy=False)
        ###log top1_accu
        wandb.log({"top1_accu":top1_accu})
            

        if args.test_at_end:
            if args.N_classes is None:
                test_acc = tester(test_dataset, render_dataset, best_enc, accuracy_calculator, "test")
                print(f"Final test acc: {test_acc}")
            else:
                test_acc = tester(enc, test_dataset, test_loader, device, "test")

                ##Save single font ref index
        

        if args.N_classes is None:
            save_ref_index(render_dataset_training, best_enc, args.run_name)

        ##Add ocr data evaluation
        print("Evaluating on paddle test set...")


        if args.hns_txt_path is None:
            print("Infering hard negatives...")
            if not args.infer_hardneg_k is None and args.N_classes is None:
                query_paths = glob.glob(train_image_folder+"/*/*")
                print(f"Num hard neg paths: {len(query_paths)}")
                transform = BASE_TRANSFORM
                infer_hardneg(query_paths, render_dataset_training, best_enc, 
                    os.path.join(args.run_name, "ref.index"), 
                    transform, os.path.join(args.run_name, "hns.txt"), os.path.join(args.run_name, "check_hns.txt"),
                    k=args.infer_hardneg_k)



    # main()

    # #     ##Set up sweep
    # sweep_configuration = {[]
    # 'method': 'random',
    # 'name': 'sweep',
    # 'metric': {'goal': 'maximize', 'name': 'val/accuracy'},
    # 'parameters': 
    # {
    #     'lr': {'max': 0.0003, 'min': 0.000002},
    #     'weight_decay': {'max': 0.5, 'min': 0.0002},
    #     'temp': {'max': 0.5 , 'min': 0.00006},
    #     'm':{ 'values': [3]},
    #     'batch_size': {'values': [126]},
    #     'lambda_reg': {'values': [0,0.1,0.2,0.3,0.4,0.5,f0.6,0.7,0.8]}
    #  }
    # }

    # sweep_configuration = {
    # 'method': 'random',
    # 'name': 'sweep',
    # 'metric': {'goal': 'maximize', 'name': 'val/accuracy'},
    # 'parameters': 
    # {
    #     'lr': {'values': [0.00002449376025029932]},
    #     'weight_decay': {'values': [0.4686]},
    #     'temp': {'values': [0.0315]},
    #     'm':{ 'values': [3]},
    #     'batch_size': {'values': [126]},
    #     'lambda_reg': {'values': [0.3]},
    #  }
    # }

    # sweep_configuration = {
    # 'method': 'random',
    # 'name': 'sweep',
    # 'metric': {'goal': 'maximize', 'name': 'val/accuracy'},
    # 'parameters': 
    # {
    #     'lr': {'values': [0.00002449376025029932]},
    #     'weight_decay': {'values': [0.4686]},
    #     'temp': {'values': [0.0315]},
    #     'm':{ 'values': [3]},
    #     'batch_size': {'values': [126]},
    #     'lambda_reg': {'values': [0]},
    #     'temp_nce': {'values': [0.0315]},
    #     'alpha': {'values': [None]}
    #  }
    # }
    ###SGD
    sweep_configuration = {
    'method': 'random',
    'name': 'sweep',
    'early_terminate': {'type': 'hyperband', 'min_iter': 1},
    'metric': {'goal': 'maximize', 'name': 'val/accuracy'},
    'parameters': 
    {   
        'num_epochs': {'values': [200]},
        'lr': {'values': [0.000002]},
        'weight_decay': {'values': [0.1]},
        'temp': {'values': [0.09]},
        # 'temp': {'values': [0.09]},
        'm':{ 'values': [3]},
        'batch_size': {'values': [252]},
        'lambda_reg': {'values': [0]},
        'temp_nce': {'values': [0.115]},
        'alpha': {'values': [None]},
        'momentum': {'values': [0.9]},
        'margin': {'values': [0.05]},
        'num_passes': {'values': [1]},
     }
    }


    sweep_id = wandb.sweep(sweep_configuration, project="visual_record_linkage")

    wandb.agent(sweep_id, function=main, count=1)
    # main()


    

