import random
import numpy as np
import math
import torch
import torch.optim as optim
from config import get_args
from model import get_model
from utils import setup_seed, evaluation, compute_local_test_accuracy, mlc_KLDiv
from prepare_data import get_dataloader
from gen_utils.generate_utils import KLDiv
from gen_utils.generate_image import ImageSynthesizer
import time


def local_train_fedgen(args, cfg, nets_this_round, global_model, train_local_dls, test_dl, traindata_cls_counts):
    global_model.cuda()
    for net_id, net in nets_this_round.items():
        net.cuda()

        num_classes = cfg['classes_size']

        # --- FedHUG: dynamic synthesis budget (client-wise) based on heterogeneity ---
        with torch.no_grad():
            gw = global_model.state_dict()
            lw = net.state_dict()
            diff_sq = 0.0
            ref_sq = 0.0
            for k, g_t in gw.items():
                l_t = lw.get(k, None)
                if l_t is None:
                    continue
                if not torch.is_floating_point(g_t):
                    continue
                diff = (l_t.float() - g_t.float())
                diff_sq += diff.pow(2).sum().item()
                ref_sq += g_t.float().pow(2).sum().item()
            weight_divergence = math.sqrt(diff_sq) / (math.sqrt(ref_sq) + 1e-12)

        scale = 1.0 + args.ua_budget_alpha * weight_divergence
        scale = max(args.ua_budget_min_scale, min(args.ua_budget_max_scale, scale))
        synthesis_bs = max(num_classes, int(math.floor(args.synthesis_batch_size * scale + 0.5)))

        # ---  FedHUG: uncertainty-aware class distribution  ---
        def _estimate_class_entropy_scores(model, dl, num_classes, max_batches):
            model.eval()
            ent_sum = torch.zeros(num_classes, device='cuda')
            cnt = torch.zeros(num_classes, device='cuda')
            batches = 0
            for x_b, y_b in dl:
                x_b = x_b.cuda(non_blocking=True)
                y_b = y_b.cuda(non_blocking=True).long()
                with torch.no_grad():
                    logits = model(x_b)
                    p = torch.softmax(logits, dim=1)
                    ent = -(p * torch.log(p + 1e-12)).sum(dim=1)
                for c in range(num_classes):
                    m = (y_b == c)
                    if m.any():
                        ent_sum[c] += ent[m].sum()
                        cnt[c] += m.sum()
                batches += 1
                if batches >= max_batches:
                    break
            max_ent = math.log(num_classes)
            ent_avg = ent_sum / torch.clamp(cnt, min=1.0)
            ent_avg = torch.where(cnt > 0, ent_avg, torch.full_like(ent_avg, max_ent))
            ent_min = ent_avg.min()
            ent_max = ent_avg.max()
            ent_norm = (ent_avg - ent_min) / (ent_max - ent_min + 1e-12)
            return ent_norm.detach().cpu().numpy()

        entropy_scores = _estimate_class_entropy_scores(
            net, train_local_dls[net_id], num_classes, args.ua_entropy_batches
        )

        cls_counts = traindata_cls_counts[net_id].astype(np.float32)
        scarcity = cls_counts.max() - cls_counts
        raw = scarcity + args.ua_uncertainty_beta * (entropy_scores * (scarcity.max() + 1.0))
        raw = np.clip(raw, a_min=1e-6, a_max=None)
        distribution = raw / raw.sum()
        # --- FedHUG：set λkd according to the relative amount of synthetic completion ---
        num_real_data = float(cls_counts.sum())
        gap = cls_counts.max() - cls_counts
        num_syn_proxy = float(gap.sum())

        syn_lr = num_syn_proxy / (num_real_data + num_syn_proxy + 1e-12)
        real_lr = 1.0 - syn_lr

        num_syn_data = synthesis_bs  
        print(f"[FedHUG] weight_divergence={weight_divergence:.6f}, synthesis_batch_size={synthesis_bs}")
        print("num_real_data: {}, num_syn_data:{}, real_lr:{}, syn_lr:{}".format(
            num_real_data, num_syn_data, real_lr, syn_lr
        ))
        print("gen data distribution:", distribution)
        print(f"num_syn_proxy:{num_syn_proxy}, syn_lr:{syn_lr}")
        
        # --- FedHUG：generate synthetic data ---
        synthesizer = ImageSynthesizer(
            args, global_model, net,
            nz=args.nz, num_classes=cfg['classes_size'], img_size=cfg["image_size"],
            iterations=args.g_steps, lr_g=args.lr_g,
            synthesis_batch_size=synthesis_bs,
            sample_batch_size=args.batch_size,
            adv=args.adv, bn=args.bn, oh=args.oh,
            dataset=args.dataset, distribution=distribution
        )

        gen_dataloader = synthesizer.synthesize()

        net.load_state_dict(global_model.state_dict())
        ce_criterion = torch.nn.CrossEntropyLoss().cuda()
        kd_criterion = KLDiv(T=args.T)
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.reg)

        net.train()
        global_model.eval()

        iterator = iter(train_local_dls[net_id])
        gen_iterator = iter(gen_dataloader)
        for iteration in range(args.num_local_iterations):

            if iteration == int(args.num_local_iterations / 2) and args.double_gen:
                gen_dataloader = synthesizer.synthesize()
                net.train()
                gen_iterator = iter(gen_dataloader)

            try:
                x, target = next(iterator)
            except StopIteration:
                iterator = iter(train_local_dls[net_id])
                x, target = next(iterator)

            x, target = x.cuda(), target.cuda()

            optimizer.zero_grad()
            target = target.long()

            out = net(x)
            loss = real_lr * ce_criterion(out, target)
            loss.backward()

            try:
                images, t_out = next(gen_iterator)
            except StopIteration:
                gen_iterator = iter(gen_dataloader)
                images, t_out = next(gen_iterator)
            images, t_out = images.cuda(), t_out.cuda()

            s_out = net(images.detach())
            loss_s = syn_lr * kd_criterion(s_out, t_out)  # TODO: not suitable for cifar10
            loss_s.backward()

            optimizer.step()

        net.to('cpu')


def local_train_fedavg(args, nets_this_round, train_local_dls):

    for net_id, net in nets_this_round.items():
        train_local_dl = train_local_dls[net_id]

        # Set Optimizer
        if args.optimizer == 'adam':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, weight_decay=args.reg)
        elif args.optimizer == 'amsgrad':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, weight_decay=args.reg,
                                   amsgrad=True)
        elif args.optimizer == 'sgd':
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, momentum=0.9,
                                  weight_decay=args.reg)

        criterion = torch.nn.CrossEntropyLoss().cuda()
        net.cuda()
        net.train()

        iterator = iter(train_local_dl)
        for iteration in range(args.num_local_iterations):
            try:
                x, target = next(iterator)
            except StopIteration:
                iterator = iter(train_local_dl)
                x, target = next(iterator)

            x, target = x.cuda(), target.cuda()

            optimizer.zero_grad()
            target = target.long()

            out = net(x)
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()

        net.to('cpu')


args, cfg = get_args()
print(args)
setup_seed(args.init_seed)

n_party_per_round = int(args.n_parties * args.sample_fraction)
party_list = [i for i in range(args.n_parties)]
party_list_rounds = []
if n_party_per_round != args.n_parties:
    for i in range(args.comm_round):
        party_list_rounds.append(random.sample(party_list, n_party_per_round))
else:
    for i in range(args.comm_round):
        party_list_rounds.append(party_list)

train_local_dls, test_dl, client_num_samples, traindata_cls_counts, data_distributions = get_dataloader(args)

model = get_model(args)

global_model = model(cfg['classes_size'])
local_models = []

for i in range(args.n_parties):
    local_models.append(model(cfg['classes_size']))

# load checkpoint
load_round = 0
if args.load_path is not None:
    ckpt = torch.load(args.load_path)
    global_model.load_state_dict(ckpt)
    load_round = int(args.load_path.split('_')[-2])
    print(f'>> Initial info: {args.load_path}')

best_acc = 0
for round in range(args.comm_round):  # Federated round loop
    if round < load_round:
        continue
    party_list_this_round = party_list_rounds[round]
    if args.sample_fraction < 1.0:
        print(f'>> Clients in this round : {party_list_this_round}')
    global_w = global_model.state_dict()  # Global Model Initialization

    nets_this_round = {k: local_models[k] for k in party_list_this_round}

    # Local Model Training
    if round < args.start_round:
        for net in nets_this_round.values():
            net.load_state_dict(global_w)
        local_train_fedavg(args, nets_this_round, train_local_dls)
    else:
        local_train_fedgen(args, cfg, nets_this_round, global_model, train_local_dls, test_dl, traindata_cls_counts)

    # Aggregation Weight Calculation
    total_data_points = sum([client_num_samples[r] for r in party_list_this_round])
    fed_avg_freqs = [client_num_samples[r] / total_data_points for r in party_list_this_round]
    if round == 0 or args.sample_fraction < 1.0:
        print(f'Dataset size weight : {fed_avg_freqs}')

    # Model Aggregation
    for net_id, net in enumerate(nets_this_round.values()):
        net_para = net.state_dict()
        if net_id == 0:
            for key in net_para:
                global_w[key] = net_para[key] * fed_avg_freqs[net_id]
        else:
            for key in net_para:
                global_w[key] += net_para[key] * fed_avg_freqs[net_id]

    global_model.load_state_dict(global_w)  # Update the global model
    acc, best_acc = evaluation(args, global_model, test_dl, best_acc, round)

    import os

    if (round + 1) % args.comm_round == 0 and args.save_model:
        save_dir = f"./models/saved_model/{args.dataset}"
        os.makedirs(save_dir, exist_ok=True)
        torch.save(
            global_w,
            f"{save_dir}/fedhug_{args.dataset}_{args.model}_{args.partition}{args.beta}"
            f"_c{args.n_parties}_it{args.num_local_iterations}"
            f"_p{args.sample_fraction}_{round + 1}_{acc:.5f}.pkl"
        )
