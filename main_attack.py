
# advwt_test_only.py
"""
Adversarial Wear and Tear (test-only runner)
Loads the generator (via Solver), a target classifier, and crafts attacks
by searching over style codes. Saves successful adversarials and reports
mean query count.
"""

import os
import argparse
import numpy as np
from numpy import random as rd
from PIL import Image

import torch
import torch.nn.functional as F
from torch.backends import cudnn
from torch.autograd import Variable

import torchvision.utils as vutils
from torchvision import transforms

from torchmetrics.image import StructuralSimilarityIndexMeasure

from core.solver import Solver
from core.target_model_traffic import densenet_hybrid

# ----------------------------- helpers ----------------------------- #

@torch.no_grad()
def test_single(nets, image, masks, s):
    """Render one fake image with a given style code s."""
    fake = nets.generator(image, s, masks=masks)
    return torch.clamp(fake * 0.5 + 0.5, 0, 1)

@torch.no_grad()
def obj_evo_new_new(cand_style_np, _, nets, image, masks, target_model, im_name, save_dir, _pert_val):
    """
    Score a candidate style vector:
    returns (prob_true_label, prob_pred_label, pred_label, still_correct)
    """
    s = torch.tensor(np.asarray(cand_style_np).astype('float32')).unsqueeze(0).to(image.device)
    fake = nets.generator(image, torch.nan_to_num(s), masks=masks)
    fake = torch.clamp(fake * 0.5 + 0.5, 0, 1)

    logits = target_model(fake)
    probs  = F.softmax(logits, dim=1)

    pred = probs.argmax(1).item()
    true = int(os.path.basename(im_name).split('_')[0])

    prob_true = probs[0, true].item()
    prob_pred = probs[0, pred].item()
    still_ok  = (pred == true)  # True means not yet adversarial
    return prob_true, prob_pred, pred, still_ok

@torch.no_grad()
def test_single_presty_evo_new_new(
    nets, image, masks, s, target_model, save_dir, im_name, *,
    n_bits=64, n_pop=30
):
    """
    Population search around style code s. Returns:
      success_flag (0/1), used_pert_value (float or None)
    """
    print(im)
    ssim = StructuralSimilarityIndexMeasure()
    n_bits = 64
    n_pop = 30

    new_sty = torch.clone(s)
    new_sty1 = torch.clone(s)
    pl = torch.clone(s)
    base_abs = torch.abs(new_sty)

    # perturbation levels correspond to your query buckets (30 queries per step)
    pert_range = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                  1.0, 1.1, 1.2, 1.3, 1.4, 1.5]

    ln = 0
    # rank_s = torch.argsort(torch.abs(s), descending=True, stable=True)
    rank_s = torch.argsort(torch.abs(pl), descending=True, stable=True)
    oo = 0

    for ii in pert_range:
        # box sampling around current style (per-channel range proportional to |s|)
        delta      = (torch.ones(1, 64) * torch.tensor((ii)).unsqueeze(0)).cuda() * torch.abs(new_sty1)
        low_range  = (new_sty - delta).cpu()
        high_range = (new_sty + delta).cpu()

        # added_vals = (decay_fac.squeeze()[rank_s.squeeze()]).unsqueeze(0) * new_sty
        # high_range = (new_sty + added_vals).detach().cpu()
        # low_range = (new_sty - added_vals).detach().cpu()

        # generate a population of candidates
        pop = [[rd.uniform(low_range[:, k], high_range[:, k], 1)[0] for k in range(n_bits)]
               for _ in range(n_pop)]

        pk=True

        # evaluate population
        scores, scores1, vals, lp = zip(
            *([obj_evo_new_new(c, mn, nets, image, masks, target_model, im, save_dir, ii) for mn, c in enumerate(pop)]))
        lk = np.where(scores == min(scores))
        print(min(scores))

        pop_un = lk[0][0]
        # print(pop_un)
        lm = pop[pop_un]
        res = all(i for i in lp)
        lp11 = np.array((lp))
        scores = np.array((scores))
        if not res:
            pk = False
            org_label = int(im.split('_')[0])
            vals11 = np.concatenate(vals)
            kkl = np.where(vals11 != org_label)

            pred_ch = np.where(lp11 == False)
            # print('pred_ch', pred_ch)

            scores_false = [scores[item] for item in pred_ch]
            # print('scores_false', scores_false)
            plk = np.max(scores_false)
            # print('plk', plk)
            lmk = np.where(scores == plk)
            lmk = np.array(lmk)

            if len(lmk[0]) < 30 and len(lmk[0])>1:
                print('kkl', kkl)
                print('kkl1', kkl[0])
                print('lmk', lmk[0])
                pop_un11 = kkl[0][lmk[0]]
                print('new_label', vals[pop_un11])
                scol = scores[pop_un11]
                # print(pop_un)
                lm11 = pop[pop_un11]

                print('img became adv,  breaking the loop', ii)
                s11 = torch.tensor(np.asarray(lm11).astype('float32')).unsqueeze(0).cuda()
                # outputs = interpolations_new(nets,image,pl,s11,masks,lerp_mode='lerp')

                fake = nets.generator(image, torch.nan_to_num(s11), masks=masks)
                # print('pl',pl)
                fake = torch.clamp(fake * 0.5 + 0.5, 0, 1)
                image11 = torch.clamp(image * 0.5 + 0.5, 0, 1)
                # for i, j in zip(mm, nn):
                #    fake[:, :, i, j] = image11[:, :, i, j]

                f_fake_logits = target_model(fake)
                f_fake_probs = F.softmax(f_fake_logits, dim=1)
                fake_labels = torch.argmax(f_fake_probs, 1)
                org_label = int(im.split('_')[0])
                # org_label = 919
                # org_label=2
                fake_labels = fake_labels.cpu().numpy()[0]
                if fake_labels != org_label:
                    oo = 1
                    print('success')

                # scr = f_fake_probs[:, org_label].cpu().detach().numpy()[0]

                min_confidence = f_fake_probs[0, fake_labels].item()
                path = os.path.join(save_dir,
                                    str(im.split('.')[0]) + '_' + str(org_label) + '_' + str(fake_labels) + '_' + str(
                                        min_confidence) + '.png')
                ssim_sc = ssim(fake.cpu(), torch.clamp(image * 0.5 + 0.5, 0, 1).cpu())
                ssim_sc = ssim_sc.cpu().detach().numpy()
                print('ss', ssim_sc)
                if ii in yr:
                    yr[ii].append(ssim_sc)
                    yr1[ii].append(im)
                else:

                    yr[ii] = [ssim_sc]
                    yr1[ii] = [im]
                # path = os.path.join(save_dir,
                #                    str(im.split('.')[0]) +'_fake_'+str(vals[pop_un11][0])+'_'+str(scol)+ '.png')

                vutils.save_image(fake.data, path, padding=0)
            
            else:

                pop_un11 = kkl[0][0]
                print('new_label', vals[pop_un11])
                scol = scores[pop_un11]
                # print(pop_un)
                lm11 = pop[pop_un11]

                print('img became adv,  breaking the loop', ii)
                s11 = torch.tensor(np.asarray(lm11).astype('float32')).unsqueeze(0).cuda()
                # outputs = interpolations_new(nets,image,pl,s11,masks,lerp_mode='lerp')

                fake = nets.generator(image, torch.nan_to_num(s11), masks=masks)
                # print('pl',pl)
                fake = torch.clamp(fake * 0.5 + 0.5, 0, 1)
                image11 = torch.clamp(image * 0.5 + 0.5, 0, 1)
                # for i, j in zip(mm, nn):
                #    fake[:, :, i, j] = image11[:, :, i, j]

                f_fake_logits = target_model(fake)
                f_fake_probs = F.softmax(f_fake_logits, dim=1)
                fake_labels = torch.argmax(f_fake_probs, 1)
                org_label = int(im.split('_')[0])
                # org_label = 919
                # org_label=2
                fake_labels = fake_labels.cpu().numpy()[0]
                if fake_labels != org_label:
                    oo = 1
                    print('success')

                # scr = f_fake_probs[:, org_label].cpu().detach().numpy()[0]

                min_confidence = f_fake_probs[0, fake_labels].item()
                path = os.path.join(save_dir,
                                    str(im.split('.')[0]) + '_' + str(org_label) + '_' + str(fake_labels) + '_' + str(
                                        min_confidence) + '.png')
                ssim_sc = ssim(fake.cpu(), torch.clamp(image * 0.5 + 0.5, 0, 1).cpu())
                ssim_sc = ssim_sc.cpu().detach().numpy()
                print('ss', ssim_sc)
                if ii in yr:
                    yr[ii].append(ssim_sc)
                    yr1[ii].append(im)
                else:

                    yr[ii] = [ssim_sc]
                    yr1[ii] = [im]
                # path = os.path.join(save_dir,
                #                    str(im.split('.')[0]) +'_fake_'+str(vals[pop_un11][0])+'_'+str(scol)+ '.png')

                vutils.save_image(fake.data, path, padding=0)
            if pk == False:
                ln = 1

                return ln, yr, yr1, oo, ii
            new_sty = torch.tensor(lm).unsqueeze(0).cuda()
        return ln, yr, yr1, oo, ii


# ----------------------------- main (test only) ----------------------------- #

def main(args):
    cudnn.benchmark = True
    os.makedirs(args.save_dir, exist_ok=True)

    # 1) Load the pretrained generator (EMA) through Solver
    solver = Solver(args)
    solver._load_checkpoint(args.resume_iter)
    nets_ema = solver.nets_ema

    # 2) Target classifier
    target_model = densenet_hybrid(False).cuda()
    target_model.load_state_dict(torch.load(args.target_ckpt))
    target_model.eval()

    # 3) Style code (npy)
    s = torch.tensor(np.load(args.style_code_path).astype('float32')).cuda()

    # 4) Preprocess
    transform = transforms.Compose([
        transforms.Resize([args.img_size, args.img_size]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])
    y2 = torch.tensor([args.y2]).long().cuda()
    y1 = torch.tensor([args.y1]).long().cuda()

    # Each perturbation step corresponds to 30 queries in your analysis
    pert_range = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    Q_PER_STEP = 30
    mj = 0

    years_dict = dict()
    years_dict1 = dict()
 
    kk = os.listdir(args.input_test)
    mkk = 0
    oo = 0
    total_queries=0
    tot_que=[]
    for ikl, im in enumerate(kk):

    tot_queries, per_image_queries = 0, []

    images = sorted(os.listdir(args.input_test))
    for im in images:
        path  = os.path.join(args.input_test, im)
        image = Variable(transform(Image.open(path).convert('RGB')).unsqueeze(0)).cuda()

        masks = nets_ema.fan.get_heatmap(image) if args.w_hpf > 0 else None

        # (optional) first render with the base style
        _ = test_single(nets_ema, image, masks, s)

        kj, yr, yr1, oo, quer = test_single_presty_evo_new_new(nets_ema, image, masks, s1, target_model, save_dir, im, y2,
                                                             years_dict, years_dict1, oo)

        if used_pert is None:
            queries = len(pert_range) * Q_PER_STEP
        else:
            step_idx = pert_range.index(used_pert)
            queries  = (step_idx + 1) * Q_PER_STEP

        tot_queries += queries
        per_image_queries.append(queries)

        print(f"[{im}] success={bool(success)}  queries={queries}")

    print(f"\nTotal images: {len(images)}")
    print(f"Average queries/image: {np.mean(per_image_queries):.2f}")
    print(f"Total queries: {tot_queries}")

if __name__ == "__main__":
    p = argparse.ArgumentParser("AWT test-only")
    # generator / solver bits (Solver may expect these fields)
    p.add_argument("--checkpoint_dir", type=str, default="expr/checkpoints")
    p.add_argument("--resume_iter",   type=int, default=100000)

    # inference I/O
    p.add_argument("--input_test",       type=str, required=True,
                   help="Folder with test images")
    p.add_argument("--save_dir",         type=str, default="expr/results_awt")
    p.add_argument("--img_size",         type=int, default=224)
    p.add_argument("--w_hpf",            type=float, default=0.0)

    # target model + style code
    p.add_argument("--target_ckpt",      type=str, required=True,
                   help="Path to target model checkpoint (.pth)")
    p.add_argument("--style_code_path",  type=str, required=True,
                   help="Path to a numpy .npy style code (shape [1,64])")

    # the fields below are typically referenced by Solver/net builder; keep defaults
    p.add_argument("--latent_dim", type=int, default=16)
    p.add_argument("--style_dim",  type=int, default=64)
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--num_domains", type=int, default=2)
    p.add_argument("--seed", type=int,   default=777)

    args = p.parse_args()
    main(args)


