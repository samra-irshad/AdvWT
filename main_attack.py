
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
    ssim = StructuralSimilarityIndexMeasure().to(image.device)

    new_sty  = s.clone()
    base_abs = torch.abs(new_sty)

    # perturbation levels correspond to your query buckets (30 queries per step)
    pert_range = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                  1.0, 1.1, 1.2, 1.3, 1.4, 1.5]

    for pert in pert_range:
        # box sampling around current style (per-channel range proportional to |s|)
        delta      = (torch.ones_like(new_sty) * pert) * base_abs
        low_range  = (new_sty - delta).cpu()
        high_range = (new_sty + delta).cpu()

        # generate a population of candidates
        pop = [[rd.uniform(low_range[:, k], high_range[:, k], 1)[0] for k in range(n_bits)]
               for _ in range(n_pop)]

        # evaluate population
        scores_true, scores_pred, preds, still_ok = zip(*[
            obj_evo_new_new(c, i, nets, image, masks, target_model, im_name, save_dir, pert)
            for i, c in enumerate(pop)
        ])

        # success? pick the candidate with the lowest true-label prob among the successful ones
        if not all(still_ok):
            scores_true = np.array(scores_true)
            still_ok    = np.array(still_ok)
            idx         = np.argmin(scores_true[~still_ok])
            cand_idx    = np.arange(len(pop))[~still_ok][idx]

            # render & save representative adversarial
            best = torch.tensor(np.asarray(pop[cand_idx]).astype('float32')).unsqueeze(0).to(image.device)
            adv  = nets.generator(image, torch.nan_to_num(best), masks=masks)
            adv  = torch.clamp(adv * 0.5 + 0.5, 0, 1)

            # (optional) SSIM log if you care
            _ssim = ssim(adv, torch.clamp(image * 0.5 + 0.5, 0, 1)).item()

            out_path = os.path.join(
                save_dir, f"{os.path.splitext(os.path.basename(im_name))[0]}_pert{pert:.1f}.png"
            )
            vutils.save_image(adv, out_path, padding=0)

            return 1, pert  # success and which perturbation step caused it

        # no success at this level: move the center to the best candidate (lowest true prob)
        best_idx = int(np.argmin(scores_true))
        new_sty  = torch.tensor(np.asarray(pop[best_idx]).astype('float32')).unsqueeze(0).to(image.device)

    return 0, None  # never flipped

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

    # Each perturbation step corresponds to 30 queries in your analysis
    pert_range = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    Q_PER_STEP = 30

    tot_queries, per_image_queries = 0, []

    images = sorted(os.listdir(args.input_test))
    for im in images:
        path  = os.path.join(args.input_test, im)
        image = Variable(transform(Image.open(path).convert('RGB')).unsqueeze(0)).cuda()

        masks = nets_ema.fan.get_heatmap(image) if args.w_hpf > 0 else None

        # (optional) first render with the base style
        _ = test_single(nets_ema, image, masks, s)

        success, used_pert = test_single_presty_evo_new_new(
            nets_ema, image, masks, s, target_model, args.save_dir, im
        )

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
    p.add_argument("--seed",            type=int,   default=777)

    args = p.parse_args()
    main(args)
