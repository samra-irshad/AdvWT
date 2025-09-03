"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import os
from os.path import join as ospj
import time
import datetime
from munch import Munch
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from .target_model import resnet18
from core.model import build_model
from core.checkpoint import CheckpointIO
from core.data_loader import InputFetcher
import core.utils as utils
from metrics.eval import calculate_metrics
from torchvision.models import resnet50
from metrics.lpips import LPIPS

class Solver(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
       ###########################################
        self.target_model = resnet50(pretrained=True)
        #self.target_model = self.target_model.cuda()
        #self.target_model.load_state_dict(
        #    torch.load("../awt/checkpoints/resnet/best_model_resnet18_224_14clsnew_now_latest_0.9904" + ".pth"))
        self.target_model.eval()
        ###########################################
        self.nets, self.nets_ema = build_model(args)

        # below setattrs are to make networks be children of Solver, e.g., for self.to(self.device)
        for name, module in self.nets.items():
            utils.print_network(module, name)
            setattr(self, name, module)
        for name, module in self.nets_ema.items():
            setattr(self, name + '_ema', module)

        if args.mode == 'train':
            self.optims = Munch()
            for net in self.nets.keys():

                if net == 'fan':
                    continue
                self.optims[net] = torch.optim.Adam(
                    params=self.nets[net].parameters(),
                    lr=args.f_lr if net == 'mapping_network' else args.lr,
                    betas=[args.adam_beta1, args.adam_beta2],
                    weight_decay=args.weight_decay)

            self.ckptios = [
                CheckpointIO(args.checkpoint_dir+'/'+ '{:06d}_nets.ckpt', **self.nets),
                CheckpointIO(args.checkpoint_dir+'/'+ '{:06d}_nets_ema.ckpt', **self.nets_ema),
                CheckpointIO(args.checkpoint_dir+'/'+ '{:06d}_optims.ckpt', **self.optims)]
        else:
            self.ckptios = [CheckpointIO(args.checkpoint_dir+'/'+ '{:06d}_nets_ema.ckpt', **self.nets_ema)]

        self.to(self.device)
        for name, network in self.named_children():
            # Do not initialize the FAN parameters
            if ('ema' not in name) and ('fan' not in name):
                print('Initializing %s...' % name)
                network.apply(utils.he_init)

        self.latent_dim  = args.latent_dim
        self.num_domains = args.num_domains

    def _save_checkpoint(self, step):
        for ckptio in self.ckptios:
            ckptio.save(step)

    def _load_checkpoint(self, step):
        for ckptio in self.ckptios:
            ckptio.load(step)

    def _reset_grad(self):
        for optim in self.optims.values():
            optim.zero_grad()

    def train(self, loaders):
        args = self.args
        nets = self.nets
        nets_ema = self.nets_ema
        optims = self.optims

        # fetch random validation images for debugging
        fetcher = InputFetcher(loaders.src, loaders.ref, self.latent_dim, 'train') ##latent_dim: 16
        fetcher_val = InputFetcher(loaders.val, None, self.latent_dim, 'val')
        inputs_val = next(fetcher_val)

        # resume training if necessary
        if args.resume_iter > 0:
            self._load_checkpoint(args.resume_iter)

        # remember the initial value of ds weight
        initial_lambda_ds = args.lambda_ds
        lpips = LPIPS(args.dist_mode).eval().cuda() if args.lambda_lpips > 0 else None

        print('Start training...')
        start_time = time.time()
        for i in range(args.resume_iter, args.total_iters):
            # fetch images and labels
            inputs = next(fetcher)
            x_real, y_org = inputs.x_src, inputs.y_src

            x_ref, x_ref2, y_trg = inputs.x_ref, inputs.x_ref2, inputs.y_ref

            z_trg, z_trg2 = inputs.z_trg, inputs.z_trg2


            masks = nets.fan.get_heatmap(x_real) if args.w_hpf > 0 else None

            # train the discriminator
            d_loss, d_losses_latent = compute_d_loss(
                nets, args, x_real, y_org, y_trg, z_trg=z_trg, masks=masks)
            self._reset_grad()
            d_loss.backward()
            optims.discriminator.step()

            d_loss, d_losses_ref = compute_d_loss(
                nets, args, x_real, y_org, y_trg, x_ref=x_ref, masks=masks)
            self._reset_grad()
            d_loss.backward()
            optims.discriminator.step()

            # train the generator
            g_loss, g_losses_latent = compute_g_loss(
                nets, args, x_real, y_org, y_trg, z_trgs=[z_trg, z_trg2], 
                masks=masks, lpips=lpips)
            self._reset_grad()
            g_loss.backward()
            optims.generator.step()
            optims.mapping_network.step()
            optims.style_encoder.step()

            g_loss, g_losses_ref = compute_g_loss(
                nets, args, x_real, y_org, y_trg, x_refs=[x_ref, x_ref2], 
                masks=masks, lpips=lpips)
            self._reset_grad()
            g_loss.backward()
            optims.generator.step()

            # compute moving average of network parameters
            moving_average(nets.generator, nets_ema.generator, beta=0.999)
            moving_average(nets.mapping_network, nets_ema.mapping_network, beta=0.999)
            moving_average(nets.style_encoder, nets_ema.style_encoder, beta=0.999)

            # decay weight for diversity sensitive loss
            if args.lambda_ds > 0:
                args.lambda_ds -= (initial_lambda_ds / args.ds_iter)
            if (args.lambda_kl > args.init_lambda_kl) and (i+1 > args.kl_start_iter):
                args.init_lambda_kl += (args.lambda_kl / args.kl_iter)

            # save model checkpoints
            if (i+1) % args.save_every == 0:
                self._save_checkpoint(step=i+1)

            # print out log info
            if (i+1) % args.print_every == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))[:-7]
                log = "Elapsed time [%s], Iteration [%i/%i], " % (elapsed, i+1, args.total_iters)
                all_losses = dict()
                for loss, prefix in zip([d_losses_latent, d_losses_ref, g_losses_latent, g_losses_ref],
                                        ['D/latent_', 'D/ref_', 'G/latent_', 'G/ref_']):
                    for key, value in loss.items():
                        all_losses[prefix + key] = value
                all_losses['G/lambda_ds'] = args.lambda_ds
                if args.lambda_kl > 0:
                    all_losses['G/init_lambda_kl'] = args.init_lambda_kl
                
                for opt in optims.keys():
                    all_losses["lr/{}".format(opt)] = optims[opt].param_groups[0]['lr']
                log += ' '.join(['%s: [%.4f]' % (key, value) for key, value in all_losses.items()])
                print(log)

            # generate images for debugging
            if (i+1) % args.sample_every == 0:
                os.makedirs(args.sample_dir, exist_ok=True)
                utils.debug_image(nets_ema, args, inputs=inputs_val, step=i+1)

            # compute FID and LPIPS if necessary
            if (i+1) % args.eval_every == 0:
                calculate_metrics(nets_ema, args, i+1, mode='latent')
                calculate_metrics(nets_ema, args, i+1, mode='reference')

    @torch.no_grad()
    def sample(self, loaders):
        args = self.args
        nets_ema = self.nets_ema
        os.makedirs(args.result_dir, exist_ok=True)
        self._load_checkpoint(args.resume_iter)

        src = next(InputFetcher(loaders.src, None, args.latent_dim, 'test'))
        ref = next(InputFetcher(loaders.ref, None, args.latent_dim, 'test'))

        fname = ospj(args.result_dir, 'reference.jpg')
        print('Working on {}...'.format(fname))
        utils.translate_using_reference(nets_ema, args, src.x, ref.x, ref.y, fname)

        fname = ospj(args.result_dir, 'video_ref.mp4')
        print('Working on {}...'.format(fname))
        utils.video_ref(nets_ema, args, src.x, ref.x, ref.y, fname)

    @torch.no_grad()
    def projector(self, loaders):
        args = self.args
        nets_ema = self.nets_ema
        self._load_checkpoint(args.resume_iter)
        enc_styles, labels = [], []

        for ref in tqdm(loaders.ref, total=len(loaders.ref)):
            x, y = ref
            sty = nets_ema.style_encoder(x.to(self.device), y.to(self.device))
            enc_styles.append(sty)
            labels.append(y)
        enc_styles = torch.cat(enc_styles)
        labels = torch.cat(labels)

        latents, lat_labs = [], []
        for i in range(1000):
            for j in range(args.num_domains):
                latents.append(nets_ema.mapping_network(torch.randn(1, args.latent_dim).to(self.device), torch.tensor([j]).long().cuda())) #utils.use_latent_project(nets_ema, self.latent_dim, 
                lat_labs.append(torch.tensor([j]).long().cuda())
        latents = torch.cat(latents)
        lat_labs = torch.cat(lat_labs)
        return [[latents, lat_labs]], [[enc_styles, labels]]


    @torch.no_grad()
    def evaluate(self):
        args = self.args
        nets_ema = self.nets_ema
        resume_iter = args.resume_iter
        self._load_checkpoint(args.resume_iter)
        calculate_metrics(nets_ema, args, step=resume_iter, mode='latent')
        calculate_metrics(nets_ema, args, step=resume_iter, mode='reference')


def compute_d_loss(
    nets, 
    args, 
    x_real, 
    y_org, 
    y_trg, 
    z_trg=None, 
    x_ref=None, 
    masks=None
):
    assert (z_trg is None) != (x_ref is None)
    # with real images
    x_real.requires_grad_()

    out_src = nets.discriminator(x_real, y_org)

    s_org = nets.style_encoder(x_real, y_org).detach()

    loss_real = adv_loss(out_src, 1) ## binary cross entropy loss, its coming from real image so comparing with 1
    loss_reg = r1_reg(out_src, x_real)

    # with fake images

    with torch.no_grad():
        if z_trg is not None:

            s_trg = nets.mapping_network(z_trg, y_trg)
        else:  # x_ref is not None
            s_trg = nets.style_encoder(x_ref, y_trg)

        x_fake = nets.generator(x_real, s_trg, masks=masks)


    loss = loss_real + args.lambda_reg * loss_reg
    all_losses = Munch(real=loss_real.item(),
                       reg=loss_reg.item())
        
    out_src = nets.discriminator(x_fake, y_trg)
    loss_fake = adv_loss(out_src, 0)
    loss += loss_fake
    all_losses.fake = loss_fake.item()

    return loss, all_losses

def compute_g_loss(
    nets, 
    args, 
    x_real, 
    y_org, 
    y_trg, 
    z_trgs=None, 
    x_refs=None, 
    masks=None, 
    lpips=None
):
    assert (z_trgs is None) != (x_refs is None)
    if z_trgs is not None:
        z_trg, z_trg2 = z_trgs
    if x_refs is not None:
        x_ref, x_ref2 = x_refs

    s_org = nets.style_encoder(x_real, y_org)

    if z_trgs is not None:
        s_trg = nets.mapping_network(z_trg, y_trg)
        s_trg2 = nets.mapping_network(z_trg2, y_trg)
        s_neg = nets.mapping_network(z_trg, y_org)
    else:
        s_trg = nets.style_encoder(x_ref, y_trg)
        s_trg2 = nets.style_encoder(x_ref2, y_trg)
        s_neg = s_org

    x_fake = nets.generator(x_real, s_trg, masks=masks)
    out_src = nets.discriminator(x_fake, y_trg)
    loss_adv = adv_loss(out_src, 1)

    # style reconstruction loss
    s_pred = nets.style_encoder(x_fake, y_trg)
    loss_sty = torch.mean(torch.abs(s_pred - s_trg))

    # diversity sensitive loss
    x_fake2 = nets.generator(x_real, s_trg2, masks=masks)
    x_fake2 = x_fake2.detach()
    loss_ds = torch.mean(torch.abs(x_fake - x_fake2))

    # cycle-consistency loss
    masks = nets.fan.get_heatmap(x_fake) if args.w_hpf > 0 else None
    x_rec = nets.generator(x_fake, s_org, masks=masks)
    loss_cyc = torch.mean(torch.abs(x_rec - x_real))

    loss = loss_adv + args.lambda_sty * loss_sty \
        - args.lambda_ds * loss_ds + args.lambda_cyc * loss_cyc

    all_losses = Munch(adv=loss_adv.item(),
                       sty=loss_sty.item(),
                       ds=loss_ds.item(),
                       cyc=loss_cyc.item())

    # kl for compacting space
    if args.init_lambda_kl > 0:
        loss_kl  = torch.mean(torch.pow(s_trg,2)) + torch.mean(torch.pow(s_org,2))
        loss += loss_kl * args.init_lambda_kl
        all_losses.kl = loss_kl.item()

    # triplet margin for adjusting the overlappings between clusters
    if args.lambda_tri > 0:
        if torch.sum(y_trg!=y_org) > 0:
            loss_tri = nn.TripletMarginLoss(margin=args.triplet_margin, p=2)(
                s_trg[y_trg!=y_org], s_trg2[y_trg!=y_org], s_neg[y_trg!=y_org])
            loss += loss_tri * args.lambda_tri
            all_losses.tri = loss_tri.item()

    # perceptual similarity loss
    if args.lambda_lpips > 0:
        loss_lpips = lpips(x_real, x_fake)
        loss += loss_lpips * args.lambda_lpips
        all_losses.lpips = loss_lpips.item()

    return loss, all_losses


def moving_average(model, model_test, beta=0.999):
    for param, param_test in zip(model.parameters(), model_test.parameters()):
        param_test.data = torch.lerp(param.data, param_test.data, beta)


def adv_loss(logits, target):
    assert target in [1, 0]
    targets = torch.full_like(logits, fill_value=target)
    loss = F.binary_cross_entropy_with_logits(logits, targets)
    return loss

def r1_reg(d_out, x_in):
    # zero-centered gradient penalty for real images

    batch_size = x_in.size(0)
    grad_dout = torch.autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]

    grad_dout2 = grad_dout.pow(2)
    assert(grad_dout2.size() == x_in.size())
    reg = 0.5 * grad_dout2.view(batch_size, -1).sum(1).mean(0)
    return reg

