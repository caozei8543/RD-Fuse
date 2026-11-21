"""Stage-wise training utility for RD-Fuse research code."""
from __future__ import annotations
import argparse
import json
import random
import math
from pathlib import Path
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from rd_fuse import RDFuse, LatencyProfile
from rd_fuse.data import PairedManifestDataset, paired_collate, target_to_device
from rd_fuse.training import set_trainable, forced_step, utility_step, regret_step


def run():
    a = argparse.ArgumentParser()
    a.add_argument('--config', default='configs/smoke.yaml')
    a.add_argument('--manifest', required=True)
    a.add_argument('--phase', choices=('forced','utility','regret'), required=True)
    a.add_argument('--resume', default=None)
    a.add_argument('--out', default='artifacts/checkpoints/reference_latest.pt')
    a.add_argument('--epochs', type=int, default=None)
    a.add_argument('--max-steps', type=int, default=None)
    a.add_argument('--device', default='cpu')
    a.add_argument('--num-workers', type=int, default=0)
    a.add_argument('--batch-size', type=int, default=None, help='forced: >=2 gives global InfoNCE negatives; utility/regret require 1')
    a.add_argument('--accum-steps', type=int, default=None)
    a.add_argument('--profile', default=None)
    a.add_argument('--seed', type=int, default=42)
    args = a.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    mcfg,t = cfg['model'],cfg['training']
    stage_index = ('forced','utility','regret').index(args.phase)
    args.epochs = args.epochs if args.epochs is not None else t.get('stage_epochs', [1,1,1])[stage_index]
    args.batch_size = args.batch_size if args.batch_size is not None else t.get('physical_batch_size', {}).get(args.phase, 1)
    args.accum_steps = args.accum_steps if args.accum_steps is not None else t.get('accumulation_steps', {}).get(args.phase, 1)
    if args.epochs < 1: a.error('--epochs must be positive')
    if args.phase != 'forced' and not args.resume:
        a.error('utility/regret require --resume with forced/utility checkpoint')
    if args.phase != 'forced' and args.batch_size != 1:
        a.error('utility/regret currently require --batch-size 1')
    if args.batch_size < 1:
        a.error('--batch-size must be >=1')
    if args.accum_steps < 1:
        a.error('--accum-steps must be >=1')
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    m = RDFuse(num_classes=mcfg['num_classes'],backbone=mcfg['backbone'],
               pretrained=mcfg.get('pretrained',False),semantic_size=mcfg['semantic_size'],grid=mcfg['grid']).to(device)
    if args.resume:
        ck = torch.load(args.resume, map_location='cpu', weights_only=True)
        m.load_state_dict(ck['model'], strict=True)
        print('Loaded checkpoint:',args.resume,'phase:',ck.get('phase'))
    profile_path = args.profile or cfg.get('profile')
    if args.phase == 'regret' and not profile_path:
        a.error('--profile is required for regret training (real target-device measurements preferred)')
    profile = LatencyProfile(profile_path) if profile_path else None
    if profile and not profile.is_verified:
        print('Using a profile without bundled measurement evidence; see profile provenance.')
    dataset = PairedManifestDataset(args.manifest, mcfg['image_size'], flip_probability=t.get('horizontal_flip_probability',0.), seed=args.seed)
    run_scope = 'synthetic_demo' if 'synthetic' in str(args.manifest).lower() else 'user_dataset'
    # Configuration effective batch size may be achieved with accumulation.
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=paired_collate, num_workers=args.num_workers)
    set_trainable(m, args.phase)
    opt = torch.optim.AdamW((p for p in m.parameters() if p.requires_grad),lr=t['learning_rate'],weight_decay=t['weight_decay'], betas=tuple(t.get('adam_betas',[.9,.999])), eps=t.get('adam_eps',1e-8))
    log_path = Path(args.out).with_suffix('.jsonl')
    log_path.parent.mkdir(parents=True, exist_ok=True)
    total_steps = 0
    target_cache = {}
    refresh = t.get('refresh_every_epochs',5)
    budget_rng = random.Random(args.seed + 991)
    ratios = t.get('budget_ratios',[t.get('budget_ratio',.7)])
    if profile: ratios = [r for r in ratios if min(profile.cost(m) for m in ('L','R','T','C0')) <= profile.budget(r)]
    if args.phase == 'regret' and not ratios: a.error('No feasible budget ratios')
    with log_path.open('w', encoding='utf-8') as logfile:
        for epoch in range(args.epochs):
            # Reference policy: stage-local cosine, warmup only for initial forced stage.
            warm = min(t.get('warmup_epochs',0), args.epochs) if args.phase == 'forced' else 0
            base_lr = t['learning_rate']; min_lr = t.get('min_learning_rate',base_lr)
            if epoch < warm:
                lr = base_lr * (epoch+1) / warm
            else:
                progress = (epoch-warm)/max(1,args.epochs-warm-1)
                lr = min_lr + .5*(base_lr-min_lr)*(1+math.cos(math.pi*progress))
            for group in opt.param_groups: group['lr'] = lr
            if args.phase in ('utility','regret'):
                if epoch % refresh == 0: target_cache.clear()
                dataset.set_epoch(epoch//refresh)
            else: dataset.set_epoch(epoch)
            temperature = t.get('regret_temperature_start',t.get('regret_temperature',.2))
            end_temp = t.get('regret_temperature_end',temperature)
            temperature += (end_temp-temperature)*epoch/max(1,args.epochs-1)
            opt.zero_grad(set_to_none=True)
            steps_in_epoch = 0
            for i,(r,th,targets) in enumerate(loader):
                if args.max_steps is not None and total_steps >= args.max_steps:
                    break
                r,th = r.to(device),th.to(device)
                targets=target_to_device(targets,device)
                if args.phase == 'forced':
                    loss, log = forced_step(m,r,th,targets,alpha=t['alpha'],beta=t['beta'],
                                            align_global=t['align_global'],align_local=t['align_local'],
                                            margin=t['reliability_margin'], info_temp=t['infonce_temperature'],
                                            hard_neg=t['hard_neg'],degradation_probability=t['degraded_probability'])
                elif args.phase == 'utility':
                    loss,log = utility_step(m,r,th,targets,sampled_windows=t['sampled_windows'],cache=target_cache)
                else:
                    ratio=budget_rng.choice(ratios)
                    budget=profile.budget(ratio)
                    reg,log=regret_step(m,r,th,targets,profile,budget,
                                         lam=t['lam_ms_inverse'], temperature=temperature, max_candidates=t.get('regret_max_candidates',6))
                    util,_=utility_step(m,r,th,targets,sampled_windows=t['sampled_windows'],cache=target_cache)
                    loss=t['gamma']*util+t['delta']*reg
                    log.update(budget_ratio=ratio,budget_ms=budget,temperature=temperature,utility_loss=float(util.detach()))
                if not torch.isfinite(loss).item():
                    raise FloatingPointError(f'Non-finite training loss: {loss}')
                (loss/args.accum_steps).backward()
                steps_in_epoch += 1; total_steps += 1
                if steps_in_epoch % args.accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_((p for p in m.parameters() if p.requires_grad), t.get('grad_clip_norm',2.0))
                    opt.step(); opt.zero_grad(set_to_none=True)
                record={'epoch':epoch,'step':total_steps,'loss':float(loss.detach()),**log,
                        'run_scope':run_scope,'learning_rate':lr,'refresh_block':epoch//refresh}
                logfile.write(json.dumps(record)+'\n'); logfile.flush()
                print(f'[{args.phase}] e{epoch} step{total_steps} loss={float(loss.detach()):.5f}')
            if steps_in_epoch % args.accum_steps:
                correction = args.accum_steps / (steps_in_epoch % args.accum_steps)
                for param in m.parameters():
                    if param.grad is not None: param.grad.mul_(correction)
                torch.nn.utils.clip_grad_norm_((p for p in m.parameters() if p.requires_grad), t.get('grad_clip_norm',2.0))
                opt.step(); opt.zero_grad(set_to_none=True)
            if args.max_steps is not None and total_steps >= args.max_steps:
                break
    out = Path(args.out); out.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'model':m.cpu().state_dict(), 'config':cfg,'phase':args.phase,
                'seed':args.seed,'steps':total_steps,
                'run_scope':run_scope,'physical_batch_size':args.batch_size,'accum_steps':args.accum_steps},out)
    print('Saved checkpoint:',out)


if __name__=='__main__':
    run()
