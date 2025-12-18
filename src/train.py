import argparse, os, math, time, json
from dataclasses import asdict
import yaml
import torch
from tqdm import tqdm

from model import GPT, GPTConfig
from data import get_batch
from optim import build_optimizer
from utils.seed import set_seed
from utils.logging import ensure_dir, save_json, MetricsWriter

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str, default='configs/base.yaml')
    p.add_argument('--data_dir', type=str, default='data')
    p.add_argument('--out_dir', type=str, default='runs/exp')
    p.add_argument('--optimizer', type=str, default='adam', choices=['adam','rmsprop','signsgd'])
    p.add_argument('--lr', type=float, default=None)
    p.add_argument('--device', type=str, default='auto', choices=['auto','cpu','cuda'])
    p.add_argument('--steps', type=int, default=None)
    p.add_argument('--eval_interval', type=int, default=None)
    p.add_argument('--eval_iters', type=int, default=None)
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--grad_clip', type=float, default=None)
    p.add_argument('--compile', action='store_true', help='Use torch.compile if available')
    return p.parse_args()

@torch.no_grad()
def estimate_loss(model, data_dir, batch_size, block_size, eval_iters, device):
    model.eval()
    losses = {}
    for split in ['train','val']:
        vals = []
        for _ in range(eval_iters):
            x, y = get_batch(split, data_dir, batch_size, block_size, device)
            _, loss = model(x, y)
            vals.append(loss.item())
        losses[split] = sum(vals) / len(vals)
    model.train()
    return losses

def main():
    args = parse_args()
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    # override cfg by args if provided
    if args.steps is not None: cfg['steps'] = args.steps
    if args.eval_interval is not None: cfg['eval_interval'] = args.eval_interval
    if args.eval_iters is not None: cfg['eval_iters'] = args.eval_iters
    if args.seed is not None: cfg['seed'] = args.seed
    if args.grad_clip is not None: cfg['grad_clip'] = args.grad_clip

    lr = args.lr if args.lr is not None else cfg.get('lr', 3e-4)
    cfg['optimizer'] = args.optimizer
    cfg['lr'] = lr

    set_seed(int(cfg['seed']))

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif args.device == 'cuda':
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    ensure_dir(args.out_dir)
    save_json(os.path.join(args.out_dir, 'config.json'), cfg)

    mcfg = GPTConfig(
        vocab_size=int(cfg['vocab_size']),
        block_size=int(cfg['block_size']),
        n_layer=int(cfg['n_layer']),
        n_head=int(cfg['n_head']),
        n_embd=int(cfg['n_embd']),
        dropout=float(cfg['dropout']),
    )
    model = GPT(mcfg).to(device)

    if args.compile and hasattr(torch, 'compile'):
        model = torch.compile(model)

    optim = build_optimizer(
        args.optimizer,
        model.parameters(),
        lr=lr,
        weight_decay=float(cfg.get('weight_decay', 0.0)),
        beta1=float(cfg.get('beta1', 0.9)),
        beta2=float(cfg.get('beta2', 0.95)),
        eps=float(cfg.get('eps', 1e-8)),
    )

    metrics = MetricsWriter(os.path.join(args.out_dir, 'metrics.csv'))

    # training loop
    t0 = time.time()
    for step in tqdm(range(1, int(cfg['steps']) + 1), desc='train'):
        x, y = get_batch('train', args.data_dir, int(cfg['batch_size']), int(cfg['block_size']), device)
        _, loss = model(x, y)
        loss.backward()

        # grad clip
        gc = float(cfg.get('grad_clip', 0.0))
        if gc and gc > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gc)

        optim.step()
        optim.zero_grad()

        if step % int(cfg['eval_interval']) == 0 or step == 1 or step == int(cfg['steps']):
            losses = estimate_loss(
                model, args.data_dir, int(cfg['batch_size']), int(cfg['block_size']),
                int(cfg['eval_iters']), device
            )
            val_ppl = math.exp(losses['val'])
            metrics.write(step=step, train_loss=losses['train'], val_loss=losses['val'], val_ppl=val_ppl, lr=lr)

    metrics.close()
    final = estimate_loss(model, args.data_dir, int(cfg['batch_size']), int(cfg['block_size']), int(cfg['eval_iters']), device)
    final_ppl = math.exp(final['val'])
    save_json(os.path.join(args.out_dir, 'final.json'), {'val_loss': final['val'], 'val_ppl': final_ppl, 'seconds': time.time()-t0})

if __name__ == '__main__':
    main()
