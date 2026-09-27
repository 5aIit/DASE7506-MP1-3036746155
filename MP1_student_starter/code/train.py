"""Default recipe: 1,200 steps x 32 sequences x 256 targets = 9,830,400 tokens."""
import argparse
from contextlib import nullcontext
import json
import math
from pathlib import Path
import time
import torch
from torch.nn import functional as F
from common import PROTOCOL, ROOT, autocast, device_metrics, load_data, make_model, setup, sha
from evaluate import score


def averaged_weights(model, averaged_parameters):
    """Temporarily load averaged parameters and restore the live training weights."""
    if averaged_parameters is None:
        return nullcontext()

    class _AveragedWeights:
        def __enter__(self):
            self.original = {}
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    self.original[name] = parameter.detach().clone()
                    parameter.copy_(averaged_parameters[name])
            return model

        def __exit__(self, exc_type, exc_value, traceback):
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    parameter.copy_(self.original[name])
            return False

    return _AveragedWeights()


def main():
    total_started = time.perf_counter()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--implementation', default='student')
    p.add_argument('--config', type=Path, default=ROOT/'configs/baseline.json')
    p.add_argument('--run-dir', type=Path, default=ROOT/'runs/baseline-s17')
    p.add_argument('--init-checkpoint', type=Path,
                   help='Optional student checkpoint to continue training from.')
    p.add_argument('--device', default='cpu')
    p.add_argument('--precision', choices=['auto','fp32','bf16'], default='auto')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--seed', type=int, default=17)
    p.add_argument('--steps', type=int, default=1200)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--learning-rate', type=float, default=.001,
                   help='Peak learning rate before the built-in warmup and cosine decay.')
    p.add_argument('--weight-decay', type=float, default=.1,
                   help='AdamW decoupled weight decay.')
    p.add_argument('--ema-decay', type=float, default=0.,
                   help='Optional exponential moving average of weights; 0 disables EMA.')
    p.add_argument('--eval-every', type=int, default=0,
                   help='Optional validation-curve interval; 0 evaluates only after training.')
    args = p.parse_args()
    if args.steps < 1 or args.batch_size < 1:
        p.error('Batch size and step count must be positive.')
    if args.learning_rate <= 0. or args.weight_decay < 0.:
        p.error('Learning rate must be positive and weight decay must be nonnegative.')
    if not 0. <= args.ema_decay < 1.:
        p.error('--ema-decay must be in [0, 1).')
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        p.error('Run directory already contains results. Use a new --run-dir.')
    device, precision = setup(args.device, args.precision, args.threads)
    torch.manual_seed(args.seed)
    prepared = time.perf_counter()
    data = load_data()
    config = json.loads(args.config.read_text())
    model, implementation_sha = make_model(args.implementation, config, device)
    parent_checkpoint_sha256 = None
    parent_train_tokens = 0
    if args.init_checkpoint is not None:
        parent = torch.load(args.init_checkpoint, map_location='cpu', weights_only=True)
        if parent.get('protocol') != PROTOCOL:
            p.error('The initialization checkpoint uses a different protocol.')
        if parent.get('implementation') != args.implementation or parent.get('config') != config:
            p.error('The initialization checkpoint must use the selected implementation and config.')
        model.load_state_dict(parent['model'])
        parent_checkpoint_sha256 = sha(args.init_checkpoint)
        parent_train_tokens = parent.get('cumulative_train_tokens', parent.get('train_tokens', 0))
    ema_start_step = min(100, args.steps)
    ema_parameters = None
    args.run_dir.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    tokens = data['train'][0].to(device)
    rng = torch.Generator().manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    preparation_seconds = time.perf_counter()-prepared
    started = time.perf_counter()
    history = []
    validation_history = []
    intermediate_validation_seconds = 0.
    best_validation_bpb = float('inf')
    best_validation_step = None
    best_validation_state = None
    for step in range(args.steps):
        starts = torch.randint(len(tokens)-257, (args.batch_size,), generator=rng).to(device)
        batch = tokens[starts[:,None]+torch.arange(257,device=device)]
        learning_rate = args.learning_rate * min(1.,(step+1)/100) * (.1+.9*.5*(1+math.cos(math.pi*step/args.steps)))
        for group in optimizer.param_groups:
            group['lr'] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        with autocast(device, precision):
            loss = F.cross_entropy(model(batch[:,:-1]).flatten(0,1).float(),batch[:,1:].flatten())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()
        if args.ema_decay > 0.:
            with torch.no_grad():
                if step + 1 == ema_start_step:
                    ema_parameters = {
                        name: parameter.detach().clone()
                        for name, parameter in model.named_parameters()
                    }
                elif step + 1 > ema_start_step:
                    for name, parameter in model.named_parameters():
                        ema_parameters[name].mul_(args.ema_decay).add_(
                            parameter.detach(), alpha=1. - args.ema_decay
                        )
        if (step+1)%100 == 0 or step+1 == args.steps:
            row = {'step':step+1,'loss':loss.item(),'seconds':time.perf_counter()-started-intermediate_validation_seconds}
            history.append(row)
            print(json.dumps(row),flush=True)
        if args.eval_every > 0 and (step+1)%args.eval_every == 0:
            with averaged_weights(model, ema_parameters):
                intermediate = score(model,*data['validation'],device,'fp32')
                if intermediate['bpb'] < best_validation_bpb:
                    best_validation_bpb = intermediate['bpb']
                    best_validation_step = step + 1
                    best_validation_state = {
                        name: value.detach().cpu().clone()
                        for name, value in model.state_dict().items()
                    }
            intermediate.pop('window_nll_nats')
            intermediate_validation_seconds += intermediate['seconds']
            validation_history.append({'step':step+1,**intermediate})
            print(json.dumps({'validation':validation_history[-1]}),flush=True)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    train_seconds = time.perf_counter()-started-intermediate_validation_seconds
    if best_validation_state is not None:
        model.load_state_dict(best_validation_state)
        validation = score(model,*data['validation'],device,'fp32')
    else:
        with averaged_weights(model, ema_parameters):
            validation = score(model,*data['validation'],device,'fp32')
        if ema_parameters is not None:
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    parameter.copy_(ema_parameters[name])
    validation.pop('window_nll_nats')
    selected_steps = best_validation_step if best_validation_step is not None else args.steps
    selected_train_tokens = selected_steps * args.batch_size * 256
    training_run_tokens = args.steps * args.batch_size * 256
    checkpoint = args.run_dir/'checkpoint.pt'
    torch.save({'protocol':PROTOCOL,'implementation':args.implementation,'config':config,
                'model':model.cpu().state_dict(),'seed':args.seed,
                'train_tokens':selected_train_tokens,
                'training_run_tokens':training_run_tokens,
                'cumulative_train_tokens':parent_train_tokens+selected_train_tokens,
                'parent_train_tokens':parent_train_tokens,
                'batch_size':args.batch_size,
                'parent_checkpoint_sha256':parent_checkpoint_sha256,
                'learning_rate':args.learning_rate,'weight_decay':args.weight_decay,
                'ema_decay':args.ema_decay if args.ema_decay > 0. else None,
                'ema_start_step':ema_start_step if args.ema_decay > 0. else None,
                'best_validation_step':best_validation_step},checkpoint)
    result = {'protocol':PROTOCOL,'implementation':args.implementation,'config':config,'seed':args.seed,
              'parameters':sum(p.numel() for p in model.parameters()),'precision':precision,
              'train_tokens':selected_train_tokens,'training_run_tokens':training_run_tokens,
              'preparation_seconds':preparation_seconds,
              'cumulative_train_tokens':parent_train_tokens+selected_train_tokens,
              'parent_checkpoint_sha256':parent_checkpoint_sha256,
              'learning_rate':args.learning_rate,'weight_decay':args.weight_decay,
              'ema_decay':args.ema_decay if args.ema_decay > 0. else None,
              'ema_start_step':ema_start_step if args.ema_decay > 0. else None,
              'train_seconds':train_seconds,'validation':validation,'history':history,
              'validation_history':validation_history,'best_validation_step':best_validation_step,
              'intermediate_validation_seconds':intermediate_validation_seconds,
              'process_seconds':time.perf_counter()-total_started,
              'torch_version':str(torch.__version__),'threads':args.threads,
              'checkpoint_sha256':sha(checkpoint),'implementation_sha256':implementation_sha,
              **device_metrics(device)}
    (args.run_dir/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result|{'history':[]},indent=2),flush=True)


if __name__ == '__main__':
    main()
