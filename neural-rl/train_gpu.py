"""GPU fitted-Q trainer: an ensemble of small MLPs, double-DQN targets, all members batched."""
import math
import time

FEATURES = 18
ACTIONS = 4


def parse_key(key):
    """'<7 tabular bits>|<FEATURES floats>' -> (scan_allowed, [floats])."""
    if not isinstance(key, str) or key.count('|') != 1:
        raise ValueError(f'Invalid neural state: {key!r}')
    tab, floats = key.split('|')
    bits = tab.split(':')
    if len(bits) != 7 or bits[0] not in ('0', '1', '2') or any(b not in ('0', '1') for b in bits[1:]):
        raise ValueError(f'Invalid tabular part: {key!r}')
    values = [float(v) for v in floats.split(',')]
    if len(values) != FEATURES or not all(math.isfinite(v) for v in values):
        raise ValueError(f'Invalid feature vector: {key!r}')
    return bits[-1] == '1', values


def validate_transitions(rows):
    for row in rows:
        if len(row) != 5:
            raise ValueError('Malformed transition')
        key, action, reward, discount, nxt = row
        scan_ok, _ = parse_key(key)
        if type(action) is not int or not 0 <= action < (4 if scan_ok else 3):
            raise ValueError('Invalid transition action')
        if not math.isfinite(reward) or not math.isfinite(discount) or not 0 <= discount <= 1:
            raise ValueError('Invalid reward/discount')
        if nxt is not None:
            parse_key(nxt)
        elif discount != 0:
            raise ValueError('Terminal transition must have zero discount')


def validate_net(net):
    if set(net) != {'trained', 'features', 'hidden', 'members'} or net['features'] != FEATURES:
        raise ValueError('Invalid network description')
    if bool(net['members']) != net['trained']:
        raise ValueError('A trained network needs members; an untrained one has none')
    sizes = [FEATURES, net['hidden'], net['hidden'], ACTIONS]
    for member in net['members']:
        if set(member) != {'w', 'b'} or len(member['w']) != 3 or len(member['b']) != 3:
            raise ValueError('Invalid ensemble member')
        for depth in range(3):
            w, b = member['w'][depth], member['b'][depth]
            if len(w) != sizes[depth + 1] or len(b) != sizes[depth + 1] or any(len(r) != sizes[depth] for r in w):
                raise ValueError('Layer shape mismatch')
            if not all(math.isfinite(v) for r in w for v in r) or not all(math.isfinite(v) for v in b):
                raise ValueError('Nonfinite weight')
    return net


def train(rows, parent, *, hidden, members, steps, batch, lr, target_sync, seed, device='auto'):
    """Fit Q on every collected transition. Continues from `parent` when it is already trained."""
    import torch
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    name = torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'
    print(f'trainer device: {device} ({name}); transitions: {len(rows)}', flush=True)
    if device != 'cuda':
        print('WARNING: CUDA not available - training on CPU', flush=True)
    torch.manual_seed(seed)
    dev = torch.device(device)

    parsed = [parse_key(r[0]) for r in rows]
    nxt = [parse_key(r[4]) if r[4] is not None else (False, [0.] * FEATURES) for r in rows]
    S = torch.tensor([p[1] for p in parsed], dtype=torch.float32, device=dev)
    S2 = torch.tensor([p[1] for p in nxt], dtype=torch.float32, device=dev)
    A = torch.tensor([r[1] for r in rows], dtype=torch.long, device=dev)
    R = torch.tensor([r[2] for r in rows], dtype=torch.float32, device=dev)
    D = torch.tensor([r[3] for r in rows], dtype=torch.float32, device=dev)
    illegal = torch.zeros(len(rows), ACTIONS, device=dev)
    illegal[:, 3] = torch.tensor([0. if p[0] else -1e9 for p in nxt], device=dev)

    sizes = [FEATURES, hidden, hidden, ACTIONS]
    params = []
    for depth in range(3):
        fan_in = sizes[depth]
        w = torch.empty(members, sizes[depth], sizes[depth + 1], device=dev).uniform_(-1, 1) / math.sqrt(fan_in)
        b = torch.zeros(members, 1, sizes[depth + 1], device=dev)
        if parent['trained']:
            if parent['hidden'] != hidden or len(parent['members']) != members:
                raise ValueError('Parent network shape differs from this run configuration')
            w = torch.tensor([m['w'][depth] for m in parent['members']], device=dev).transpose(1, 2).contiguous()
            b = torch.tensor([m['b'][depth] for m in parent['members']], device=dev).unsqueeze(1)
        params += [w.requires_grad_(), b.requires_grad_()]

    def forward(p, x):  # x: [members, batch, features]
        x = torch.relu(torch.baddbmm(p[1], x, p[0]))
        x = torch.relu(torch.baddbmm(p[3], x, p[2]))
        return torch.baddbmm(p[5], x, p[4])

    target = [p.detach().clone() for p in params]
    optimizer = torch.optim.Adam(params, lr=lr)
    started, running = time.perf_counter(), 0.
    for step in range(1, steps + 1):
        # Every member draws its own minibatch, so members differ by data as well as by init.
        index = torch.randint(len(rows), (members, batch), device=dev)
        with torch.no_grad():
            choice = (forward(params, S2[index]) + illegal[index]).argmax(-1, keepdim=True)
            future = forward(target, S2[index]).gather(-1, choice).squeeze(-1)
            goal = R[index] + D[index] * future
        q = forward(params, S[index]).gather(-1, A[index].unsqueeze(-1)).squeeze(-1)
        loss = torch.nn.functional.smooth_l1_loss(q, goal)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        running = loss.item() if step == 1 else .99 * running + .01 * loss.item()
        if step % target_sync == 0:
            target = [p.detach().clone() for p in params]
        if step % max(1, steps // 10) == 0:
            print(f'  step {step}/{steps} loss {running:.5f} ({time.perf_counter() - started:.0f}s)', flush=True)

    with torch.no_grad():
        mean_q = forward(params, S.unsqueeze(0).expand(members, -1, -1)).mean(dim=(0, 1)).tolist()
    cpu = [p.detach().cpu() for p in params]
    result = []
    for k in range(members):
        result.append(dict(w=[[[round(v, 6) for v in row] for row in cpu[2 * d][k].t().tolist()] for d in range(3)],
                           b=[[round(v, 6) for v in cpu[2 * d + 1][k, 0].tolist()] for d in range(3)]))
    net = validate_net(dict(trained=True, features=FEATURES, hidden=hidden, members=result))
    stats = dict(device=name, steps=steps, batch=batch, final_loss=running, mean_q=mean_q,
                 seconds=time.perf_counter() - started, transitions=len(rows))
    return net, stats
