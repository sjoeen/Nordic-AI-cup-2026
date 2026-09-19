"""Neural replacement for the tabular search-pace learner; public observations only.

Same protected modes, options, rewards and credit rules as ForagingRLPolicy. Only the value
function changes: a small ensemble of MLPs over continuous features replaces the 7-bit Q table.
Inference is pure Python so the exported submission needs neither torch nor numpy. Games never
learn online; they only record transitions for the GPU trainer.
"""

NEURAL_FEATURES = 18


class NeuralRLPolicy(ForagingRLPolicy):
    def __init__(self, preset='appetite', **kwargs):
        super().__init__(preset, **kwargs)
        self.rl_net = RL_MODEL['net']
        self.rl_population = 0

    def decide_all(self, states, sim_time=None):
        self.rl_population = len(states)
        return super().decide_all(states, sim_time)

    def rl_state(self, view):
        # "<tabular key>|<features>": the tabular bits keep the scan-availability flag that
        # decides which actions are legal; the floats are what the network sees.
        tab = super().rl_state(view)
        s = view['state']
        aid = s['agent_id']
        clip = lambda x, hi=1.: max(0., min(hi, x))
        vision = max(1., s['vision_range'])
        trees = [o['distance'] for o in s['observations'] if o['type'] == 'Tree']
        others = [o['distance'] for o in s['observations'] if o['type'] == 'Agent']
        walk = min(s['speed'], s['sprint_speed'])
        features = (
            clip(s['energy'] / 200., 2.), clip(s['energy'] / max(1., s['max_energy']), 2.),
            clip(s['age'] / 100., 2.), clip(walk / 20., 2.), clip(vision / 200., 2.),
            float(s['biome'] == 'river'), float(s['biome'] == 'swamp'), float(s['biome'] == 'desert'),
            clip(min(trees) / vision) if trees else 1., clip(len(trees) / 10.),
            clip(min(others) / vision) if others else 1., clip(len(others) / 10.),
            clip((self.clock - self.rl_foodless_since.get(aid, self.clock)) / 30.),
            clip((self.clock - self.rl_last_scan.get(aid, -1e9)) / 12.),
            float(tab.endswith(':1')), clip(self.clock / 3000., 2.),
            clip(self.rl_population / 60., 2.), clip(self.rl_meals[aid] / 5.))
        return tab + '|' + ','.join(format(x, '.4f') for x in features)

    def rl_allowed(self, key):
        return [0, 1, 2, 3] if key.split('|')[0].endswith(':1') else [0, 1, 2]

    def rl_members(self, key):
        """Q values of every ensemble member: one list of four floats per member."""
        x0 = [float(v) for v in key.split('|')[1].split(',')]
        result = []
        for member in self.rl_net['members']:
            x, last = x0, len(member['w']) - 1
            for depth, (weights, biases) in enumerate(zip(member['w'], member['b'])):
                x = [b + sum(w * v for w, v in zip(row, x)) for row, b in zip(weights, biases)]
                if depth < last:
                    x = [v if v > 0. else 0. for v in x]
            result.append(x)
        return result

    def rl_choose(self, key):
        allowed = self.rl_allowed(key)
        if RL_SETTINGS['training']:
            if not self.rl_net['trained'] or self.rl_rng.random() < RL_SETTINGS['epsilon']:
                return self.rl_rng.choice(allowed)
            members = self.rl_members(key)
            return max(allowed, key=lambda i: (sum(m[i] for m in members), -i))
        if not self.rl_net['trained']:
            self.rl_stats['fallback_untrained'] += 1
            return 0
        # Ensemble disagreement stands in for the tabular visit-count guard: leave the
        # baseline only when the advantage survives an uncertainty penalty.
        members = self.rl_members(key)
        count = len(members)
        best, best_score = 0, 0.
        for i in allowed[1:]:
            gains = [m[i] - m[0] for m in members]
            mean = sum(gains) / count
            spread = (sum((g - mean) ** 2 for g in gains) / count) ** .5
            score = mean - RL_SETTINGS['neural_uncertainty'] * spread
            if score >= RL_SETTINGS['advantage_margin'] and score > best_score:
                best, best_score = i, score
        if best == 0:
            self.rl_stats['fallback_no_supported_gain'] += 1
        return best

    def rl_close(self, aid, next_key=None, death=False):
        p = self.rl_pending.pop(aid, None)
        if p is None:
            return
        if death:
            p['reward'] -= p['discount'] * RL_SETTINGS['death_penalty']
            self.rl_stats['credited_deaths'] += 1
        key, choice = p['state'], p['choice']
        discount = p['discount'] if next_key is not None else 0.
        self.rl_stats['closed_options'] += 1
        self.rl_stats['return_sum_' + RL_ACTION_NAMES[choice]] += p['reward']
        self.rl_stats['return_count_' + RL_ACTION_NAMES[choice]] += 1
        if RL_SETTINGS['training']:
            self.rl_transitions.append([key, choice, p['reward'], discount, next_key])
            self.rl_stats['q_updates'] += 1  # recorded transitions; the network itself never updates in-game
