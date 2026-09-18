"""Tested observation-only controller. No simulator or hidden-world imports."""
import heapq
import json
import math
import random
import statistics
from collections import Counter,defaultdict,deque
from dataclasses import dataclass,field
from typing import List,Dict
from math import atan2,cos,hypot,pi,sin
from pydantic import BaseModel


VARIANTS = ('modes', 'persistent', 'allocated', 'sliding', 'tangent')

PRESETS = {
    'economy': {},
    'ripen': {'ripen_seconds': 12.},
    'patch': {'ripen_seconds': 12., 'patch_wait': 12.},
    'gaze': {'ripen_seconds': 12., 'patch_wait': 12., 'escape': 'gaze'},
    'predict': {'ripen_seconds': 12., 'patch_wait': 12., 'escape': 'predict'},
    'mapped': {'ripen_seconds': 8., 'patch_wait': 8., 'escape': 'predict', 'search_speed': .8},
    'direct': {'escape': 'direct', 'gaze_turn': math.pi, 'escape_radius': 110.},
    'agile': {'escape': 'predict', 'gaze_turn': math.pi, 'search_speed': .7,
              'prediction_horizon': 6, 'prediction_angles': 12},
    'vigilant': {'escape': 'direct', 'gaze_turn': math.pi, 'scan': .65,
                 'threat_memory': .8, 'threat_cooldown': 3., 'rest_radius': 60.,
                 'face_predators': True, 'escape_radius': 120.},
    'roamer': {'escape': 'direct', 'gaze_turn': math.pi, 'boundary_bias': True,
               'search_reorient': 5., 'scan': .35},
    'guard': {'escape': 'direct', 'gaze_turn': math.pi, 'scan': .3,
              'threat_memory': .5, 'threat_cooldown': 1.5, 'rest_radius': 65.,
              'escape_radius': 130., 'sprint_radius': 80., 'emergency_fraction': .22},
}

QUALITY={'forest':1.,'grassland':.7,'swamp':.35,'desert':.1,'river':0.}

def wrap(angle):
    return (angle + pi) % (2 * pi) - pi

def closest_point(edge):
    (ax, ay), (bx, by) = edge
    dx, dy = bx - ax, by - ay
    length2 = dx * dx + dy * dy
    t = max(0.0, min(1.0, -(ax * dx + ay * dy) / length2)) if length2 else 0.0
    return ax + t * dx, ay + t * dy

def rotate(point, angle):
    c, s = math.cos(angle), math.sin(angle)
    return c * point[0] - s * point[1], s * point[0] + c * point[1]

def unit(point, fallback=(1., 0.)):
    length = math.hypot(*point)
    return (point[0] / length, point[1] / length) if length > 1e-9 else fallback

def point(obs):
    return obs['distance'] * math.cos(obs['angle']), obs['distance'] * math.sin(obs['angle'])

def point_segment(p,a,b):
    dx,dy=b[0]-a[0],b[1]-a[1]
    t=max(0.,min(1.,((p[0]-a[0])*dx+(p[1]-a[1])*dy)/max(1e-9,dx*dx+dy*dy)))
    return math.hypot(p[0]-a[0]-t*dx,p[1]-a[1]-t*dy)

def segment_clear(a,b,edges,margin=6.):
    def cross(x,y,z):return (y[0]-x[0])*(z[1]-x[1])-(y[1]-x[1])*(z[0]-x[0])
    for c,d in edges:
        if cross(a,b,c)*cross(a,b,d)<0. and cross(c,d,a)*cross(c,d,b)<0.:return False
        if min(point_segment(a,c,d),point_segment(b,c,d),point_segment(c,a,b),point_segment(d,a,b))<margin:return False
    return True

class ActionRequest(BaseModel):
    """
    Data transfer object to request an action from specific agent
    """
    agent_id: int
    move_distance: float
    move_direction: float # Relative angle in radians
    turn_angle: float # Relative angle in radians
    spawn_agent: bool

class ObservationResponse(BaseModel):
    """
    Data transfer object to receive an observation for specific agent
    """
    agent_id: int
    energy: float
    biome: str
    age: float

    # Agent attributes
    speed: float
    sprint_speed: float
    hearing_radius: float
    vision_angle: float
    vision_range: float
    max_energy: float

    observations: List[Dict]

class StepResponse(BaseModel):
    """
    Data transfer object to receive a step response
    """
    game_status: str
    score: float
    # These fields were omitted by an older testing client.  Defaults keep
    # that request shape valid while newer clients continue to send values.
    sim_time: float = 0.0
    n_agents: int = 0
    agent_status: List[ObservationResponse] = []

@dataclass
class Memory:
    target_id: object = None
    fruit_tracks: list = field(default_factory=list)
    landmarks: list = field(default_factory=list)
    predators: list = field(default_factory=list)
    last_move: tuple = (0., 0.)
    last_turn: float = 0.
    heading: float = 0.
    search_heading: float = 0.
    sequence: int = 0
    last_age: float = -1.
    escape_side: int = 0
    rng: object = None

class SpeciesFields:
    def __init__(self, variant='tangent'):
        stage = VARIANTS.index(variant)
        self.variant = variant
        self.persistent = stage >= 1
        self.allocated = stage >= 2
        self.sliding = stage >= 3
        self.tangent = stage >= 4
        self.memory = {}
        self.parents = {}
        self.metrics = Counter()

    def canonical(self, key):
        self.parents.setdefault(key, key)
        root = key
        while self.parents[root] != root:
            root = self.parents[root]
        while self.parents[key] != key:
            parent = self.parents[key]
            self.parents[key] = root
            key = parent
        return root

    def merge(self, a, b):
        a, b = self.canonical(a), self.canonical(b)
        if a != b:
            self.parents[max(a, b)] = min(a, b)
            self.metrics['cross_agent_track_merges'] += 1

    def observe(self, state):
        aid = state['agent_id']
        mem = self.memory.setdefault(aid, Memory(rng=random.Random(7919 + aid * 104729)))
        observations = state['observations']
        if getattr(self,'config',{}).get('stale_correction',False):
            # Removing an agent while the engine iterates its list can skip
            # the next survivor's sensing/living-cost update. Its action still
            # ran, so unchanged age means the returned bearings are stale.
            if state['age']==mem.last_age and mem.last_age>=0.:
                previous_turn=getattr(mem,'stale_turn',0.)
                delta=rotate(mem.last_move,previous_turn)
                old_shift=getattr(mem,'stale_shift',(0.,0.))
                total_shift=(old_shift[0]+delta[0],old_shift[1]+delta[1])
                total_turn=previous_turn+mem.last_turn
                corrected=[]
                def transform(p):return rotate((p[0]-total_shift[0],p[1]-total_shift[1]),-total_turn)
                for original in observations:
                    obs=dict(original)
                    if obs['type']=='Edge':
                        obs['coords']=tuple(transform(p) for p in obs['coords'])
                    else:
                        p=transform(point(obs))
                        obs['distance']=math.hypot(*p);obs['angle']=math.atan2(p[1],p[0])
                        if 'rel_dir' in obs:
                            heading=original['angle']+math.pi-original['rel_dir']-total_turn
                            obs['rel_dir']=wrap(obs['angle']+math.pi-heading)
                    corrected.append(obs)
                observations=corrected
                state={**state,'observations':observations}
                mem.stale_turn=total_turn;mem.stale_shift=total_shift
                self.metrics['stale_observation_corrections']+=1
            else:
                mem.stale_turn=0.;mem.stale_shift=(0.,0.)
        fruits = sorted((o for o in observations if o['type'] == 'Fruit' and o['distance'] > 6),
                        key=lambda o: (o['distance'], o['angle']))
        predators = sorted((o for o in observations if o['type'] == 'Predator'),
                           key=lambda o: (o['distance'], o['angle']))
        agents = sorted((o for o in observations if o['type'] == 'Agent'),
                        key=lambda o: (o['distance'], o.get('id', -1)))
        edges = sorted({tuple(tuple(p) for p in o['coords']) for o in observations if o['type'] == 'Edge'})
        landmarks = [('Fruit', point(o)) for o in fruits]
        landmarks += [('Tree', point(o)) for o in observations if o['type'] == 'Tree']
        landmarks += [('Corner', p) for p in sorted({p for edge in edges for p in edge})]
        # Estimate actual translation from static landmarks. Commanded motion
        # is only a fallback because the simulator can redirect collisions.
        shift = rotate(mem.last_move, -mem.last_turn)
        rotated_old = [(kind, rotate(p, -mem.last_turn)) for kind, p in mem.landmarks]
        estimates = []
        for kind, old in rotated_old:
            candidates = [(q[0], q[1]) for k, q in landmarks if k == kind]
            if not candidates:
                continue
            expected = (old[0] - shift[0], old[1] - shift[1])
            q = min(candidates, key=lambda p: math.hypot(p[0] - expected[0], p[1] - expected[1]))
            if math.hypot(q[0] - expected[0], q[1] - expected[1]) < 3.:
                estimates.append((old[0] - q[0], old[1] - q[1]))
        if len(estimates) >= 2:
            shift = (statistics.median(p[0] for p in estimates), statistics.median(p[1] for p in estimates))
        elif rotated_old and landmarks:
            # Collision recovery: test rigid translations supported by at
            # least three landmarks of matching types, never hidden poses.
            max_shift = math.hypot(*mem.last_move) + 1.
            proposals = []
            for kind, p in rotated_old[:24]:
                for other_kind, q in landmarks[:24]:
                    delta = (p[0] - q[0], p[1] - q[1])
                    if kind == other_kind and math.hypot(*delta) <= max_shift:
                        proposals.append(delta)
            if proposals:
                best = max(proposals, key=lambda d: sum(math.hypot(d[0]-p[0], d[1]-p[1]) < .75 for p in proposals))
                support = [p for p in proposals if math.hypot(best[0]-p[0], best[1]-p[1]) < .75]
                if len(support) >= 3:
                    shift = (statistics.median(p[0] for p in support), statistics.median(p[1] for p in support))
                    self.metrics['motion_corrections'] += 1

        old_tracks = []
        for key, old in mem.fruit_tracks:
            p = rotate(old, -mem.last_turn)
            old_tracks.append((key, (p[0] - shift[0], p[1] - shift[1])))
        current_points = [point(o) for o in fruits]
        matches = sorted((math.hypot(old[0] - p[0], old[1] - p[1]), i, j)
                         for i, (_, old) in enumerate(old_tracks)
                         for j, p in enumerate(current_points)
                         if math.hypot(old[0] - p[0], old[1] - p[1]) < 2.)
        old_used, new_used, identifiers = set(), set(), {}
        for distance, i, j in matches:
            if i not in old_used and j not in new_used:
                identifiers[j] = old_tracks[i][0]
                old_used.add(i)
                new_used.add(j)
                self.metrics['fruit_track_matches'] += 1
        tracks = []
        for j, (obs, p) in enumerate(zip(fruits, current_points)):
            if j not in identifiers:
                mem.sequence += 1
                identifiers[j] = (aid, mem.sequence)
            tracks.append(dict(key=identifiers[j], p=p, obs=obs))

        # Predators have headings but no IDs/speeds. Estimate displacement
        # between observations when possible, otherwise use observed heading.
        old_pred = []
        for p in mem.predators:
            q = rotate(p, -mem.last_turn)
            old_pred.append((q[0] - shift[0], q[1] - shift[1]))
        pred_data = []
        for obs in predators:
            p = point(obs)
            heading = wrap(obs['angle'] + math.pi - obs.get('rel_dir', 0.))
            velocity = (0., 0.)
            measured = False
            if old_pred:
                previous = min(old_pred, key=lambda q: math.hypot(p[0]-q[0], p[1]-q[1]))
                delta = (p[0]-previous[0], p[1]-previous[1])
                if math.hypot(*delta) <= 18.:
                    velocity, measured = delta, True
            pred_data.append(dict(obs=obs, p=p, heading=heading, velocity=velocity, measured=measured))
        mem.fruit_tracks = [(t['key'], t['p']) for t in tracks]
        mem.landmarks = landmarks
        mem.predators = [point(o) for o in predators]
        mem.heading = wrap(mem.heading + mem.last_turn)
        mem.last_age = state['age']
        threshold = max(state['hearing_radius'], .6 * state['vision_range'])
        escaping = bool(predators and predators[0]['distance'] < threshold)
        return dict(state=state, mem=mem, fruits=tracks, predators=pred_data,
                    agents=agents, edges=edges, escaping=escaping, threshold=threshold, translation=shift)

    def shared_frames(self, views):
        graph = defaultdict(list)
        for aid, view in views.items():
            for obs in view['agents']:
                bid = obs.get('id')
                if bid not in views:
                    continue
                offset = point(obs)
                angle = wrap(obs['angle'] + math.pi - obs['rel_dir'])
                graph[aid].append((bid, offset, angle))
                reverse = rotate((-offset[0], -offset[1]), -angle)
                graph[bid].append((aid, reverse, -angle))
        poses = {}
        for root in sorted(views):
            if root in poses:
                continue
            poses[root] = (root, (0., 0.), 0.)
            queue = deque([root])
            while queue:
                aid = queue.popleft()
                component, origin, orientation = poses[aid]
                for bid, offset, angle in sorted(graph[aid]):
                    if bid in poses:
                        continue
                    rotated = rotate(offset, orientation)
                    poses[bid] = (component, (origin[0]+rotated[0], origin[1]+rotated[1]), wrap(orientation+angle))
                    queue.append(bid)
        self.metrics['component_ticks'] += len({p[0] for p in poses.values()})
        self.metrics['hivemind_ticks'] += 1
        # Spatial hashing only compares fruit in a currently aligned frame.
        cells = defaultdict(list)
        located = []
        for aid, view in sorted(views.items()):
            component, origin, orientation = poses[aid]
            for fruit in view['fruits']:
                p = rotate(fruit['p'], orientation)
                p = (p[0]+origin[0], p[1]+origin[1])
                cx, cy = math.floor(p[0]/2.), math.floor(p[1]/2.)
                candidates = [item for dx in (-1,0,1) for dy in (-1,0,1)
                              for item in cells[(component,cx+dx,cy+dy)]
                              if item[0] != aid and math.hypot(item[2][0]-p[0], item[2][1]-p[1]) < 1.]
                if candidates:
                    match = min(candidates, key=lambda item: math.hypot(item[2][0]-p[0], item[2][1]-p[1]))
                    self.merge(fruit['key'], match[1])
                cells[(component,cx,cy)].append((aid, fruit['key'], p))
                located.append((component, fruit['key'], p))
        # Share currently observed fruit throughout each reconstructed frame,
        # allowing the nearest agent to be assigned even when facing away.
        # No coordinates from the simulation engine enter this calculation.
        pooled = {}
        for component, key, p in located:
            pooled.setdefault((component, self.canonical(key)), p)
        for aid, view in views.items():
            component, origin, orientation = poses[aid]
            view['shared_pose'] = poses[aid]
            own = {self.canonical(f['key']) for f in view['fruits']}
            for (fruit_component, key), p in pooled.items():
                if fruit_component != component or key in own:
                    continue
                local = rotate((p[0]-origin[0], p[1]-origin[1]), -orientation)
                distance = math.hypot(*local)
                if distance <= 6.:
                    continue
                view['fruits'].append(dict(key=key,p=local,shared=True,
                                          obs=dict(type='Fruit',distance=distance,angle=math.atan2(local[1],local[0]))))
                self.metrics['shared_fruit_observations'] += 1

    def allocate(self, views):
        if self.allocated:
            self.shared_frames(views)
        for aid, view in views.items():
            if 'max_fruit_distance' in view:
                view['fruits'] = [f for f in view['fruits'] if f['obs']['distance'] <= view['max_fruit_distance']]
            for fruit in view['fruits']:
                fruit['id'] = self.canonical(fruit['key']) if self.allocated else fruit['key']
            mem = view['mem']
            if mem.target_id is not None and self.allocated:
                mem.target_id = self.canonical(mem.target_id)
            visible = {f['id']: f for f in view['fruits']}
            if view['escaping'] or not view.get('eligible',True) or mem.target_id not in visible or not self.persistent:
                mem.target_id = None
            view['visible'] = visible

        if not self.allocated:
            for view in views.values():
                mem = view['mem']
                if not view['escaping'] and view.get('eligible',True) and mem.target_id is None and view['fruits']:
                    mem.target_id = min(view['fruits'], key=lambda f:f['obs']['distance'])['id']
            return
        # Reserve valid existing targets before distance-based matching.
        proposals = []
        for aid, view in views.items():
            key = view['mem'].target_id
            if key is not None:
                proposals.append((view['visible'][key]['obs']['distance'], aid, key))
        owners, assigned = {}, set()
        for distance, aid, key in sorted(proposals):
            if key not in owners:
                owners[key], assigned = aid, assigned | {aid}
                self.metrics['preserved_assignments'] += 1
            else:
                views[aid]['mem'].target_id = None
                self.metrics['newly_discovered_conflicts'] += 1
        candidates = sorted((fruit['obs']['distance'], aid, fruit['id'])
                            for aid, view in views.items() if not view['escaping'] and view.get('eligible',True) and aid not in assigned
                            for fruit in view['fruits'])
        for distance, aid, key in candidates:
            if aid not in assigned and key not in owners:
                views[aid]['mem'].target_id = key
                owners[key] = aid
                assigned.add(aid)
        assert len(owners) == len(assigned)

    def slide(self, desired, view, distance):
        original = unit(desired)
        result = original
        terrain = {'swamp':.5, 'river':.3, 'desert':.8}.get(view['state']['biome'], 1.)
        travel = distance * terrain
        used = False
        walls = sorted(((math.hypot(*closest_point(e)), e) for e in view['edges']), key=lambda item:item[0])
        for clearance, edge in walls:
            p = closest_point(edge)
            normal = unit((-p[0], -p[1]))
            inward = -(result[0]*normal[0] + result[1]*normal[1])
            if clearance > 7. + max(0., inward)*travel or inward <= 0.:
                continue
            tangent = unit((edge[1][0]-edge[0][0], edge[1][1]-edge[0][1]))
            along = result[0]*tangent[0] + result[1]*tangent[1]
            if abs(along) < .05:
                prior = rotate(view['mem'].last_move, -view['mem'].last_turn)
                along = prior[0]*tangent[0] + prior[1]*tangent[1]
                if abs(along) < .05:
                    along = 1. if view['state']['agent_id'] % 2 == 0 else -1.
            sign = 1. if along >= 0 else -1.
            outward = .2 if clearance < 7. else .04
            result = unit((sign*tangent[0] + outward*normal[0], sign*tangent[1] + outward*normal[1]))
            used = True
        return result, used

    def arc_escape(self, view, distance):
        predator = view['predators'][0]
        away = unit((-predator['p'][0], -predator['p'][1]))
        velocity = predator['velocity']
        heading = (math.atan2(velocity[1], velocity[0])
                   if predator['measured'] and math.hypot(*velocity) > .5 else predator['heading'])
        # An approach line points from predator toward prey. If the observed
        # heading points away, use the line of sight for escape construction.
        if math.cos(heading)*away[0] + math.sin(heading)*away[1] < .3:
            heading = math.atan2(away[1], away[0])
        terrain = {'swamp':.5, 'river':.3, 'desert':.8}.get(view['state']['biome'], 1.)
        candidates = []
        for side in (-1, 1):
            angle = heading + side*math.pi/3.  # 60-degree evasion arc
            candidate = (math.cos(angle), math.sin(angle))
            if candidate[0]*away[0] + candidate[1]*away[1] < .1:
                angle = math.atan2(away[1], away[0]) + side*math.pi/3.
                candidate = (math.cos(angle), math.sin(angle))
            candidate, slid = self.slide(candidate, view, distance)
            clearance = math.inf
            for enemy in view['predators']:
                # When newly observed, only heading is known; compare current
                # separation rather than inventing the enemy's actual speed.
                vx, vy = enemy['velocity'] if enemy['measured'] else (0., 0.)
                for horizon in (1., 2.):
                    ax, ay = candidate[0]*distance*terrain*horizon, candidate[1]*distance*terrain*horizon
                    px, py = enemy['p'][0]+vx*horizon, enemy['p'][1]+vy*horizon
                    clearance = min(clearance, math.hypot(ax-px, ay-py))
            score = clearance + (2. if side == view['mem'].escape_side else 0.)
            candidates.append((score, side, candidate))
        _, side, result = max(candidates, key=lambda c:(c[0], c[1]))
        view['mem'].escape_side = side
        return result

    def steer(self, view):
        state, mem = view['state'], view['mem']
        target = view['visible'].get(mem.target_id)
        fraction = state['energy'] / max(state['max_energy'], 1e-9)
        walk = max(0., min(state['speed'], state['sprint_speed']))
        distance = walk
        if view['escaping']:
            mode = 'ESCAPE'
            self.metrics['escape_ticks'] += 1
            if fraction > .4:
                distance = max(0., state['sprint_speed'])
                self.metrics['sprint_ticks'] += 1
            predator = view['predators'][0]
            desired = unit((-predator['p'][0], -predator['p'][1]))
            if self.tangent:
                desired = self.arc_escape(view, distance)
        elif target is not None:
            mode = 'EAT'
            self.metrics['eat_ticks'] += 1
            # Exactly one fruit contributes attraction; no predator field.
            strength = 40. / max(target['obs']['distance'], 8.)
            tx, ty = unit(target['p'])
            desired = (tx*strength, ty*strength)
            sx = sy = 0.
            for teammate in view['agents']:
                if teammate['distance'] < 25.:
                    weight = .2*(1.-teammate['distance']/25.)
                    sx -= weight*math.cos(teammate['angle'])
                    sy -= weight*math.sin(teammate['angle'])
            # Keep separation mild relative to attraction, even in a crowd.
            cap = min(1., .2*strength/max(math.hypot(sx, sy), 1e-9))
            desired = (desired[0]+cap*sx, desired[1]+cap*sy)
        else:
            mode = 'SEARCH'
            self.metrics['search_ticks'] += 1
            if self.allocated and view['fruits'] and view['agents']:
                nearest = view['agents'][0]
                mem.search_heading = wrap(mem.heading + nearest['angle'] + math.pi)
                self.metrics['displaced_search_ticks'] += 1
            mem.search_heading = wrap(mem.search_heading + mem.rng.uniform(-.025, .025))
            angle = wrap(mem.search_heading-mem.heading)
            desired = (math.cos(angle), math.sin(angle))

        if mode != 'ESCAPE' and state['age'] > 60. and (target is None or target['obs']['distance'] > state['hearing_radius']):
            distance *= max(.5, 60./state['age'])
        if self.sliding:
            desired, slid = self.slide(desired, view, distance)
            self.metrics['wall_slide_ticks'] += int(slid)
        else:
            # Earlier-stage control uses normal repulsion plus the same small
            # tangent used in the hierarchical experiment.
            fx, fy = desired
            radius = max(20., min(40., 2.*walk+5.))
            for edge in view['edges']:
                px, py = closest_point(edge)
                d = max(math.hypot(px, py), .1)
                if d >= radius:
                    continue
                strength = min(12., 4.*(1.-d/radius)**2*5./max(d-5., 1.))
                nx, ny = -px/d, -py/d
                tx, ty = -ny, nx
                if tx*fx + ty*fy < 0:
                    tx, ty = -tx, -ty
                fx += strength*(nx+.5*tx)
                fy += strength*(ny+.5*ty)
            desired = (fx, fy)
        direction = math.atan2(desired[1], desired[0]) if math.hypot(*desired) > 1e-9 else wrap(mem.search_heading-mem.heading)
        if mode == 'EAT' and abs(wrap(direction-target['obs']['angle'])) < .5:
            terrain = {'swamp':.5, 'river':.3, 'desert':.8}.get(state['biome'], 1.)
            distance = min(distance, max(0., target['obs']['distance']-6.)/terrain)
        threshold = max(.8*state['max_energy'], 100.+.2*state['max_energy'])
        spawn = state['energy'] > threshold and not view['predators']
        turn = max(-.4, min(.4, direction))
        terrain = {'swamp':.5, 'river':.3, 'desert':.8}.get(state['biome'], 1.)
        mem.last_move = (distance*terrain*math.cos(direction), distance*terrain*math.sin(direction))
        mem.last_turn = turn
        # Persist the heading actually chosen, including wall deflections.
        # Keeping the pre-wall search heading repeatedly drives into the wall.
        mem.search_heading = wrap(mem.heading+direction)
        return ActionRequest(agent_id=state['agent_id'], move_distance=distance,
                             move_direction=direction, turn_angle=turn, spawn_agent=spawn)

    def decide_all(self, states):
        # A new game can reuse IDs and ages. Clear all cross-game state.
        if any(s['agent_id'] in self.memory and s['age'] < self.memory[s['agent_id']].last_age for s in states):
            self.memory.clear()
            self.parents.clear()
        live = {s['agent_id'] for s in states}
        self.memory = {key: value for key, value in self.memory.items() if key in live}
        views = {s['agent_id']:self.observe(s) for s in sorted(states,key=lambda s:s['agent_id'])}
        self.allocate(views)
        return [self.steer(views[s['agent_id']]) for s in states]

class SurvivalPolicy(SpeciesFields):
    def __init__(self, preset='economy', **overrides):
        super().__init__('tangent')
        self.config = dict(search_speed=.4, scan=.2, birth_energy=220.,
                           population=12, population_floor=3, population_decay=900.,
                           retirement=105., ripen_seconds=0., patch_wait=0.,
                           escape='arc', escape_radius=100., sprint_radius=55.,
                           idle_energy=.87, young_age=65., reserve=80.)
        self.config.update(gaze_turn=1., prediction_horizon=4, prediction_angles=6)
        self.config.update(threat_memory=0.,threat_cooldown=0.,rest_radius=45.,face_predators=False,
                           selective_breeding=0.,weighted_allocation=False,boundary_bias=False,search_reorient=0.,emergency_fraction=.4,
                           fresh_ripen=0.,ripen_age_limit=65.,hard_idle_energy=1e9,birth_floor=135.,fertile_population=False,
                           quality_mode='basic',fitness_population=False)
        self.config.update(PRESETS.get(preset, {}))
        self.config.update(overrides)
        self.preset = preset
        self.clock = 0.
        self.first_seen = {}
        self.birth_times = {}
        self.spawn_ids = set()
        self.retired_ids = set()

    def observe(self, state):
        old_mem=self.memory.get(state['agent_id'])
        old_keys={key for key,p in old_mem.fruit_tracks} if old_mem is not None else set()
        had_previous=old_mem is not None and old_mem.last_age>=0.
        view = super().observe(state)
        mem=view['mem']
        if view['predators']:
            mem.enemy_cache=[dict(p) for p in view['predators']]
            mem.enemy_seen=self.clock
        elif self.config['threat_memory'] and hasattr(mem,'enemy_cache') and self.clock-mem.enemy_seen<self.config['threat_memory']:
            remembered=[]
            for enemy in mem.enemy_cache:
                q=rotate(enemy['p'],-mem.last_turn)
                v=rotate(enemy['velocity'],-mem.last_turn)
                q=(q[0]-view['translation'][0]+v[0]*.7,q[1]-view['translation'][1]+v[1]*.7)
                d=math.hypot(*q)
                remembered.append(dict(p=q,heading=wrap(enemy['heading']-mem.last_turn),velocity=v,measured=True,
                                       remembered=True,obs=dict(type='Predator',distance=d,angle=math.atan2(q[1],q[0]))))
            view['predators']=remembered
            mem.enemy_cache=remembered
        for fruit in view['fruits']:
            key = self.canonical(fruit['key'])
            self.first_seen.setdefault(key, self.clock)
            if had_previous and fruit['key'] not in old_keys and fruit['obs']['distance']+math.hypot(*view['translation'])+3.<state['hearing_radius']:
                # It was inside the previous hearing disk and absent then:
                # unlike a newly discovered distant fruit, this is a birth.
                self.birth_times.setdefault(key,self.clock)
        if self.config['escape'] != 'arc':
            threats = []
            for enemy in view['predators']:
                d = enemy['obs']['distance']
                v = enemy['velocity']
                closing = -(v[0]*enemy['p'][0]+v[1]*enemy['p'][1])/max(d,1.)
                radius = self.config['escape_radius']
                # Still predators may wake without advance notice.
                if enemy['measured'] and math.hypot(*v) < .5:
                    radius = min(radius, self.config['rest_radius'])
                if d < radius or (closing > 2 and d/closing < 7.):
                    threats.append(enemy)
            view['escaping'] = bool(threats)
            if threats:
                view['predators'].sort(key=lambda p: (p not in threats,p['obs']['distance']))
        mem = view['mem']
        if not hasattr(mem, 'patch_since'):
            mem.patch_since = None
        return view

    def merge(self, a, b):
        aa, bb = self.canonical(a), self.canonical(b)
        first = min(self.first_seen.get(aa,self.clock), self.first_seen.get(bb,self.clock))
        births=[self.birth_times[k] for k in (aa,bb) if k in self.birth_times]
        super().merge(a,b)
        self.first_seen[self.canonical(a)] = first
        if births:self.birth_times[self.canonical(a)]=min(births)

    def decide_all(self, states, sim_time=None):
        if any(s['agent_id'] in self.memory and s['age'] < self.memory[s['agent_id']].last_age for s in states):
            self.memory.clear()
            self.parents.clear()
            self.first_seen.clear()
            self.birth_times.clear()
            self.clock = 0.
        self.clock = float(sim_time) if sim_time is not None else self.clock + .1
        live = {s['agent_id'] for s in states}
        self.memory = {k:v for k,v in self.memory.items() if k in live}
        views = {s['agent_id']:self.observe(s) for s in sorted(states,key=lambda s:s['agent_id'])}
        c = self.config
        cap = max(c['population_floor'], round(c['population'] * .5**(self.clock/c['population_decay'])))
        young = sum(s['age'] < c['young_age'] and (not c['fertile_population'] or s['max_energy']>c['birth_floor']+5.) for s in states)
        def quality(s):
            if c['quality_mode']!='basic':
                walk=max(.1,min(s['speed'],s['sprint_speed']))/10.
                hearing=max(.1,s['hearing_radius'])/50.
                vision=max(.1,s['vision_range'])/200.
                cone=max(.01,s['vision_angle'])/(math.pi/3.)
                fertility=min(1.,max(.01,(s['max_energy']-100.)/100.))
                if c['quality_mode']=='endurance':return walk**1.4*hearing**.9*vision**.4*cone**.3*(max(100.,s['max_energy'])/500.)**.65*fertility
                if c['quality_mode']=='sensory':return walk**1.3*hearing**1.2*vision**.5*cone**.4*fertility
                return walk**1.7*hearing**.8*vision**.4*cone**.3*fertility
            return min(s['speed'],s['sprint_speed'])*math.sqrt(max(1.,s['hearing_radius']+.5*s['vision_range']))
        qualities=sorted(quality(s) for s in states)
        median_quality=qualities[len(qualities)//2] if qualities else 1.
        if c['fitness_population'] and qualities:
            young=round(sum(min(1.,quality(s)/max(qualities)) for s in states if s['age']<c['young_age']))
        elite_cutoff=c.get('elite_population',0.)
        if elite_cutoff and qualities:
            young=sum(s['age']<c['young_age'] and quality(s)>=elite_cutoff*max(qualities) for s in states)
        self.retired_ids = {s['agent_id'] for s in states if s['age']>c['retirement'] and young>=max(2,cap//2)}
        if elite_cutoff and qualities and sum(quality(s)>=elite_cutoff*max(qualities) for s in states)>=2:
            self.retired_ids.update(s['agent_id'] for s in states if quality(s)<.8*max(qualities) and s['age']>2.)
        candidates = []
        for aid, view in views.items():
            s = view['state']
            threshold = min(c['birth_energy'], .75*s['max_energy'])
            threshold = max(c['birth_floor'], threshold)
            if c['selective_breeding']:
                threshold=max(135.,threshold+c['selective_breeding']*(1.-quality(s)/median_quality))
            if s['age'] > 60 and young < max(2,cap//2):
                threshold = min(threshold,160.)
            safe = not view['predators'] or min(p['obs']['distance'] for p in view['predators']) > 130.
            safe = safe and self.clock-getattr(view['mem'],'enemy_seen',-10000.)>=c['threat_cooldown']
            if s['energy'] > threshold and safe and aid not in self.retired_ids:
                parent_quality = quality(s) if c['selective_breeding'] or c['quality_mode']!='basic' else min(s['speed'],s['sprint_speed']) + .015*s['hearing_radius'] + .003*s['vision_range']
                candidates.append((parent_quality, s['energy'], -aid, aid))
        self.spawn_ids = {x[-1] for x in sorted(candidates,reverse=True)[:max(0,cap-young)]}
        for aid,view in views.items():
            s=view['state']
            if aid in self.retired_ids or (s['energy']>min(c['hard_idle_energy'],c['idle_energy']*s['max_energy']) and aid not in self.spawn_ids):
                view['resting'] = True
            else:
                view['resting'] = False
            view['eligible'] = not view['resting']
        self.allocate(views)
        return [self.steer(views[s['agent_id']]) for s in states]

    def predictive_escape(self, view, walk, sprint):
        """Sample short trajectories against a bounded-turn pursuit model."""
        terrain = {'swamp':.5,'river':.3,'desert':.8}.get(view['state']['biome'],1.)
        nearest=view['predators'][0]
        away=math.atan2(-nearest['p'][1],-nearest['p'][0])
        previous=rotate(view['mem'].last_move,-view['mem'].last_turn)
        prev_angle=math.atan2(previous[1],previous[0])
        choices=[]
        speeds=[walk]
        if sprint>walk+.01 and view['state']['energy']>.21*view['state']['max_energy']+10:
            speeds.append(sprint)
        for speed in speeds:
            count=self.config['prediction_angles']
            offsets=[(i-count/2)*math.pi/count for i in range(count+1)]
            if count>=12:
                offsets.extend([-2*math.pi/3,2*math.pi/3])
            for offset in offsets:
                direction,slid=self.slide((math.cos(away+offset),math.sin(away+offset)),view,speed)
                vx,vy=direction[0]*speed*terrain,direction[1]*speed*terrain
                minimum=math.inf
                end=math.inf
                for enemy in view['predators'][:4]:
                    px,py=enemy['p']; heading=enemy['heading']
                    ex,ey=enemy['velocity']
                    moving=not enemy['measured'] or math.hypot(ex,ey)>.5
                    if moving:
                        # Observations precede the predator's last move.
                        px+=ex; py+=ey
                    for horizon in range(1,self.config['prediction_horizon']+1):
                        ax,ay=vx*horizon,vy*horizon
                        if moving or horizon>=3:
                            target_angle=math.atan2(ay-py,ax-px)
                            heading=wrap(heading+max(-.3,min(.3,wrap(target_angle-heading)*.5)))
                            pspeed=15.*terrain
                            px+=pspeed*math.cos(heading); py+=pspeed*math.sin(heading)
                        d=math.hypot(ax-px,ay-py)
                        minimum=min(minimum,d)
                    end=min(end,d)
                cost=.05*min(speed,walk)+.5*max(0,speed-walk)
                angle=math.atan2(direction[1],direction[0])
                # Collision avoidance dominates, then useful clearance and cost.
                value=(-1000.*max(0,22.-minimum) + min(minimum,65.)
                       +.15*min(end,150.)-4.*cost+1.5*math.cos(angle-prev_angle))
                choices.append((value,speed,direction))
        _,speed,direction=max(choices,key=lambda x:x[0])
        return direction,speed

    def steer(self, view):
        s,mem,c=view['state'],view['mem'],self.config
        target=view['visible'].get(mem.target_id)
        terrain={'swamp':.5,'river':.3,'desert':.8}.get(s['biome'],1.)
        walk=max(0.,min(s['speed'],s['sprint_speed']))
        speed=walk
        turn=None
        mode='SEARCH'
        if view['escaping']:
            mode='ESCAPE'
            if c['escape']=='predict':
                desired,speed=self.predictive_escape(view,walk,s['sprint_speed'])
            else:
                if s['energy']>max(c['emergency_fraction']*s['max_energy'],.2*s['max_energy']+5.) and view['predators'][0]['obs']['distance']<c['sprint_radius']:
                    speed=s['sprint_speed']
                desired=(unit((-view['predators'][0]['p'][0],-view['predators'][0]['p'][1]))
                         if c['escape']=='direct' else self.arc_escape(view,speed))
            if c['escape']!='arc':
                pred=view['predators'][0]
                turn=max(-c['gaze_turn'],min(c['gaze_turn'],pred['obs']['angle']))
            mem.patch_since=None
        elif view.get('resting'):
            mode='REST'
            desired=(1.,0.); speed=0.; turn=c['scan']
        elif target is not None:
            mode='EAT'
            desired=unit(target['p'])
            d=target['obs']['distance']
            seen=self.clock-self.first_seen.get(self.canonical(target['key']),self.clock)
            wait=c['ripen_seconds']
            born=self.birth_times.get(self.canonical(target['key']))
            if c['fresh_ripen'] and born is not None:
                wait=c['fresh_ripen'];seen=self.clock-born
            ripening=(seen<wait and s['energy']>c['reserve']+2*(wait-seen) and s['age']<c['ripen_age_limit'])
            stop=17. if ripening else 6.
            speed=min(walk,max(0.,d-stop)/terrain)
            if ripening and d<19.:
                mode='RIPEN'; turn=c['scan']
            mem.patch_since=None
        else:
            trees=[o for o in s['observations'] if o['type']=='Tree']
            nearby=min(trees,key=lambda o:o['distance']) if trees else None
            if c['patch_wait'] and nearby and nearby['distance']<55. and s['energy']>c['reserve'] and s['age']<65.:
                if mem.patch_since is None:
                    mem.patch_since=self.clock
                if self.clock-mem.patch_since<c['patch_wait']:
                    mode='PATCH'; desired=unit(point(nearby))
                    speed=min(walk,max(0.,nearby['distance']-35.)/terrain)
                    turn=c['scan']
                else:
                    nearby=None
            else:
                mem.patch_since=None
                nearby=None
            if mode=='SEARCH':
                if c['search_reorient'] and self.clock>=getattr(mem,'next_reorient',0.):
                    mem.search_heading=wrap(mem.search_heading+mem.rng.choice((-1,1))*mem.rng.uniform(.6,1.6))
                    mem.next_reorient=self.clock+c['search_reorient']
                if view['fruits'] and view['agents']:
                    a=view['agents'][0]
                    mem.search_heading=wrap(mem.heading+a['angle']+math.pi)
                mem.search_heading=wrap(mem.search_heading+mem.rng.uniform(-.015,.015))
                angle=wrap(mem.search_heading-mem.heading)
                desired=(math.cos(angle),math.sin(angle))
                if c['boundary_bias']:
                    dx,dy=desired
                    for edge in view['edges']:
                        length=math.hypot(edge[1][0]-edge[0][0],edge[1][1]-edge[0][1])
                        p=closest_point(edge);d=math.hypot(*p)
                        if length>800. and d<120.:
                            normal=unit((-p[0],-p[1]))
                            weight=3.*max(0.,1.-d/120.)
                            dx+=weight*normal[0];dy+=weight*normal[1]
                    desired=unit((dx,dy))
                speed=walk*c['search_speed']
                turn=c['scan']
        desired,slid=self.slide(desired,view,speed)
        direction=math.atan2(desired[1],desired[0])
        if turn is None:
            turn=max(-.6,min(.6,direction))
        if c['face_predators'] and view['predators'] and (target is None or target['obs']['distance']<s['hearing_radius']):
            enemy=view['predators'][0]
            if enemy['obs']['distance']<180.:
                turn=max(-c['gaze_turn'],min(c['gaze_turn'],enemy['obs']['angle']))
        self.metrics[mode.lower()+'_ticks']+=1
        mem.last_mode=mode
        self.metrics['sprint_ticks']+=int(speed>walk+.01)
        self.metrics['wall_slide_ticks']+=int(slid)
        mem.last_move=(speed*terrain*math.cos(direction),speed*terrain*math.sin(direction))
        mem.last_turn=turn
        if speed>0:
            mem.search_heading=wrap(mem.heading+direction)
        return ActionRequest(agent_id=s['agent_id'],move_distance=speed,move_direction=direction,
                             turn_angle=turn,spawn_agent=s['agent_id'] in self.spawn_ids)

class AdaptivePolicy(SurvivalPolicy):
    def __init__(self,preset='adaptive',**overrides):
        config=dict(escape='direct',gaze_turn=math.pi,population=6,population_floor=2,
                    emergency_birth=True,emergency_birth_distance=55.,renewal_age=75.,
                    crowd_escape=True,crowd_margin=45.,hunger_feed=True,search_speed=.4,sprint_ceiling=None)
        config.update(overrides)
        super().__init__(preset,**config)

    def decide_all(self,states,sim_time=None):
        actions=super().decide_all(states,sim_time)
        c=self.config
        if c['sprint_ceiling'] is not None:
            byid={s['agent_id']:s for s in states}
            for action in actions:
                s=byid[action.agent_id]
                cap=max(min(s['speed'],s['sprint_speed']),min(c['sprint_ceiling'],s['sprint_speed']))
                if action.move_distance>cap:
                    action.move_distance=cap
                    terrain={'river':.3,'swamp':.5,'desert':.8}.get(s['biome'],1.)
                    self.memory[action.agent_id].last_move=(cap*terrain*math.cos(action.move_direction),cap*terrain*math.sin(action.move_direction))
        if c['emergency_birth']:
            young=sum(s['age']<c['renewal_age'] for s in states)
            byid={s['agent_id']:s for s in states}
            for action in actions:
                s=byid[action.agent_id]
                if action.spawn_agent:continue
                enemies=[o['distance'] for o in s['observations'] if o['type']=='Predator']
                safe=not enemies or min(enemies)>c['emergency_birth_distance']
                cost=.05*min(action.move_distance,s['speed'])+.5*max(0.,action.move_distance-s['speed'])+abs(action.turn_angle)/(2*math.pi)
                if safe and s['energy']>112.+cost and (len(states)<3 or (s['age']>c['renewal_age'] and young<max(2,self.config['population_floor']))):
                    action.spawn_agent=True;young+=1
                    self.metrics['emergency_births']+=1
        return actions

    def steer(self,view):
        s,mem,c=view['state'],view['mem'],self.config
        action=super().steer(view)
        if not view['escaping'] or not c['crowd_escape']:return action
        enemies=view['predators'][:5]
        terrain={'river':.3,'swamp':.5,'desert':.8}.get(s['biome'],1.)
        walk=min(s['speed'],s['sprint_speed'])
        sx=sy=0.
        for p in enemies:
            d=max(10.,p['obs']['distance'])
            sx-=p['p'][0]/d**3;sy-=p['p'][1]/d**3
        away=math.atan2(sy,sx)
        speeds=[walk]
        if s['energy']>.2*s['max_energy']+5. and s['sprint_speed']>walk:
            speeds.extend(((walk+s['sprint_speed'])/2.,s['sprint_speed']))
        angles=[away+i*math.pi/12 for i in range(-6,7)]
        angles.append(action.move_direction)
        food=min(view['fruits'],key=lambda f:f['obs']['distance']) if view['fruits'] else None
        if food and c['hunger_feed']:angles.append(food['obs']['angle'])
        options=[]
        for speed in speeds:
            for angle in angles:
                direction,_=self.slide((math.cos(angle),math.sin(angle)),view,speed)
                vx,vy=direction[0]*speed*terrain,direction[1]*speed*terrain
                minimum=math.inf;ending=math.inf
                for enemy in enemies:
                    ex,ey=enemy['velocity']
                    if not enemy['measured']:
                        ex,ey=15.*terrain*math.cos(enemy['heading']),15.*terrain*math.sin(enemy['heading'])
                    for t in (1.,2.,3.):
                        d=math.hypot(enemy['p'][0]+ex*(t+1)-vx*t,enemy['p'][1]+ey*(t+1)-vy*t)
                        minimum=min(minimum,d)
                    ending=min(ending,d)
                cost=.05*min(speed,s['speed'])+.5*max(0.,speed-s['speed'])
                progress=0.
                if food and c['hunger_feed']:
                    d=food['obs']['distance']
                    progress=(d-math.hypot(food['p'][0]-vx*3,food['p'][1]-vy*3))*min(1.,75./max(50.,s['energy']))
                score=-1000.*max(0.,25.-minimum)-1.5*max(0.,c['crowd_margin']-minimum)+.15*min(ending,150.)-cost*5.+progress*.5
                goal=view.get('escape_goal')
                if goal is not None:
                    score+=view.get('escape_goal_weight',.5)*(math.hypot(*goal)-math.hypot(goal[0]-vx*3,goal[1]-vy*3))
                options.append((score,speed,direction))
        _,speed,direction=max(options,key=lambda x:x[0])
        action.move_distance=speed;action.move_direction=math.atan2(direction[1],direction[0])
        action.turn_angle=enemies[0]['obs']['angle']
        mem.last_move=(speed*terrain*direction[0],speed*terrain*direction[1])
        mem.last_turn=action.turn_angle;mem.search_heading=wrap(mem.heading+action.move_direction)
        mem.last_mode='CROWD_ESCAPE';self.metrics['crowd_escape_ticks']+=1
        return action

class RefugePolicy(AdaptivePolicy):
    def __init__(self,preset='refuge',**overrides):
        config=dict(face_predators=True,blocked_threat_distance=65.,crowd_escape=True,
                    hard_idle_energy=250.,birth_energy=180.)
        config.update(overrides)
        super().__init__(preset,**config)

    def observe(self,state):
        view=super().observe(state)
        mem=view['mem']
        previous=[]
        for edge,seen in getattr(mem,'shelter_edges',[]):
            if self.clock-seen>2.:continue
            transformed=[]
            for p in edge:
                q=rotate(p,-mem.last_turn)
                transformed.append((q[0]-view['translation'][0],q[1]-view['translation'][1]))
            previous.append((tuple(transformed),seen))
        current=[(edge,self.clock) for edge in view['edges']]
        mem.shelter_edges=(previous+current)[-60:]
        threats=[]
        for enemy in view['predators']:
            blocked=[edge for edge,t in mem.shelter_edges if not segment_clear((0.,0.),enemy['p'],[edge],0.)]
            if blocked:
                length=max(min(math.hypot(*q)+math.hypot(q[0]-enemy['p'][0],q[1]-enemy['p'][1]) for q in edge) for edge in blocked)
                if length>self.config['blocked_threat_distance'] and enemy['obs']['distance']>22.:
                    self.metrics['sheltered_threat_ticks']+=1
                    continue
            d=enemy['obs']['distance'];v=enemy['velocity']
            closing=-(v[0]*enemy['p'][0]+v[1]*enemy['p'][1])/max(1.,d)
            radius=self.config['rest_radius'] if enemy['measured'] and math.hypot(*v)<.5 else self.config['escape_radius']
            if d<radius or (closing>2. and d/closing<7.):threats.append(enemy)
        view['escaping']=bool(threats)
        if threats:view['predators'].sort(key=lambda e:(e not in threats,e['obs']['distance']))
        return view

class DispersalPolicy(RefugePolicy):
    def __init__(self,preset='dispersal',**overrides):
        config=dict(dispersal_distance=200.,dispersal_time=6.,stale_correction=True)
        config.update(overrides)
        super().__init__(preset,**config)

    def observe(self,state):
        view=super().observe(state);mem=view['mem']
        if not hasattr(mem,'birth_origin'):
            mem.birth_origin=(0.,0.)
            mem.colonist=state['age']<.5 and state['energy']<100.
            mem.migration_heading=mem.rng.uniform(-math.pi,math.pi)
        q=rotate(mem.birth_origin,-mem.last_turn)
        mem.birth_origin=(q[0]-view['translation'][0],q[1]-view['translation'][1])
        if state['age']>self.config['dispersal_time'] or state['energy']<40. or math.hypot(*mem.birth_origin)>self.config['dispersal_distance']:
            mem.colonist=False
        return view

    def allocate(self,views):
        for view in views.values():
            if view['mem'].colonist:
                view['eligible']=False;view['resting']=False
        super().allocate(views)

    def steer(self,view):
        action=super().steer(view)
        s,mem=view['state'],view['mem']
        if not mem.colonist or view['escaping']:return action
        angle=wrap(mem.migration_heading-mem.heading)
        if view['predators']:
            enemy=view['predators'][0]
            if enemy['obs']['distance']<160. and math.cos(angle-enemy['obs']['angle'])>.2:
                mem.migration_heading=wrap(mem.heading+enemy['obs']['angle']+math.pi)
                angle=wrap(mem.migration_heading-mem.heading)
        speed=min(s['speed'],s['sprint_speed'])
        direction,_=self.slide((math.cos(angle),math.sin(angle)),view,speed)
        action.move_distance=speed;action.move_direction=math.atan2(direction[1],direction[0])
        action.turn_angle=max(-.6,min(.6,action.move_direction))
        terrain={'river':.3,'swamp':.5,'desert':.8}.get(s['biome'],1.)
        mem.last_move=tuple(v*speed*terrain for v in direction);mem.last_turn=action.turn_angle
        mem.search_heading=wrap(mem.heading+action.move_direction);mem.last_mode='COLONIZE'
        self.metrics['colonization_ticks']+=1
        return action

class GenerationPolicy(DispersalPolicy):
    def __init__(self,preset='generation',**overrides):
        config=dict(dispersal_distance=100.,generation_mode='baseline',generation_fraction=.65,
                    generation_lifetime=90.,birth_spacing=0.,maximum_birth_gap=0.,
                    elder_birth_priority=False,young_energy_weight=0.)
        config.update(overrides);super().__init__(preset,**config)
        self.last_normal_birth=-1000.

    def decide_all(self,states,sim_time=None):
        if ((sim_time is not None and sim_time<self.clock) or
                any(s['agent_id'] in self.memory and s['age']<self.memory[s['agent_id']].last_age for s in states)):
            self.last_normal_birth=-1000.
        return super().decide_all(states,sim_time)

    def allocate(self,views):
        c=self.config
        if c['generation_mode']!='baseline' or c['birth_spacing'] or c['maximum_birth_gap'] or c['elder_birth_priority']:
            states=[v['state'] for v in views.values()]
            cap=max(c['population_floor'],round(c['population']*.5**(self.clock/c['population_decay'])))
            young=sum(s['age']<c['young_age'] for s in states)
            budget=max(0,cap-young)
            if c['generation_mode']=='linear':
                effective=sum(max(0.,1.-s['age']/c['generation_lifetime']) for s in states)
                budget=max(0,math.floor(cap*c['generation_fraction']-effective+.5))
            elif c['generation_mode']=='energy':
                effective=sum(min(1.,max(.25,s['energy']/120.)) for s in states if s['age']<c['young_age'])
                budget=max(0,math.floor(cap-effective+.5))
            if c['maximum_birth_gap'] and len(states)<2*cap:
                youngest=min((s['age'] for s in states if s['energy']>40.),default=1000.)
                if youngest>c['maximum_birth_gap']:budget=max(budget,1)
            if c['birth_spacing']:
                budget=min(budget,int(self.clock-self.last_normal_birth>=c['birth_spacing']))
            candidates=[]
            for aid,v in views.items():
                s=v['state']
                threshold=max(c['birth_floor'],min(c['birth_energy'],.75*s['max_energy']))
                if s['age']>60 and young<max(2,cap//2):threshold=min(threshold,160.)
                safe=not v['predators'] or min(p['obs']['distance'] for p in v['predators'])>130.
                if s['energy']<=threshold or not safe or aid in self.retired_ids:continue
                quality=min(s['speed'],s['sprint_speed'])+.015*s['hearing_radius']+.003*s['vision_range']
                elder=(s['age']>70.,s['age']) if c['elder_birth_priority'] else (False,0.)
                candidates.append((elder,quality,s['energy'],-aid,aid))
            self.spawn_ids={item[-1] for item in sorted(candidates,reverse=True)[:budget]}
            if self.spawn_ids:self.last_normal_birth=self.clock
            for aid,v in views.items():
                s=v['state']
                v['resting']=aid in self.retired_ids or (s['energy']>min(c['hard_idle_energy'],c['idle_energy']*s['max_energy']) and aid not in self.spawn_ids)
                v['eligible']=not v['resting']
        super().allocate(views)

class HarvestPolicy(GenerationPolicy):
    def __init__(self, preset='harvest', **overrides):
        config = dict(maximum_birth_gap=20., harvest_depth=2,
                      harvest_discount=.7, harvest_switch=7., harvest_start=0.,
                      harvest_range=180., harvest_food_value=42.)
        config.update(overrides)
        super().__init__(preset, **config)

    def allocate(self, views):
        super().allocate(views)
        c = self.config
        if self.clock < c['harvest_start']:
            return
        owners = {v['mem'].target_id: aid for aid,v in views.items() if v['mem'].target_id is not None}
        for aid,v in sorted(views.items(), key=lambda kv: kv[1]['state']['energy']):
            if v['escaping'] or not v.get('eligible', True):
                continue
            s,m = v['state'],v['mem']
            terrain = {'river':.3,'swamp':.5,'desert':.8}.get(s['biome'],1.)
            speed = max(.1,min(s['speed'],s['sprint_speed']))
            # Age is public; the exact random aging onset is deliberately
            # not read. Expected late metabolic cost rises gradually.
            aging = max(0.,min(1.,(s['age']-60.)/60.)) * .1*s['age']
            unit_cost = (.05 + (1.32+aging)*.1/speed)/terrain
            foods = [f for key,f in v['visible'].items()
                     if owners.get(key,aid)==aid and f['obs']['distance']>10.]
            foods = sorted(foods,key=lambda f:f['obs']['distance'])[:12]
            values = []
            for f in foods:
                d = max(0.,f['obs']['distance']-10.)
                if d*unit_cost+1. >= s['energy']:
                    continue
                if not segment_clear((0.,0.),f['p'],v['edges'],0.):
                    continue
                def value(item):
                    born = self.birth_times.get(self.canonical(item['key']))
                    return min(60.,20.+2.*(self.clock-born)) if born is not None else c['harvest_food_value']
                score = value(f)-d*unit_cost
                remaining = [g for g in foods if g['id']!=f['id']]
                previous = f
                for depth in range(1,c['harvest_depth']):
                    options = []
                    for g in remaining:
                        gap = math.hypot(g['p'][0]-previous['p'][0],g['p'][1]-previous['p'][1])
                        if gap < c['harvest_range'] and segment_clear(previous['p'],g['p'],v['edges'],0.):
                            options.append((value(g)-max(0.,gap-10.)*unit_cost,g))
                    if not options:
                        break
                    extra,next_food = max(options,key=lambda row:row[0])
                    score += c['harvest_discount']**depth * max(0.,extra)
                    remaining = [g for g in remaining if g['id']!=next_food['id']]
                    previous = next_food
                if f['id']==m.target_id:
                    score += c['harvest_switch']
                values.append((score,-d,f['id']))
            if not values:
                continue
            key = max(values)[-1]
            if key == m.target_id:
                continue
            if m.target_id is not None:
                owners.pop(m.target_id,None)
            m.target_id = key
            owners[key] = aid
            self.metrics['harvest_target_switches'] += 1

class AppetitePolicy(HarvestPolicy):
    def __init__(self, preset='appetite', **overrides):
        config = dict(birth_energy=135., birth_floor=112., maximum_birth_gap=20.,
                      appetite_level=250., appetite_release=0., appetite_near=0.,
                      appetite_age_slope=0., appetite_start=0.)
        config.update(overrides)
        super().__init__(preset, **config)

    def shared_frames(self, views):
        super().shared_frames(views)
        c = self.config
        if self.clock < c['appetite_start']:
            return
        for aid,v in views.items():
            s,m = v['state'],v['mem']
            if aid in self.retired_ids or m.colonist:
                continue
            level = c['appetite_level']+c['appetite_age_slope']*max(0.,s['age']-50.)
            level = min(max(130.,level),c['idle_energy']*s['max_energy'])
            near = c['appetite_near'] and any(10.<f['obs']['distance']<c['appetite_near'] for f in v['fruits'])
            if getattr(m,'appetite_resting',False) and not near:
                threshold = level-c['appetite_release']
            else:
                threshold = level
            resting = s['energy']>threshold and aid not in self.spawn_ids
            m.appetite_resting = resting
            v['resting'] = resting
            v['eligible'] = not resting

STRATEGY_NAME = 'appetite'

CONFIG = json.loads('{"appetite_release": 70.0, "appetite_near": 80.0}')

def make_policy():
    return AppetitePolicy('appetite', **CONFIG)


from threading import RLock
from fastapi import Body, FastAPI

app = FastAPI(title="Survival Agent")
_lock = RLock()
_policy = make_policy()
_last_time = None
_last_signature = None
_last_response = None
_largest_id = -1

@app.post("/predict")
def predict(step: StepResponse = Body(...)):
    global _policy, _last_time, _last_signature, _last_response, _largest_id
    with _lock:
        signature = json.dumps(step.model_dump(), sort_keys=True, separators=(",", ":"))
        if signature == _last_signature:
            return _last_response
        supplied_time = step.sim_time if "sim_time" in step.model_fields_set else None
        # Some clients serialize the DTO's default zero on every step.
        if supplied_time == 0.0 and any(a.age > 0.0 for a in step.agent_status):
            supplied_time = None
        current_ids = {a.agent_id for a in step.agent_status}
        live_before = set(getattr(_policy, "memory", {}))
        ids_restarted = bool(current_ids and live_before and not current_ids & live_before
                             and min(current_ids) < _largest_id)
        if ids_restarted or (_last_time is not None and supplied_time is not None and supplied_time < _last_time):
            _policy = make_policy()
            _largest_id = -1
        if step.game_status != "ok" or not step.agent_status:
            _policy = make_policy()
            _largest_id = -1
            response = {"actions": []}
        else:
            states = [agent.model_dump() for agent in step.agent_status]
            response = {"actions": [a.model_dump() for a in _policy.decide_all(states, supplied_time)]}
            _largest_id = max(_largest_id, max(current_ids))
        _last_time = supplied_time
        _last_signature = signature
        _last_response = response
        return response

@app.get("/")
def index():
    return {"status": "ok", "strategy": STRATEGY_NAME}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9052)

