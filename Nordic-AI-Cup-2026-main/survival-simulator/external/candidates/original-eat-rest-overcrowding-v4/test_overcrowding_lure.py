"""Decision checks, not survival benchmarks or changed-size simulations."""
import copy
import math
import random
import unittest

from locked_eat_rest_baseline import Memory, make_policy
from overcrowding_lure_policy import OvercrowdingLurePolicy


def state(aid, age=25., energy=75.):
    return dict(agent_id=aid, age=age, energy=energy, observations=[],
                biome='forest', speed=20., sprint_speed=30., hearing_radius=100.,
                vision_range=300., vision_angle=math.pi/3., max_energy=500.)


def scenario(count, fed=False, fruit_count=0):
    views={}
    positions={0:(0.,0.),1:(140.,50.),2:(150.,-50.)}
    for aid in range(count):
        s=state(aid,110. if aid==0 else 25.,400. if fed else 75.)
        if aid==0 and not fed:s['energy']=240.
        agents=[]
        enemies=[]
        if aid in positions:
            x,y=positions[aid]
            for bid,(bx,by) in positions.items():
                if bid==aid or bid>=count:continue
                angle=math.atan2(by-y,bx-x)
                agents.append(dict(type='Agent',id=bid,distance=math.hypot(bx-x,by-y),
                                   angle=angle,rel_dir=angle+math.pi))
            p=(65.-x,-y)
            obs=dict(type='Predator',distance=math.hypot(*p),angle=math.atan2(p[1],p[0]),rel_dir=0.)
            enemies=[dict(p=p,heading=math.pi,velocity=(-15.,0.),measured=True,obs=obs)]
        fruits=[dict(key=('meal',i),p=(80.,float(i)),
                     obs=dict(type='Fruit',distance=math.hypot(80.,i),angle=math.atan2(i,80.)))
                for i in range(fruit_count)]
        s['observations']=agents+[e['obs'] for e in enemies]+[f['obs'] for f in fruits]
        m=Memory(rng=random.Random(aid));m.colonist=False;m.patch_since=None
        views[aid]=dict(state=s,mem=m,agents=agents,predators=enemies,fruits=fruits,
                        edges=[],escaping=bool(enemies),resting=False,eligible=True,
                        visible={},translation=(0.,0.))
    p=OvercrowdingLurePolicy();p.clock=4.;p.retired_ids={0};p.spawn_ids={1}
    return p,views


def confirm(p,views):
    for t in (4.,5.,6.,7.):
        p.clock=t;p.shared_frames(views)


class OvercrowdingChecks(unittest.TestCase):
    def test_small_population_never_selects_even_if_starving_and_retired(self):
        for count in (3,4,10,20,99):
            with self.subTest(count=count):
                p,v=scenario(count);confirm(p,v)
                self.assertEqual(p.active_pawns,set())
                self.assertEqual(p.metrics['pawn_assignments'],0)

    def test_large_fed_population_does_not_trigger(self):
        p,v=scenario(100,fed=True);confirm(p,v)
        self.assertEqual(p.active_pawns,set())

    def test_distinct_observed_food_for_every_hungry_agent_blocks_selection(self):
        p,v=scenario(100,fruit_count=100);confirm(p,v)
        self.assertEqual(p.active_pawns,set())
        self.assertEqual(p.pawn_pressure['hungry_without_distinct_nearby_fruit'],0)

    def test_large_starving_population_requires_sustained_shortage(self):
        p,v=scenario(100)
        p.shared_frames(v)
        self.assertEqual(p.active_pawns,set())
        confirm(p,v)
        self.assertEqual(p.active_pawns,{0})
        self.assertEqual(p.spawn_ids,{1});self.assertEqual(p.retired_ids,{0})
        event=p.pawn_selection_events[0]
        self.assertEqual(event['population'],100)
        self.assertEqual(event['hungry_without_distinct_nearby_fruit'],99)
        self.assertGreaterEqual(event['shortage_seconds'],3.)

    def test_same_fruit_is_not_counted_as_100_meals(self):
        p,v=scenario(100,fruit_count=1);confirm(p,v)
        self.assertEqual(p.active_pawns,{0})
        self.assertEqual(p.pawn_pressure['hungry_without_distinct_nearby_fruit'],98)

    def test_small_population_ends_active_pawn_immediately(self):
        p,v=scenario(100);confirm(p,v)
        self.assertEqual(p.active_pawns,{0})
        p.clock=7.1;p.shared_frames({aid:v[aid] for aid in (0,1,2)})
        self.assertEqual(p.active_pawns,set())
        self.assertLessEqual(v[0]['mem'].pawn_until,p.clock)

    def test_food_recovery_ends_role_and_resets_confirmation(self):
        p,v=scenario(100);confirm(p,v)
        for view in v.values():view['state']['energy']=400.
        p.clock=7.1;p.shared_frames(v)
        self.assertEqual(p.active_pawns,set())
        self.assertIsNone(p.shortage_since)
        for aid,view in v.items():view['state']['energy']=240. if aid==0 else 75.
        p.clock=8.;p.shared_frames(v)
        self.assertEqual(p.active_pawns,set())
        self.assertEqual(p.pawn_pressure['shortage_seconds'],0.)

    def test_no_predator_no_pawn_even_under_confirmed_pressure(self):
        p,v=scenario(100)
        for view in v.values():view['predators']=[]
        confirm(p,v);self.assertEqual(p.active_pawns,set())

    def test_original_actions_configuration_births_and_retirement_are_preserved(self):
        for count in (3,20,100):
            p=OvercrowdingLurePolicy();baseline=make_policy()
            self.assertEqual(p.config,baseline.config)
            for tick in range(10):
                states=[state(aid,110.+tick*.1 if aid==0 else 25.+tick*.1,
                              400. if count==100 else 180.) for aid in range(count)]
                actual=p.decide_all(copy.deepcopy(states),100.+tick*.1)
                expected=baseline.decide_all(copy.deepcopy(states),100.+tick*.1)
                self.assertEqual([a.model_dump() for a in actual],
                                 [a.model_dump() for a in expected])
                self.assertEqual(p.spawn_ids,baseline.spawn_ids)
                self.assertEqual(p.retired_ids,baseline.retired_ids)


if __name__=='__main__':unittest.main()
