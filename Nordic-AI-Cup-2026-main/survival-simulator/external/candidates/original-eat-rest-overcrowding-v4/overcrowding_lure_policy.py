"""Experimental overcrowding-gated pawn overlay on the frozen eat/rest controller.

Baseline snapshot b7d0451c71616042: appetite_release=70, appetite_near=80.
All baseline configuration stays exact. Pawn parameters live separately.
"""
import math
from locked_eat_rest_baseline import AppetitePolicy, CONFIG, ActionRequest, point, unit, wrap, segment_clear
from locked_pawn_valuation import evaluate_agent


class OvercrowdingLurePolicy(AppetitePolicy):
    def __init__(self,**pawn_options):
        super().__init__('appetite',**CONFIG)
        self.pawn_options=dict(enabled=True,radius=160.,duration=20.,cooldown=20.,
                               minimum_runway=6.,target_gap=110.,burn_distance=25.,
                               minimum_population=100, shortage_fraction=.3,
                               hungry_fraction=.4, shortage_seconds=3., food_range=160.)
        unknown=set(pawn_options)-set(self.pawn_options)
        if unknown:
            raise ValueError(f'Only pawn options may change: {sorted(unknown)}')
        self.pawn_options.update(pawn_options)
        self.pawn_views={}
        self.active_pawns=set()
        self.valuations={}
        self.record_pawn_diagnostics=True
        self.shortage_since=None
        self.pressure_last_clock=None
        self.pawn_selection_events=[]
        self.pawn_pressure={}

    def shared_frames(self,views):
        # Finish the original rest/eat and reproduction decisions first.
        super().shared_frames(views)
        self.pawn_views=views
        self.active_pawns=set()
        if not self.pawn_options['enabled']:
            return
        if not self.overcrowding_confirmed(views):
            # Cancel existing missions too; a past shortage is not permission
            # to continue sacrificing when the population or food recovers.
            for v in views.values():
                if getattr(v['mem'],'pawn_until',-1.) > self.clock:
                    self.metrics['pawn_cancelled_pressure_cleared']+=1
                v['mem'].pawn_until=self.clock
                v.pop('pawn',None)
            return
        self.valuations={aid:evaluate_agent(v['state']) for aid,v in views.items()}
        covered=set()
        for aid,v in views.items():
            m=v['mem']
            protected=set(getattr(m,'pawn_protected',())) & set(views)
            if v['predators'] and min(p['obs']['distance'] for p in v['predators'])<180.:
                m.pawn_last_threat=self.clock
            continuing=(self.clock<getattr(m,'pawn_until',-1.) and protected and
                        self.clock-getattr(m,'pawn_last_threat',-100.)<2. and
                        aid in self.retired_ids)
            if continuing and v['predators']:
                nearest=min(v['predators'],key=lambda p:p['obs']['distance'])
                peers=[point(o) for o in v['agents'] if o.get('id') in protected]
                if peers and min(math.hypot(nearest['p'][0]-p[0],nearest['p'][1]-p[1])
                                 for p in peers)<nearest['obs']['distance']-10.:
                    continuing=False
                    m.pawn_until=self.clock
                    self.metrics['pawn_aborted_lost_target']+=1
            breeder_backup=(not self.valuations[aid]['can_reproduce'] or
                            any(self.valuations[bid]['can_reproduce'] for bid in protected))
            if continuing and breeder_backup:
                self.active_pawns.add(aid)
                covered.update(protected|{aid})

        proposals=[]
        for aid,v in views.items():
            s,m=v['state'],v['mem']
            if (aid in self.active_pawns or s['age']<10. or m.colonist or
                    aid in self.spawn_ids or self.clock<getattr(m,'pawn_cooldown_until',-1.)):
                continue
            # Only convert surplus agents identified by the ORIGINAL policy.
            # A local food-pressure estimate cannot nominate a productive agent.
            if aid not in self.retired_ids:
                continue
            neighbors={o['id']:point(o) for o in v['agents']
                       if o.get('id') in views and o['id']!=aid and
                       o['distance']<self.pawn_options['radius']}
            if not neighbors:
                continue
            members=set(neighbors)|{aid}
            breeders=[bid for bid in members if self.valuations[bid]['can_reproduce']]
            best=max(breeders or members,key=lambda bid:(self.valuations[bid]['keep_value'],-bid))
            if aid==best:
                continue
            values=sorted(self.valuations[bid]['keep_value'] for bid in members)
            if self.valuations[aid]['keep_value']>values[len(values)//2]:
                continue
            retired=aid in self.retired_ids
            # Retirement only ranks candidates AFTER the independent
            # large-population and sustained-food-shortage guard has passed.
            self.metrics['pawn_surplus_opportunities']+=1
            maintenance=self.valuations[aid]['maintenance_per_second']
            walk=max(0.,min(s['speed'],s['sprint_speed']))
            burn_rate=maintenance+.5*walk
            if s['energy']<10.+self.pawn_options['minimum_runway']*burn_rate:
                self.metrics['pawn_rejected_short_runway']+=1
                continue
            center=(sum(p[0] for p in neighbors.values())/len(neighbors),
                    sum(p[1] for p in neighbors.values())/len(neighbors))
            if math.hypot(*center)<10.:
                continue
            terrain={'river':.3,'swamp':.5,'desert':.8}.get(s['biome'],1.)
            routes=[]
            for enemy in v['predators']:
                d=enemy['obs']['distance']
                if not 45.<d<130. or not segment_clear((0.,0.),enemy['p'],v['edges'],0.):
                    continue
                if not enemy['measured'] or math.hypot(*enemy['velocity'])<.5:
                    continue
                away=unit((-enemy['p'][0],-enemy['p'][1]))
                predator_speed=math.hypot(*enemy['velocity'])
                approach=sum(a*b for a,b in zip(enemy['velocity'],away))/predator_speed
                if approach<.5 or walk*terrain<predator_speed:
                    continue
                other_distance=min(math.hypot(enemy['p'][0]-p[0],enemy['p'][1]-p[1])
                                   for p in neighbors.values())
                if d+10.>=other_distance:
                    continue
                route=self.safe_lure_exit(v,neighbors,enemy)
                if route is not None:routes.append(route)
            if not routes:
                self.metrics['pawn_rejected_no_safe_lure']+=1
                continue
            _,exit_direction=max(routes,key=lambda x:x[0])
            cost=self.valuations[aid]['keep_value']+.1*s['energy']
            proposals.append((not retired,cost,aid,members,exit_direction))
        for _,_,aid,members,direction in sorted(proposals,key=lambda p:p[:3]):
            if members & covered:
                continue
            m=views[aid]['mem']
            m.pawn_started=self.clock
            m.pawn_until=self.clock+self.pawn_options['duration']
            m.pawn_cooldown_until=m.pawn_until+self.pawn_options['cooldown']
            m.pawn_last_threat=self.clock
            m.pawn_protected=tuple(sorted(members-{aid}))
            m.pawn_route_heading=wrap(m.heading+math.atan2(direction[1],direction[0]))
            self.active_pawns.add(aid)
            covered.update(members)
            self.pawn_selection_events.append(dict(
                time=self.clock, agent_id=aid, **self.pawn_pressure))
            self.metrics['pawn_assignments']+=1
            self.metrics['pawn_retired_assignments']+=int(aid in self.retired_ids)
        for aid in self.active_pawns:
            v=views[aid]
            v['pawn']=True
            v['resting'],v['eligible']=False,False
            v['mem'].target_id=None
            # Do not change spawn_ids, retired_ids, or any teammate's flags.
            # A pawn's own reproduction is suppressed in its final action.

    def overcrowding_confirmed(self,views):
        """Observation-based shortage proxy, never a world-state food oracle.

        Greedily match hungry agents to distinct nearby observed fruit. This
        avoids counting the same fruit as a meal for every teammate. Unmatched
        hungry agents indicate pressure, not proven long-run carrying capacity.
        The population threshold gates pawns only; it does not limit births.
        """
        opts=self.pawn_options
        if self.pressure_last_clock is not None and self.clock<self.pressure_last_clock:
            self.shortage_since=None
        self.pressure_last_clock=self.clock
        population=len(views)
        hungry=sorted((aid for aid,v in views.items()
                       if v['state']['energy'] < opts['hungry_fraction']*
                          v['state']['max_energy']),
                      key=lambda aid:(views[aid]['state']['energy'],aid))
        used=set()
        matched=0
        for aid in hungry:
            view=views[aid]
            # Fruit already in eating range is omitted from baseline tracking.
            # Credit the agent's own immediate meal conservatively here.
            if any(o['type']=='Fruit' and o['distance']<=6.
                   for o in view['state']['observations']):
                matched+=1
                continue
            for fruit in sorted(view['fruits'],key=lambda f:f['obs']['distance']):
                key=self.canonical(fruit['key'])
                if key in used or fruit['obs']['distance']>opts['food_range']:
                    continue
                if not segment_clear((0.,0.),fruit['p'],view['edges'],0.):
                    continue
                used.add(key)
                matched+=1
                break
        unserved=len(hungry)-matched
        raw=(population>=opts['minimum_population'] and
             unserved>=max(1,math.ceil(opts['shortage_fraction']*population)))
        if not raw:
            self.shortage_since=None
        elif self.shortage_since is None:
            self.shortage_since=self.clock
        duration=0. if self.shortage_since is None else self.clock-self.shortage_since
        self.pawn_pressure=dict(population=population,hungry=len(hungry),
                                hungry_without_distinct_nearby_fruit=unserved,
                                shortage_seconds=round(duration,6))
        confirmed=raw and duration+1e-9>=opts['shortage_seconds']
        self.metrics['pawn_pressure_confirmed_ticks']+=int(confirmed)
        return confirmed

    def safe_lure_exit(self,view,neighbors,enemy):
        s=view['state']
        terrain={'river':.3,'swamp':.5,'desert':.8}.get(s['biome'],1.)
        walk=max(0.,min(s['speed'],s['sprint_speed']))
        away=math.atan2(-enemy['p'][1],-enemy['p'][0])
        current=min(math.hypot(*p) for p in neighbors.values())
        options=[]
        for offset in (0.,-math.pi/6.,math.pi/6.,-math.pi/3.,math.pi/3.,-math.pi/2.,math.pi/2.):
            direction=(math.cos(away+offset),math.sin(away+offset))
            end=tuple(3.*walk*terrain*x for x in direction)
            if not segment_clear((0.,0.),end,view['edges'],5.):continue
            clearance=self.predicted_clearance(view,walk,direction)
            if clearance<35.:continue
            distance=min(math.hypot(p[0]-end[0],p[1]-end[1]) for p in neighbors.values())
            if distance<current+10.:continue
            # Ensure the route does not pass close to a teammate on the way out.
            if any(min(math.hypot(p[0]-end[0]*t,p[1]-end[1]*t) for t in (.33,.67,1.))
                   <min(40.,math.hypot(*p)-5.) for p in neighbors.values()):continue
            options.append((distance-current+.1*min(clearance,100.),direction))
        return max(options,key=lambda x:x[0]) if options else None

    def pawn_steer(self,view):
        s,m=view['state'],view['mem']
        enemies=sorted(view['predators'],key=lambda p:p['obs']['distance'])[:4]
        terrain={'river':.3,'swamp':.5,'desert':.8}.get(s['biome'],1.)
        walk=max(0.,min(s['speed'],s['sprint_speed']))
        maximum=max(0.,s['sprint_speed']) if s['energy']>=s['max_energy']/5. else walk
        route=wrap(m.pawn_route_heading-m.heading)
        outward=(math.cos(route),math.sin(route))
        if not enemies:
            direction,_=self.slide(outward,view,.4*walk)
            return self.pawn_action(view,.4*walk,direction,.2,'PAWN_REPOSITION')
        nearest=enemies[0]
        d=nearest['obs']['distance']
        quiet=all(p['measured'] and math.hypot(*p['velocity'])<.5 for p in enemies)
        if quiet and d>80.:
            return self.pawn_action(view,0.,outward,nearest['obs']['angle'],'PAWN_WAIT')
        # Reuse the original escape controller for direction and emergency speed.
        # Its normal-agent behavior and configuration are untouched.
        escape_view=dict(view,fruits=[],visible={},resting=False,eligible=False,
                         escaping=True,escape_goal=tuple(240.*x for x in outward),
                         escape_goal_weight=1.)
        action=AppetitePolicy.steer(self,escape_view)
        direction=(math.cos(action.move_direction),math.sin(action.move_direction))
        speed=action.move_distance
        if d>=80.:
            predator_speed=math.hypot(*nearest['velocity'])/terrain if nearest['measured'] else 15.
            pace=max(0.,min(walk,predator_speed+2.+.2*(self.pawn_options['target_gap']-d)/terrain))
            pace=min(speed,pace)
            paced_direction,_=self.slide(direction,view,pace)
            clearance=self.predicted_clearance(view,pace,paced_direction)
            if clearance>=55.:
                speed,direction=pace,paced_direction
        clearance=self.predicted_clearance(view,speed,direction)
        burning=d<self.pawn_options['burn_distance'] or clearance<15.
        if burning:
            speed=maximum
            direction,_=self.slide(direction,view,speed)
            turn=math.copysign(math.pi,nearest['obs']['angle'] or 1.)
        else:
            turn=action.turn_angle
        self.metrics['pawn_burn_ticks']+=int(burning)
        self.metrics['pawn_observed_moving_predator_ticks']+=int(not quiet)
        return self.pawn_action(view,speed,direction,turn,'PAWN_BURN' if burning else 'PAWN_LURE')

    def predicted_clearance(self,view,speed,direction):
        terrain={'river':.3,'swamp':.5,'desert':.8}.get(view['state']['biome'],1.)
        vx,vy=tuple(speed*terrain*x for x in direction)
        clearance=math.inf
        for enemy in view['predators'][:4]:
            ex,ey=(enemy['velocity'] if enemy['measured'] else
                   (15.*terrain*math.cos(enemy['heading']),15.*terrain*math.sin(enemy['heading'])))
            for tick in (1.,2.,3.):
                clearance=min(clearance,math.hypot(enemy['p'][0]+ex*tick-vx*tick,
                                                  enemy['p'][1]+ey*tick-vy*tick))
        return clearance

    def pawn_action(self,view,speed,direction,turn,mode):
        s,m=view['state'],view['mem']
        terrain={'river':.3,'swamp':.5,'desert':.8}.get(s['biome'],1.)
        angle=math.atan2(direction[1],direction[0])
        m.last_move=tuple(speed*terrain*x for x in direction)
        m.last_turn=turn
        m.last_mode=mode
        m.search_heading=wrap(m.heading+angle)
        self.metrics['pawn_ticks']+=1
        self.metrics['pawn_energy_requested']+=(.05*min(speed,s['speed'])+
            .5*max(0.,speed-s['speed'])+min(math.pi,abs(turn))/(2.*math.pi))
        return ActionRequest(agent_id=s['agent_id'],move_distance=speed,
                             move_direction=angle,turn_angle=turn,spawn_agent=False)

    def steer(self,view):
        if view['state']['agent_id'] in self.active_pawns:
            return self.pawn_steer(view)
        return super().steer(view)

    def decide_all(self,states,sim_time=None):
        actions=super().decide_all(states,sim_time)
        for action in actions:
            if action.agent_id in self.active_pawns:
                # The original emergency-birth pass runs after steering.
                action.spawn_agent=False
        return actions
