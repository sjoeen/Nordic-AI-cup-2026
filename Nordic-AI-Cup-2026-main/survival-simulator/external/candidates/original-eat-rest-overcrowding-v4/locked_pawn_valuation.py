"""Standalone valuation helper; no policy inheritance or configuration changes."""

def evaluate_agent(state, *, nearest_food_distance=None,
                   food_income_per_second=0., activity_cost_per_second=0.,
                   aging_observed=False):
    """Return an explainable keep-value for one agent; higher means retain it.

    Inputs are public status, observed food distance, and rates inferred from
    the agent's own energy history. No true lifespan or unseen food is used.
    Rates default to zero until enough history exists.
    """
    energy = max(0.,float(state['energy']))
    age = max(0.,float(state['age']))
    walk = max(.1,min(state['speed'],state['sprint_speed']))
    terrain = {'river':.3,'swamp':.5,'desert':.8}.get(state['biome'],1.)
    aging_fraction = 1. if aging_observed else min(1.,max(0.,(age-60.)/60.))
    maintenance = 1.+.1*age*aging_fraction
    net_income = max(0.,food_income_per_second)-max(0.,activity_cost_per_second)-maintenance
    reserve_after_20s = max(0.,energy+20.*net_income)
    # Age discounts expected future contribution; it never determines a role.
    future_weight = max(.08,1.-age/120.)
    trait_quality = (.5*walk/10.+.25*max(0.,state['hearing_radius'])/50.+
                     .25*max(0.,state['vision_range'])/200.)
    fertile = state['max_energy']>100.
    food_opportunity = 0.
    if nearest_food_distance is not None:
        distance = max(0.,nearest_food_distance-10.)
        trip_cost = distance/terrain*(.05+maintenance/(10.*walk))
        if trip_cost<energy:
            food_opportunity = max(0.,42.-trip_cost)
    keep_value = future_weight*(trait_quality*(reserve_after_20s+25.*fertile)+.5*food_opportunity)
    return dict(agent_id=state['agent_id'],keep_value=keep_value,
                reserve_after_20s=reserve_after_20s,net_income_per_second=net_income,
                maintenance_per_second=maintenance,trait_quality=trait_quality,
                future_weight=future_weight,food_opportunity=food_opportunity,
                can_reproduce=bool(fertile),newborn=age<10.,energy=energy,
                rationale='Estimated future energy contribution; role selection also requires crowding and a useful lure.')
