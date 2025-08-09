# Minimal planner stub that demonstrates generating candidate action sequences
def plan_candidates(state_repr, n=4):
    return [f"plan_{i}_for_{state_repr}" for i in range(n)]
