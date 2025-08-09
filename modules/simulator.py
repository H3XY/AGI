# Simple imagination simulator using QNN predictions as rollouts (placeholder)
def rollout_simulation(agent, state, steps=3):
    traces = []
    for i in range(steps):
        decision = agent.reason(state)
        traces.append((i, decision))
    return traces
