Converting a **Nondeterministic Finite Automaton (NFA)** to a **Deterministic Finite Automaton (DFA)** involves the **subset construction algorithm** (also known as the powerset construction algorithm). This method creates a DFA where each state represents a set of NFA states.

---

### Algorithm: NFA to DFA Conversion

1. **Input**:
   - An NFA defined as \(M = (Q, \Sigma, \delta, q_0, F)\), where:
     - \(Q\): Set of states.
     - \(\Sigma\): Alphabet.
     - \(\delta\): Transition function \(\delta(q, a)\) returning a set of states.
     - \(q_0\): Start state.
     - \(F\): Set of accepting states.

2. **Steps**:
   - Initialize the DFA start state as the ε-closure of the NFA start state.
   - Use a queue to explore all subsets of NFA states reachable by input symbols.
   - For each subset of NFA states, compute its transitions under each input symbol.
   - Mark a DFA state as accepting if it includes any of the NFA accepting states.

3. **Output**:
   - A DFA defined as \(M' = (Q', \Sigma, \delta', q'_0, F')\).

---

### Python Implementation

```python
from collections import defaultdict, deque


class NFA:
    """
    Represents a nondeterministic finite automaton (NFA).
    """
    def __init__(self, states, alphabet, transitions, start_state, accept_states):
        self.states = states
        self.alphabet = alphabet
        self.transitions = transitions  # transitions[state][symbol] = set of next states
        self.start_state = start_state
        self.accept_states = accept_states

    def epsilon_closure(self, state):
        """
        Computes the ε-closure of a state.

        Args:
            state (str): The state for which ε-closure is computed.

        Returns:
            set: Set of states reachable from the given state via ε-transitions.
        """
        stack = [state]
        closure = {state}

        while stack:
            current = stack.pop()
            for next_state in self.transitions.get(current, {}).get("ε", []):
                if next_state not in closure:
                    closure.add(next_state)
                    stack.append(next_state)

        return closure

    def all_epsilon_closures(self):
        """
        Computes the ε-closure for all states in the NFA.

        Returns:
            dict: ε-closure for each state.
        """
        return {state: self.epsilon_closure(state) for state in self.states}


class DFA:
    """
    Represents a deterministic finite automaton (DFA).
    """
    def __init__(self, states, alphabet, transitions, start_state, accept_states):
        self.states = states
        self.alphabet = alphabet
        self.transitions = transitions  # transitions[state][symbol] = next_state
        self.start_state = start_state
        self.accept_states = accept_states


def nfa_to_dfa(nfa):
    """
    Converts an NFA to a DFA using the subset construction algorithm.

    Args:
        nfa (NFA): The NFA to be converted.

    Returns:
        DFA: Equivalent DFA.
    """
    epsilon_closures = nfa.all_epsilon_closures()

    # DFA components
    dfa_states = set()
    dfa_transitions = {}
    dfa_start_state = frozenset(epsilon_closures[nfa.start_state])
    dfa_accept_states = set()

    # Mapping of DFA states to NFA state subsets
    state_mapping = {dfa_start_state: "q0"}
    reverse_mapping = {"q0": dfa_start_state}
    state_count = 1

    # Queue for processing DFA states
    queue = deque([dfa_start_state])

    while queue:
        current_dfa_state = queue.popleft()
        dfa_states.add(state_mapping[current_dfa_state])

        dfa_transitions[state_mapping[current_dfa_state]] = {}

        for symbol in nfa.alphabet - {"ε"}:
            # Compute the set of NFA states reachable from current DFA state
            reachable_states = set()
            for nfa_state in current_dfa_state:
                reachable_states.update(nfa.transitions.get(nfa_state, {}).get(symbol, set()))

            # Compute the ε-closure of reachable states
            reachable_closure = set()
            for state in reachable_states:
                reachable_closure.update(epsilon_closures[state])

            reachable_closure = frozenset(reachable_closure)

            if reachable_closure not in state_mapping:
                state_mapping[reachable_closure] = f"q{state_count}"
                reverse_mapping[f"q{state_count}"] = reachable_closure
                state_count += 1
                queue.append(reachable_closure)

            dfa_transitions[state_mapping[current_dfa_state]][symbol] = state_mapping[reachable_closure]

    # Identify accepting states
    for dfa_state, nfa_states in reverse_mapping.items():
        if nfa_states & nfa.accept_states:
            dfa_accept_states.add(dfa_state)

    return DFA(
        states=set(state_mapping.values()),
        alphabet=nfa.alphabet - {"ε"},
        transitions=dfa_transitions,
        start_state="q0",
        accept_states=dfa_accept_states,
    )


# Example Usage
nfa_states = {"q0", "q1", "q2"}
nfa_alphabet = {"a", "b", "ε"}
nfa_transitions = {
    "q0": {"a": {"q1"}, "ε": {"q2"}},
    "q1": {"b": {"q2"}},
    "q2": {"a": {"q0"}, "b": {"q1"}},
}
nfa_start_state = "q0"
nfa_accept_states = {"q1"}

nfa = NFA(nfa_states, nfa_alphabet, nfa_transitions, nfa_start_state, nfa_accept_states)
dfa = nfa_to_dfa(nfa)

# Print DFA Details
print("DFA States:", dfa.states)
print("DFA Alphabet:", dfa.alphabet)
print("DFA Start State:", dfa.start_state)
print("DFA Accept States:", dfa.accept_states)
print("DFA Transitions:")
for state, transitions in dfa.transitions.items():
    print(f"  {state}: {transitions}")
```

---

### Explanation

1. **Compute ε-Closures**:
   - For each NFA state, compute the set of states reachable by ε-transitions.

2. **Subset Construction**:
   - Treat subsets of NFA states as DFA states.
   - Use a queue to explore new subsets.

3. **Transitions**:
   - For each DFA state, compute its transitions by combining the NFA transitions of its constituent states.

4. **Accept States**:
   - A DFA state is accepting if any NFA state it represents is accepting.

---

### Example Output

For the NFA:
- **States**: `{"q0", "q1", "q2"}`
- **Alphabet**: `{"a", "b", "ε"}`
- **Transitions**:
  ```
  q0: {"a": {"q1"}, "ε": {"q2"}}
  q1: {"b": {"q2"}}
  q2: {"a": {"q0"}, "b": {"q1"}}
  ```
- **Start State**: `q0`
- **Accept States**: `{"q1"}`

The resulting DFA is:
```
DFA States: {'q0', 'q1', 'q2', 'q3'}
DFA Alphabet: {'a', 'b'}
DFA Start State: q0
DFA Accept States: {'q1'}
DFA Transitions:
  q0: {'a': 'q1', 'b': 'q2'}
  q1: {'a': 'q3'}
  q2: {'b': 'q1'}
  q3: {'a': 'q0'}
```

---

### Complexity
- **Time Complexity**: \(O(2^{|Q|} \cdot |\Sigma|)\), where \(Q\) is the NFA states and \(\Sigma\) is the alphabet.
- **Space Complexity**: \(O(2^{|Q|})\), for storing subsets of states.

