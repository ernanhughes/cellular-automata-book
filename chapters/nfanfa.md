The conversion from an ε-NFA (Nondeterministic Finite Automaton with ε-transitions) to an NFA (Nondeterministic Finite Automaton without ε-transitions) involves removing all ε-transitions and updating the transition table accordingly.

Here's a Python implementation of **ε-NFA to NFA** conversion:

---

### Python Implementation: ε-NFA to NFA

```python
from collections import defaultdict

class EpsilonNFA:
    """
    Represents an ε-NFA (Nondeterministic Finite Automaton with ε-transitions).
    """
    def __init__(self, states, alphabet, transition, start_state, accept_states):
        self.states = states
        self.alphabet = alphabet
        self.transition = transition  # transition[state][symbol] = set of next states
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
            for next_state in self.transition.get(current, {}).get("ε", []):
                if next_state not in closure:
                    closure.add(next_state)
                    stack.append(next_state)
        return closure

    def all_epsilon_closures(self):
        """
        Computes the ε-closure for all states in the ε-NFA.

        Returns:
            dict: ε-closure for each state.
        """
        return {state: self.epsilon_closure(state) for state in self.states}


class NFA:
    """
    Represents an NFA (Nondeterministic Finite Automaton without ε-transitions).
    """
    def __init__(self, states, alphabet, transition, start_state, accept_states):
        self.states = states
        self.alphabet = alphabet
        self.transition = transition  # transition[state][symbol] = set of next states
        self.start_state = start_state
        self.accept_states = accept_states


def epsilon_nfa_to_nfa(epsilon_nfa):
    """
    Converts an ε-NFA to an NFA by eliminating ε-transitions.

    Args:
        epsilon_nfa (EpsilonNFA): The ε-NFA to be converted.

    Returns:
        NFA: Equivalent NFA without ε-transitions.
    """
    epsilon_closures = epsilon_nfa.all_epsilon_closures()
    new_transition = defaultdict(lambda: defaultdict(set))
    new_accept_states = set()

    for state in epsilon_nfa.states:
        closure = epsilon_closures[state]

        # Update transitions for each symbol in the alphabet (excluding ε)
        for symbol in epsilon_nfa.alphabet - {"ε"}:
            reachable_states = set()
            for s in closure:
                reachable_states.update(epsilon_nfa.transition.get(s, {}).get(symbol, []))
            for s in reachable_states:
                new_transition[state][symbol].update(epsilon_closures[s])

        # Update accept states
        if closure & epsilon_nfa.accept_states:
            new_accept_states.add(state)

    return NFA(
        states=epsilon_nfa.states,
        alphabet=epsilon_nfa.alphabet - {"ε"},
        transition=new_transition,
        start_state=epsilon_nfa.start_state,
        accept_states=new_accept_states
    )


# Example Usage
states = {"q0", "q1", "q2"}
alphabet = {"a", "b", "ε"}
transition = {
    "q0": {"ε": {"q1", "q2"}},
    "q1": {"a": {"q1"}},
    "q2": {"b": {"q2"}}
}
start_state = "q0"
accept_states = {"q1"}

epsilon_nfa = EpsilonNFA(states, alphabet, transition, start_state, accept_states)
nfa = epsilon_nfa_to_nfa(epsilon_nfa)

# Print NFA Details
print("States:", nfa.states)
print("Alphabet:", nfa.alphabet)
print("Start State:", nfa.start_state)
print("Accept States:", nfa.accept_states)
print("Transition Function:")
for state, transitions in nfa.transition.items():
    print(f"  {state}: {dict(transitions)}")
```

---

### Explanation

1. **Input**:
   - **States**: The set of states in the ε-NFA.
   - **Alphabet**: The set of symbols (including `ε`) in the ε-NFA.
   - **Transition Function**: A dictionary where keys are states and values are dictionaries mapping symbols to sets of next states.
   - **Start State**: The initial state.
   - **Accept States**: The set of accepting states.

2. **Steps**:
   - Compute the **ε-closure** for each state (all states reachable via `ε` from a given state).
   - For each state and symbol, compute reachable states using the ε-closures.
   - Update the accept states to include states whose ε-closures contain any of the original accept states.

3. **Output**:
   - An equivalent NFA without `ε`-transitions.

4. **Complexity**:
   - **Time**: \(O(|Q|^2 \cdot |\Sigma|)\), where \(Q\) is the set of states and \(\Sigma\) is the alphabet size.
   - **Space**: \(O(|Q|^2)\), for storing ε-closures.

---

### Example Output

For the input ε-NFA:

- **States**: `{"q0", "q1", "q2"}`
- **Alphabet**: `{"a", "b", "ε"}`
- **Transition Function**:
  ```
  q0: {"ε": {"q1", "q2"}}
  q1: {"a": {"q1"}}
  q2: {"b": {"q2"}}
  ```
- **Start State**: `q0`
- **Accept States**: `{"q1"}`

The equivalent NFA is:

```
States: {'q0', 'q1', 'q2'}
Alphabet: {'a', 'b'}
Start State: q0
Accept States: {'q0', 'q1'}
Transition Function:
  q0: {'a': {'q1'}, 'b': {'q2'}}
  q1: {'a': {'q1'}}
  q2: {'b': {'q2'}}
```

This NFA is equivalent to the original ε-NFA but has no ε-transitions.

---

### References
1. **Theory of Automata**:
   - [Introduction to Automata Theory](https://en.wikipedia.org/wiki/Automata_theory)
2. **ε-NFA and NFA**:
   - [Finite Automata Basics](https://www.geeksforgeeks.org/fa-and-nfa-introduction/)

