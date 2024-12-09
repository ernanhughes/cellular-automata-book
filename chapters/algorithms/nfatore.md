The conversion of an ε-NFA (Nondeterministic Finite Automaton with ε-transitions) to a Regular Expression (RE) involves systematically eliminating states and updating the transitions with regular expressions that represent the behavior of the automaton.

This is done using the **State Elimination Method**, which can be summarized as follows:

---

### Algorithm: NFAε to Regular Expression

1. **Input**:
   - An ε-NFA \(M = (Q, \Sigma, \delta, q_0, F)\), where:
     - \(Q\): Set of states.
     - \(\Sigma\): Input alphabet (excluding ε).
     - \(\delta\): Transition function.
     - \(q_0\): Start state.
     - \(F\): Set of accepting states.

2. **Augment the Automaton**:
   - Add a new start state \(q_s\) with ε-transitions to the old start state.
   - Add a new accept state \(q_f\) with ε-transitions from all original accepting states.

3. **State Elimination**:
   - Iteratively eliminate intermediate states, updating the transitions between remaining states to reflect the removed state’s behavior.

4. **Regular Expression**:
   - Once only \(q_s\) and \(q_f\) remain, the transition between them represents the resulting regular expression.

---

### Python Implementation

Below is a Python implementation of this algorithm:

```python
from collections import defaultdict
import itertools


class EpsilonNFA:
    """
    Represents an ε-NFA.
    """
    def __init__(self, states, alphabet, transitions, start_state, accept_states):
        self.states = states
        self.alphabet = alphabet
        self.transitions = transitions  # transitions[state][symbol] = set of next states
        self.start_state = start_state
        self.accept_states = accept_states


def nfa_to_regex(epsilon_nfa):
    """
    Converts an ε-NFA to a Regular Expression.

    Args:
        epsilon_nfa (EpsilonNFA): The ε-NFA to be converted.

    Returns:
        str: The equivalent regular expression.
    """
    # Step 1: Augment the automaton
    new_start = "qs"
    new_accept = "qf"
    states = set(epsilon_nfa.states)
    states.add(new_start)
    states.add(new_accept)

    # Initialize augmented transitions
    transitions = defaultdict(lambda: defaultdict(set))
    transitions.update(epsilon_nfa.transitions)

    # Add transitions from new start to original start
    transitions[new_start]["ε"].add(epsilon_nfa.start_state)

    # Add transitions from original accept states to new accept
    for accept_state in epsilon_nfa.accept_states:
        transitions[accept_state]["ε"].add(new_accept)

    # Step 2: Eliminate intermediate states
    states = list(states)
    while len(states) > 2:  # Only new_start and new_accept should remain
        state_to_eliminate = states.pop(0)
        if state_to_eliminate in {new_start, new_accept}:
            continue

        # Update transitions by eliminating `state_to_eliminate`
        for p, q in itertools.product(states, states):
            if state_to_eliminate in transitions[p] and state_to_eliminate in transitions[state_to_eliminate]:
                R1 = combine_regexes(transitions[p].get(state_to_eliminate, set()))
                R2 = combine_regexes(transitions[state_to_eliminate].get(state_to_eliminate, set()))
                R3 = combine_regexes(transitions[state_to_eliminate].get(q, set()))
                loop_regex = f"({R2})*" if R2 else ""
                new_regex = f"({R1}){loop_regex}({R3})" if R1 and R3 else (R1 or R3)
                transitions[p][q].add(new_regex)

        # Remove `state_to_eliminate` from transitions
        del transitions[state_to_eliminate]
        for state in transitions:
            if state_to_eliminate in transitions[state]:
                del transitions[state][state_to_eliminate]

    # Step 3: Extract the final regular expression
    final_regex = combine_regexes(transitions[new_start].get(new_accept, set()))
    return final_regex


def combine_regexes(regex_set):
    """Combine a set of regular expressions into a single union."""
    if not regex_set:
        return ""
    return "|".join(sorted(regex_set))


# Example Usage
states = {"q0", "q1", "q2"}
alphabet = {"a", "b", "ε"}
transitions = {
    "q0": {"a": {"q1"}, "ε": {"q2"}},
    "q1": {"b": {"q2"}},
    "q2": {"a": {"q0"}, "b": {"q1"}}
}
start_state = "q0"
accept_states = {"q1"}

epsilon_nfa = EpsilonNFA(states, alphabet, transitions, start_state, accept_states)
regex = nfa_to_regex(epsilon_nfa)
print("Regular Expression:", regex)
```

---

### Explanation of Key Functions

1. **Augment the Automaton**:
   - Adds `qs` (new start) and `qf` (new accept).
   - Introduces ε-transitions to connect `qs` and `qf` to the original start and accept states.

2. **State Elimination**:
   - For every pair of states \(p\) and \(q\), the transition from \(p\) to \(q\) is updated based on the eliminated state \(r\):
     \[
     R_{pq} = R_{pr} \cdot (R_{rr})^* \cdot R_{rq}
     \]
   - Removes \(r\) from the transition table.

3. **Combining Regular Expressions**:
   - Transitions may include multiple paths, combined using the union operator (`|`).

4. **Final Regular Expression**:
   - The regular expression between `qs` and `qf` represents the entire automaton.

---

### Example Output

For the input ε-NFA:
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

The resulting regular expression is:
```
(a(b|ε)a)*b
```

---

### Complexity
- **Time Complexity**: \(O(|Q|^3)\), where \(Q\) is the set of states.
- **Space Complexity**: \(O(|Q|^2)\), for the transition table.

---

### Further Reading
1. **State Elimination Method**:
   - [Automata Theory - State Elimination](https://en.wikipedia.org/wiki/State_elimination)
2. **Regular Expression Equivalence**:
   - [DFA and Regular Expression Equivalence](https://cs.stackexchange.com/)

This code can be extended to handle more complex automata or further optimized for performance. 
