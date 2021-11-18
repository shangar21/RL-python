<style TYPE="text/css">
code.has-jax {font: inherit; font-size: 100%; background: inherit; border: inherit;}
</style>
<script type="text/x-mathjax-config">
MathJax.Hub.Config({
    tex2jax: {
        inlineMath: [['$','$'], ['\\(','\\)']],
        skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'] // removed 'code' entry
    }
});
MathJax.Hub.Queue(function() {
    var all = MathJax.Hub.getAllJax(), i;
    for(i = 0; i < all.length; i += 1) {
        all[i].SourceElement().parentNode.className += ' has-jax';
    }
});
</script>
<script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js?config=TeX-AMS_HTML-full"></script>
<script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
<script>mermaid.initialize({startOnLoad:true});</script>

# Chapter 3
# Finite Markov Decision

- Markov Decision Process = MDP
- Instead of looking at action and reward, we also look at action and reward in a specific situation
- Intead of $q_*(a)$ we look at $q_*(s,a)$ where $s$ is a specific state
  - Or estimate $v_*(s)$, value of each state given optimal action selections

## The Agent-Environment Interface

- ![](plots/markov_decision_process.png)
- The agent is the thing that makes the action and the environment is everything outside of that and gives the agent rewards as well as new sttes
- The agent of course wants to maximize its reward
- Time steps are discrete i.e. $t = 0, 1, 2, 3, ...$
- For each $t$ the agent receives a *state* $S_t \in \mathcal{S}$
- Based on $S_t$, the agent selects an *action* $A_t \in \mathcal{A}(s)$ 
- 1 time step later receives a *reward*  $R_{t+1} \in \mathcal{R} \subset \mathbb{R}$, then finds a new state $S_{t+1}$
- To put the above together, the sequence of state, action reward is as follows:
  - $S_0, A_0, R_1, S_1, A_1, R_2, \dots$
- For *finite* MDP, $\mathcal{S}, \mathcal{A}, \mathcal{R}$ have a finite number of elements
- Probability of a state $s'$ and a reward $r$ given a previous state and action is a discrete probability
  - $p(s', r|s, a) = Pr\{S_t = s', R_t = r| S_{t-1}=s, A_{t-1} = a\}$
  - $p: \mathcal{S} \times \mathcal{R} \times \mathcal{S} \times \mathcal{A} \rightarrow [0,1]$
  - Since $p(s', r|s, a)$ is a probability distribution:
    - $\sum\limits_{s'\in\mathcal{S}}\sum\limits_{r\in\mathcal{R}}p(s', r|s, a) = 1$
- *state-transition probabilities*:  $p(s'|s,a) := Pr\{S_t = s'|S_{t-1}=s, A_{t-1}  = a\} = \sum\limits_{r\in\mathcal{R}}p(s',r|s,a)$
- *Expected reward for State-Action pairs*: $r(s,a) := E[R_t|S_{t-1}=s, A_{t-1}=a] = \sum\limits_{r\in\mathcal{R}}r\sum\limits_{s'\in\mathcal{S}}p(s',r|s,a)$
- *Expected reward for state-action-next-state*: $r(s,a,s') := E[R_t|S_{t-1}=s, A_{t-1}=a, S_t=s'] = \sum\limits_{r\in\mathcal{R}}r\dfrac{p(s',r|s,a)}{p(s'|s,a)}$