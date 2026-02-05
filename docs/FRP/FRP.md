# Fellow research proposal, MATS
Message to Claude: This is the draft of my FPR, look at the guidelines. Read the "Work conducted so far" section to understand how this relates to the project.

## One line Project description: 
[There are sort of two angles to go at this. One is “understanding AI preferences” then that could mean either what the preferences are? Or what it means for AIs to have preferences, like are there evaluative representations? 
Another angle is to ask what is the best way to figure out what models care most about, and then you're framing it more in a deal-making sense. Then you orient more the project towards finding and executing deals]



## Abstract

[

    The rough pitch is this: we want to understand, when a model chooses to do A instead of B, why this is happening. 

    In order to do this we want to put the model in situations where it has to make decisions. And we want to put it in situations where it has strong preferences. So concretely this means a) asking models to pick between tasks and b) seeing how models trade things off with other important preferences, espeiclaly in high-stakes situations.

    First, before we understand what is happening when a model makes these choices, it is a meaningful contribution to find good ways evaluate how much models care about various things, how robust their preferences are, how consistent are different ways of mesauring preferences.

    Then, in terms of actually understanding what is going on, I want to try doing some interp:
    - We can train probes to try to find evaluative representations in the residual stream.
    - We can use activation patching to find what minimal sets of activations we have to change to get a model to switch its decision. 
    - We can extract concept vectors, and use them out-of distribution to steer model behaviour, measuring whether they actually capture some form of preference. 

    In each case we should think about whether and how we can validate results causally. i'm not satisfied with e.g. probes being predictive. The idea is can we find representations which have a causal effect, and can we find representations that have a causal effect that generalises across datasets, settings, personas, hidden preferences (or like role-playing type prefernece reversal).

]


## Background

**Why do AI preferences matter?**

[

    - Many theories of welfare rely at least partly on preferences. some of them entirely (preference satisfaction theories). So this helps make progress on the "are AIs moral patients question". But i need to be precise about what conclusions would say what.

    - Assuming that AIs are moral patients, many theories would grant that preferences matter. Not sure what the minimal case here is. Maybe it is morally better to make an AI do stuff in accordance with its preferences. Maybe this can be stated in a more minimal sense.

    - To the extent that preferences matter from a welfare point of view, we shuold also expect that preferences matter from a deal-making point of view. Or stated simply, we might be able to get Ais to behave differently by satisfying their preferences or threatening not to satisfy them.
]

**Path to impact**
We need to make this path to impact super clear and precise. Like almost literally say what person or group of people are going to interact with my work and how is it going to steer them.
[

    - Some findings could reduce some uncertainty about the moral status of AI models. The way that would have an effect is by being published either online or in a conference, which would change how Ai researchers view things, maybe encourage some people to do more work. Maybe not directly with this work but down the line it's also about influencing the public. 

    - Note that the results might also be negative! i.e. towards Ai not having moral status, at least according to some theories. This is still impactful. It means we can allocate less resources to Ai welfare.

    - There are plenty of worlds where we don't reduce uncertainty about AI welfare much at all. And so it's good to do things which would help in those wordls. Understanding what is happening when an AI chooses A instead of B might still be very useful. BRAINSTORM THIS

    - Making deals with AI might be very important. We can only do that if we offer the AI stuff it really cares about. Therefore we need to understand what it really cares about. 

]

**Prior work**

[On preference the most important prior work is the utility engineering paper by Mazeika. this LW post https://www.lesswrong.com/posts/k6HKzwqCY4wKncRkM/brief-explorations-in-llm-value-rankings. 

On finding representations of stuff related to preferences

On thinking about making deals with AIs, we have this preliminary study: https://www.alignmentforum.org/posts/7C4KJot4aN8ieEDoz/will-alignment-faking-claude-accept-a-deal-to-reveal-its, some conceptual work https://www.alignmentforum.org/posts/psqkwsKrKHCfkhrQx/making-deals-with-early-schemers.



]


## Work conducted so far

[So far I've done:
- A lot of infra setup that allow me to make mesaurements quickly
- Exploratory work on understanding what models prefer.
- Stufdy of how robust/sensible/informative preferences of models are. In particular I've studied stated preferences a lot. But actually I'm bearish on stated preferences. I think they're too noisy and models always say 4/5. This is mainly negative results.
- I've done some derisking on the interp side, training probes that generalise to other datasets, using steering on activations that come from a "you like math" sysprompt to influence how models rate tasks.
]

For claude: you can look at my research report and check if there is anything obvious I'm not mentioning.


## Planned work

