---
layout: post
title: "First steps into agents coaching"
author: "Rafael Cardenas"
categories: personal projects
tags: [Reinforcement Learning, post]
image: 
---


Today (and the last days) we have been training some deep Q network agents to control a fermentation process.

We can control the Nitrate inflow rate.

Our objective is defined as:

![Plot1](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/7d6ae2d6-35cf-4923-b234-ed1e3c90b5a2/Untitled.png "Plot1")

We want to minimize waste (nitrate N) while maximizing concentration of biomass (x) (mazimisation of productivity)

For that, we went through a series of steps.

Instead of jumping right in to the problem and build the best arquitecture I can find on ArXiv, I started with a use case i found in the awesome RL class of Stanford CS234, in assignment #2 students are asked to implement the famous DQN paper by Mnih (2015) in an incremental fashion. For the first part, it is instructed to build a linear function approximation for the networks used in the DQN implementation, we can do that implemementing a NN as deep as we want but **without** activation functions, (at the end it will be a fancy linear regression ;))

Since the assignment is originally implemented in Tensorflow, I translated it to Pytorch ( 🔥) and by doing that I became more aware of what was going on under the hood in the script. 
The architecture of the network was very simple, the input layer was batch_size x state_dimension, I used the predetermined configurations of the assignment to keep it simple hoping to get similar results. 

Even though at first it sounded fairly straight forward, playing with neural Networks (even simple ones) is tricky, I had to do a fairly amount of fine tunning to the networks before I got a good result; although not the best, as can be seen on the following pictures. 
On the right is the expected result, the maximum possible reward (validation) is 4.1, my implementation (left) was able to reach it after some training but when I did the validation, the result I got was 4. Reaching that 4.1 was key in this use case because it demonstrated 'long term' planning of the agent, which when I became aware, was really awesome.

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/70d2b4f5-18d7-462b-a696-af16aecc7f63/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/70d2b4f5-18d7-462b-a696-af16aecc7f63/Untitled.png)

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/4f09062b-d8dd-49b2-9161-3dcfe735cdfa/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/4f09062b-d8dd-49b2-9161-3dcfe735cdfa/Untitled.png)

Anyways, once I overcame that little dissapointment, I started pivoting all that accumulated knowledge (and scripts) from Stanford's assignment to solve the *real* use case: Control of the bioprocess. 

Some preliminaries:

- We are going to use two neural nets (NN) for the DQN implementation, one is the "estimation network" and the other is the "target" network. 
Intuitively, the estimation network is the one that is constantly learning and moving its parameters to best approximate the values that the "target" network is providing. The values provided by this last network encode the best possible actions we can take with our agent. If everything works, both nets would have learned a good representation of the action value function and, as result, the policy that is encoded on them will be "optimal".

For the implementation, I used the tools that where presented in the paper by Mnih et. al.: 

- Freezing of the taget net's weights
- Experience replay
- Not convnets, though.

Still in the realm of linear function approximation, out of curiosity, I tried to implement the control agent for the use case using only **linear functions**, as expected, the agent did not managed to achieve much reward consistently and its policy, in its best cases, the reward achieved had huge variance and much of the apparent performance depended on 'having' luck in the visited states. 

After not a lot of tweaking in the hyperparameters of the 'Linear' agent, and not seeing improvement in the agent's performance, I resolved to move on to the powerful universal approximation realm that our beloved and popular neural nets offer.

I started again with a relatively simple architecture for the layers with one layer and ReLu as activation function, along the way I decorated the NN with learning rate decay and switched from SGD to ADAM for the optimizer (Tip from [Andrej](https://www.youtube.com/watch?v=gYpoJMlgyXA&list=PLkt2uSq6rBVctENoVBg1TpCC7OQi31AlC&index=5)).

One of the best results I could get with those settings was the following. In the right, we can see that the agent is able to reach a very fairly high reward on approximately 50 % of the training episodes, which of course is really bad for practical applications, i.e. *how would you feel if the controller that moves the fuel needle in a refinery messes up 50% of the time?*  😢

On the right, we can see one of the major drawbacks of function approximation (to my knowledge), which is *estimating correctly the wrong estimation.* i.e. reaching convergence to the target function but, since this target is also estimated, we could get it wrong; this was one of the perfect examples of that case.

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/feb3f577-d3ec-4001-8c03-b7ddd1f6d9ec/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/feb3f577-d3ec-4001-8c03-b7ddd1f6d9ec/Untitled.png)

Left: loss function. Right: reward per episode.

We can explain that in several ways, one could be that the small NN has not enough expressive power to represent the function we want to approximate (we need more layers/neurons), other possible reason is that we are 'updating' the parameters of the target network too son (see Mnih paper for details about that); also it is possible we have a bug in the code!

> Reinforcement Learning tends to fail silently, the code will run, but the agent won't learn.

I decided to bet on the first possibility and augment the networks a bit with another fully connected layer with ReLU and the output layer without activation function, nothing esotheric here yet (I am looking at you "[Inception](https://static.googleusercontent.com/media/research.google.com/es//pubs/archive/43022.pdf)").

After that relatively simple change, things got exciting!. Within the first training cycles, I got good and stable results (and nicer plots ;))

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/4df246a0-76e4-4abf-9620-3d599ef2e82b/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/4df246a0-76e4-4abf-9620-3d599ef2e82b/Untitled.png)

*Hyperparameters used*: 30k training eps, 1k: length of the buffer in experience replay...

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/1a1bcb60-115a-4ecf-a696-86aa156efb36/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/1a1bcb60-115a-4ecf-a696-86aa156efb36/Untitled.png)

...decay rate of ADAM: 5e-3 to 1e-3. Swapping of parameters every 1000 steps.

The left plot is showing one training cycle using the new Nets' configuration and can be seen clearly how the reward is high and consistent throughout, further, we achieve better performance than the linear Net in less training steps; finally, the loss function is decaying quickly and as expected.

In the next plot we can see the validation reward of the above mentioned agents. This plot is interesting and lets us highlight important remarks.

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/bcb0b03a-431a-45c4-a57b-0a009b1ab53a/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/bcb0b03a-431a-45c4-a57b-0a009b1ab53a/Untitled.png)

Right away the low validation of the grey agent stands out, if we look at its reward line in the previous plot (in the left), we can see how it fell down rapidly as it approached the end of the training episode the same behavior took place on its validation round. In contrast, the 'orange' agent ended up the training episode with higher reward and performed similarly in the validation test.

One reason of the bad performance of the grey agent could be what I call *blind faith*, illness that tends to atack the estimation network when the target network starts to move from a good function approximation to a bad guess, route that the estimation network will follow for a while (eps 20k to 25k), then, the swaping of the weights will come (25kth step) and both networks, already tripping around a bad approximation, will fall in a region where both of them will be close, as shows the loss, but the estimation will end up following *blindly the target* until the end of the episode.

(*The latter was a conclusion that looks possible to me, I have looked some distribution plots of Net's parameters aiming to justify this claims, but haven't found substantial evidence yet)*

Keeping the eyes on the goal, we are optimizing loss and rewards so that out agent provide us the best set of actions to maximise the biomass producction, thus, we are also interested in seeing the actions that the agent took the most where the reward was peaking: 

 

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/b10d27b4-da47-480b-9874-f5c03b5c508b/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/b10d27b4-da47-480b-9874-f5c03b5c508b/Untitled.png)

Distribution of actions of a well trained agent i.e final reward of 200

These results are interesting, as we can see, the actions that cultivated the improvement of reward are somewhat evenly distributed, but tend to agregate a little at both ends of the action espectrum but not completely. 

Lets compare it, for example, with a bad performing agent,  as we can see below, actions tend to completely agregate in both low and high end of the espectrum, our agent holds on to a *all or nothing* policy; which of course is too agressive for the mayority of the cases where we would like to change the influx of nitate just a little, for example.

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/d585390d-9072-4b73-86fc-8e23f6ba7389/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/d585390d-9072-4b73-86fc-8e23f6ba7389/Untitled.png)

Distributions of actions of a bad performing agent, i.e final reward = -100

Digging more a little bit into our *blind faith* theory, if we look at the same epoch where the loss function fell abruptly (right), in this case at +-5kth step, two things stand out: 

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/52033fd3-b878-4d12-846f-598bdb48704e/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/52033fd3-b878-4d12-846f-598bdb48704e/Untitled.png)

Ill-conditioned agent's reward during training

The falling is around the same time the swaping of parameters takes place, and second, the distributions of actions suddenly change; we cas see that the density starts allocating at the extremes of the actions' spectrum. I just tried to present my previous hypothesis more concretely, but I am not truly convinced yet, further experiments have to be made in this direction to gain (or loose) coinfidence in these affirmations. 

### Final thoughts

These last weeks I have been playing a lot with reinforcement learning theory and implementations, it has been really interesting and I have barely scratched the surface of this field, fact that enhances my motivation to keep learning.

Because of my amateurism in the field I am specially cautious when I point out areas of opportunity in this exciting field. However, I have noticed the gravitational pull that the need of robustness and generalization exerts over the RL community. I have experienced the difficulties of reproducing experiment results and the feeling of both magic and weekness emanating from RL algorithms implementation, as a result, I have a lot of questions (concerns I will address in another post) and the feeling that we don't have yet good answers. These are exciting times for this community, truly great advancements have been made, but there is still a long way to go. This journey, far from tedious, promises an exciting future.

Thanks for reading, until next time :).

  

*Acknowledgements:*

Antonio, Max, Tom.