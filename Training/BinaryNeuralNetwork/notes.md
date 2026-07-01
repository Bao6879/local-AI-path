# Binary Neural Network Notes

To build up to GPTv2, we must first start at the lowest level. I built the gradient accumulation and back propagation myself after understanding it from Karpathy, so that it could be the basis for what's to come.

## Basics

Neurons consist of a weights vector and a bias, both containing or being numbers that start out between [-1, 1] — could grow out of those over the training period. It's like how much to warp the input with basic math. Then there's an activation function, tanh in this case, to squash the output to between (-1, 1).

So to get a basic input, you can give it a number, it'll give you a different one.

Lots of neurons being next to each other is called a layer, each seeing the full input vector. Each neuron transforms it, then passes the new modified numbers to the next layer.

However, you're just getting random numbers. To get the network to 'learn', you need back propagation.

## Back Propagation and Gradient

You give the network something to aim for (a goal). In this case, the word binary is because the network should output a binary answer — could be true/false, black/white, anything with just two possible values.

Take my example, which was checking light/dark, light being 1, dark being -1. If the network outputs the wrong number, we want to nudge it a bit in the correct direction. This is done by multiplying the -gradient with the learning rate.

Parameters are adjustable numbers in the network that will be changed while training (or teaching) it. The weights in each neuron and their biases are parameters. This model is tiny, with about 521 parameters.

The gradient is how much each parameter affects the final output. Say, if the gradient of `a` is 0.2, if I reduce `a` by 1, the final output would be reduced by 0.2 as well.

Learning rate scaling means you make sure the learning rate isn't too big (overshoots the goal and makes it wrong in the opposite way — say, if we wanted 0.1, but the learning rate is 0.2, so something starting at 0.2 can never reach the optimum, no matter how many steps), or too small (training takes forever).

And that's it! You run it for however long you want, and the model would slowly get better at its task.

## Notes

When I first learnt this, it kinda blew my mind. It's just math. Just a whole bunch of math.

But I'm sure there's lots to optimize and ways to make training take less time or compute, or the network smarter.

Thanks to Karpathy's videos on makemore and backpropagation!
