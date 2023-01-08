# Wisp Trainers

This folder mostly contains useful trainers which apps may extend or build upon.

## The purpose of Wisp Trainers

<img src="../../media/trainer.jpg" alt="Wisp's Trainer" width="750"/>

In wisp, a **Trainer** typically governs the optimization process of some neural field, or other objects.
Trainers are commonly application specific, and it is common practice for users to extend Wisp's defaults trainers with their own. 

Wisp's `BaseTrainer` provides a framework of lifecycle events, which make it easier to integrate with the interactive renderer and various loggers
(such as tensorboard, weights & biases), as well as control scheduled events like lr decay, or grid pruning.

## Where Trainers fit in Wisp

* Trainers are usually created within an app's main script.
* Trainers typically take as input references to `Datasets` & `Pipelines` (containing `Tracers` and `NeuralFields`).
* Trainers typically initialize the weights of neural blocks with an optimizer.
* Finally, Trainers determine the losses used, and validation method applied.


## Trainer's Lifecycle Events

Wisp Trainers follow a structured scheme of life-cycle events:

```
    pre_training()                   # Runs once before training starts
    (i) for every epoch:
        |- pre_epoch()               # Runs once before every epoch starts
        (ii) for every iteration:
            |- pre_step()            # Runs once before every train step starts
            |- step()                # Runs a single train step
            |- post_step()           # Runs once after every iteration step ends
        |- post_epoch()              # Runs once after every epoch ends
    post_training()                  # Runs once after training ends
```
Each of these events can be overridden, or extended with `super()`.

To step through the various functions, wisp implements the following:

`iterate()` ensures a single `step()` gets executed. Any `pre` / `post` function required to step will be executed
along with the step function. 

`train()` runs the `iterate()` function indefinitely, until the optimization ends.

Typically, gui based apps which use the interactive renderer will invoke the `iterate()` function,
where `train()` is useful for headless runs.