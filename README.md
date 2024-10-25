# atlas-gan
May The Great Programmer have mercy on my code

Making sure this is done right

I tried and failed for multiple hours straight to do the advanced project
I'll take the opportunity to add my notes from the original GAN Project:

# Notes about the initial training:
# epoch 01 - noise like normal

# epoch 03 - This is where David had the start of some decent results
#           This is where our experiments differed, 
#           despite having the same code

# epoch 05 - We're looking at the start of some rough numbers. 
#           I think this is where I should consider early stopping as an option.
#           Most Gans that I've seen benefit from shorter training times
#           We'll see. That sounds pretty boring honestly
#           But it's also the only difference between David and I's code.

# epoch 10 - This does not look like numbers. They all look like eights.
#           Bad eights at that. This frustrates me.

# epoch 15 - I'm beginning to wonder if I'm remembering David's code right.
#           It's possible he stopped training at 25 rather than 10.

# epoch 20 - I see sevens and potential eights.
#           threes too. Maybe some Zeroes

# epoch 25 - These numbers are not much better. This GAN is not what was promised.

# epoch 30 - Okay!! These are starting to look decent!
#           I've got some nines. A few sevens. a one and some zeroes

# epoch 35 - I can't tell if these are getting better.
#           Maybe more defined.
#           Threes are looking decent

# epoch 40 - This looks really good!!!
#           Idk if it's human level. But close

# epoch 45 - Still decent but I feel we fell back some.
#           I oughta actually learn how gradient descent works
#           I'm going to be given a run for my money tonight.

# epoch 50 - We fell back some
#           RIP

# Notes about the new training w/ architecture change:
# epoch 01 - noise like normal

# Really quick, this is really quick.
# And bad.
# Like, really really bad. 

# epoch 01 - noise
# epoch 03 - We're cooking (ish)
# epoch 05 - nvm
# epoch 10 - Ehh? Probs not.
# epoch 15 - There's something that may resemble numbers
#           Not very well though.
# epoch 20 - Yeah we've backtracked
# epoch 25 - There's what might be some sevens
#           Some ones and fives too
# epoch 30 - Honestly this really isn't looking half bad
#           Considering we've done the equivelant of turning off brain cells
# epoch 35 - I don't think we've made much improvement but we haven't fallen back much either
# epoch 40 - Same as before. I don't think we're getting much more than this
# epoch 45 - We're definitely backtracking
# epoch 50 - Again, not terrible for not having as many brain cells as the last one.

# Notes about the new training w/ hyperperameter change:
# epoch 01 - noise like normal

# epoch 02 - I'm noticing more centralized shapes.
# epoch 11 - I'm seeing some nines and sevens
# epoch 15 - Nothing major to note, we're seeing more definition though
# epoch 20 - We're seeing some sixes and fives join the mix
# epoch 28 - I think we've lost our progress
# epoch 36 - I don't see much specific change personally
# epoch 42 - Still nothing more of note
# epoch 44 - Some of these look decent actually!
# epoch 50 - I'm kind of proud of how some of these turned out.

# Notes about the new training w/ precision change:
# epoch 01 - noise like normal

# epoch 02 - everything is dark
# epoch 03 - everything is bright
# epoch 10 - I see some 3s?
#           Also this is going much faster than expected
# epoch 20 - these aren't resembling much at this time
# epoch 30 - there's maybe some nines
# epoch 40 - These aren't looking half bad. Not yet passable, but not terrible.
# epoch 45 - I think this is the best luck we've had so far
# epoch 50 - It could do with a few more rounds of training
#           But for now it doesn't look half bad.
