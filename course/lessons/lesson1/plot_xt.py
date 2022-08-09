"""
Lesson 1: Explaining xT
=======================

This is an example lesson, which is copied from this excellent
`blog post <https://soccermatics.medium.com/explaining-expected-threat-cbc775d97935>`_ by
`David Sumpter (@soccermatics) <https://twitter.com/Soccermatics>`_. I have
dispersed some random python code inbetween to demonstrate how Sphinx gallery
can be used to generate examples.

One of the key questions for everyone interested in football — 
from coaches, through scouts to the fans — 
is how do we assess the quality of a player using data.
If they score a lot of goals they must be good and, more recently,
we have understood that finding good scoring opportunities
(having high xG) is also good. But what about all those passes,
dribbles, blocks and interceptions. How do we value them?

Some inline math :math:`s_{x,y}` for the probability of shooting at position
x y and the full formula for expected threat
from `Karun's blog <https://karun.in/blog/expected-threat.html>`_

.. math::

    \\texttt{xT}_{x,y} = (s_{x,y} \\times g_{x,y}) + (m_{x,y} \\times \\sum_{z=1}^{16} \\sum_{w=1}^{12} T_{(x,y)\\rightarrow(z,w)}\\texttt{xT}_{z,w})

It is with this in mind that
`The Athletic <https://theathletic.com/2751525/2021/08/06/introducing-expected-threat-or-xt-the-new-metric-on-the-block/>`_
have started using expected threat when talking about player and team performance.
The idea is to assign a value to every point on the football field based on the probability
that having the ball at that point will lead to a goal. One example of these probability maps is shown below.

.. image:: ../../../lessons/lesson1/twelve_xt.png

"""

import matplotlib.pyplot as plt

##############################################################################
# Markov chains
# -------------
# In order to evaluate actions we look at how an action changes the probability of scoring.
# It is this change in probability of scoring which is the expected threat (xT).
# If a player makes a pass which moves the ball from a place where it is unlikely
# for their team to score, to a place where they are more likely to score,
# then they have increased the xT in favour of their team. In general,
# the nearer you get the ball to the goal the more likely your team is to
# score (although if you look carefully passes back to the goalkeeper are also valuable).
# 
# More details of how expected threat is calculated can be found 
# in Friends of Tracking in this video.

##############################################################################
#..  youtube:: 0VAdzaid8L8
#   :width: 375
#   :height: 210

##############################################################################
# Expected threat history
# -----------------------
# Expected Threat was invented by Sarah Rudd in 2011. She didn’t call it that.
# In fact, she didn’t call it anything, but she had the mathematical insight,
# using Markov chains, on which it is based. In this video you can see her go
# through all the steps. And, on that basis she was recruited to StatDNA,
# who were very soon after bought up by Arsenal.
# The name xT was first used by Karun Singh, who reproposed it in
# the public sphere in a blog post in 2018.
#
# It is extra important that when we have a clear example of an idea from a
# female scientist in a male-dominated area, which is now used everywhere,
# that we pause to make sure everyone knows where it came from.
# There is a history of womens’ contributions being forgotten in Science.
# It would be embarrassing if we made this same error in the so-called modern
# era, especially in football.
#
# So when we hear about how Liverpool used expected goals added
# (yes, that is expected threat) in recruitment during 2018–19
# or we about how Opta and Statsbomb have there own version of expected
# threat, remember that all this came from the work of one very determined
# young woman, more than ten years ago, who went to as many sports
# analytics conferences as she could and pestered everyone she met
# until she got one of the first ever jobs in football analytics.
#
# (I wrote about Sarah Rudd in Soccermatics
# and I have booked in her in for a Friends of Tracking video during the autumn,
# so there will be a chance to hear more about her story soon)

fig, ax = plt.subplots()
plt.show()
