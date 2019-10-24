"""
functions to run TOAH tours.
"""


# Copyright 2013, 2014, 2017 Gary Baumgartner, Danny Heap, Dustin Wehr,
# Bogdan Simion, Jacqueline Smith, Dan Zingaro
# Distributed under the terms of the GNU General Public License.
#
# This file is part of Assignment 1, CSC148, Winter 2017.
#
# This is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This file is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this file.  If not, see <http://www.gnu.org/licenses/>.
# Copyright 2013, 2014 Gary Baumgartner, Danny Heap, Dustin Wehr


# you may want to use time.sleep(delay_between_moves) in your
# solution for 'if __name__ == "main":'
import time
from toah_model import TOAHModel


def tour_of_four_stools(model, delay_btw_moves=0.5, animate=False):
    """Move a tower of cheeses from the first stool in model to the fourth.

    @type model: TOAHModel
        TOAHModel with tower of cheese on first stool and three empty
        stools
    @type delay_btw_moves: float
        time delay between moves if console_animate is True
    @type animate: bool
        animate the tour or not
    @rtype: None
    """
    n = model.get_number_of_cheeses()
    lst = four_stools_hanoi(n, 0, 1, 2, 3)
    for sub in lst:
        model.move(sub[0], sub[1])
        if animate:
            print(str(model))
            time.sleep(delay_btw_moves)


def three_stools_hanoi(n, first_stool, second_stool, third_stool):
    """Move the n th chesse from the first_stool to the third_stool

    @type n: int
    @type first_stool: int
        the origin position of cheeses
    @type second_stool: int
        the intermediate stool
    @type third_stool: int
        the destination cheesed have to be moved to
    @rtype: lst of touple

    """
    if n == 1:
        return [(first_stool, third_stool)]
    else:
        return three_stools_hanoi(n-1, first_stool, third_stool, second_stool) \
               + [(first_stool, third_stool)] \
               + three_stools_hanoi(n-1, second_stool, first_stool, third_stool)


def compute(n):
    """Compute how many times the function run according to differetnt given
    i.

    @type n: int
    @rtype : Number
    """
    if n == 1:
        return 1
    else:
        i = find_i(n)
        return 2 * compute(n - i) + 2 ** i - 1


def find_i(n):
    """Find i which can make the best solution

    @type n: int
    @rtype: Number
    """
    lst = []
    for i in range(1, n):
        lst.append(2 * compute(n - i) + 2 ** i - 1)
    result = min(lst)
    return lst.index(result) + 1


def four_stools_hanoi(n, first_stool, second_stool, third_stool, fourth_stool):
    """Move the n th chesse from the first_stool to the third_stool
    @type n: int
    @type first_stool: int
        the origin position of cheeses
    @type second_stool: int
        the intermediate stool
    @type third_stool: int
        the intermediate stool
    @type fourth_stool: int
        the destination cheesed have to be moved to
    @rtype: list of tuple
    """
    if n == 1:
        return [(first_stool, fourth_stool)]
    else:
        i = find_i(n)
        a = four_stools_hanoi(n - i, first_stool, third_stool, fourth_stool,
                              second_stool)
        b = three_stools_hanoi(i, first_stool, third_stool, fourth_stool)
        c = four_stools_hanoi(n - i, second_stool, third_stool, first_stool,
                              fourth_stool)
        return a + b + c


if __name__ == '__main__':
    cheeses = 5
    delay_between_moves = 0.5
    console_animate = True

    # DO NOT MODIFY THE CODE BELOW.
    four_stools = TOAHModel(4)
    four_stools.fill_first_stool(number_of_cheeses=cheeses)

    tour_of_four_stools(four_stools,
                        animate=console_animate,
                        delay_btw_moves=delay_between_moves)

    print(four_stools.number_of_moves())
    # Leave files below to see what python_ta checks.
    # File tour_pyta.txt must be in same folder
    import python_ta
    python_ta.check_all(config="tour_pyta.txt")
