"""
ConsoleController: User interface for manually solving
Anne Hoy's problems from the console.
"""


# Copyright 2014, 2017 Dustin Wehr, Danny Heap, Bogdan Simion,
# Jacqueline Smith, Dan Zingaro
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


from toah_model import TOAHModel, IllegalMoveError


def move(model, origin, dest):
    """ Apply move from origin to destination in model.

    May raise an IllegalMoveError.

    @param TOAHModel model:
        model to modify
    @param int origin:
        stool number (index from 0) of cheese to move
    @param int dest:
        stool number you want to move cheese to
    @rtype: None
    """
    if isinstance(origin, int) and isinstance(dest, int):
        model.move(origin, dest)
    else:
        raise IllegalMoveError()


def check_end(user_input):
    """ Checking the user input if it is End, it will exit
        if it is end, continue otherwise

        @param str user_input:
        @rtype: None
        """
    if user_input == 'END':
        print("game is finished")
        exit(0)


class ConsoleController:
    """ Controller for text console.
    """

    def __init__(self, number_of_cheeses, number_of_stools):
        """ Initialize a new ConsoleController self.

        @param ConsoleController self:
        @param int number_of_cheeses:
        @param int number_of_stools:
        @rtype: None
        """
        self.number_of_cheeses = number_of_cheeses
        self.number_of_stools = number_of_stools

    def play_loop(self):
        """ Play Console-based game.

        @param ConsoleController self:
        @rtype: None

        TODO:
        -Start by giving instructions about how to enter moves (which is up to
        you). Be sure to provide some way of exiting the game, and indicate
        that in the instructions.
        -Use python's built-in function input() to read a potential move from
        the user/player. You should print an error message if the input does
        not meet the specifications given in your instruction or if it denotes
        an invalid move (e.g. moving a cheese onto a smaller cheese).
        You can print error messages from this method and/or from
        ConsoleController.move; it's up to you.
        -After each valid move, use the method TOAHModel.__str__ that we've
        provided to print a representation of the current state of the game.
        """
        m = TOAHModel(stool)
        m.fill_first_stool(cheese)
        print(m)
        cond_check = True
        while cond_check:
            move_lst = []
            user_input = input('give me instruction')
            # check condition
            check_end(user_input)
            if cond_check:
                user_input = user_input.split(',')
                try:
                    move_lst.append(int(user_input[0]))
                    move_lst.append(int(user_input[1]))
                    move(m, move_lst[0], move_lst[1])
                except Exception as a:
                    print("input should be in the format int,int ", a)
                finally:
                    print(m)
                    if len(m.get_stools_lst()[-1]) == cheese:
                        print("you win")


if __name__ == '__main__':
    # You should initiate game play here. Your game should be playable by
    # running this file.
    stool = 0
    cheese = 0
    flag = True
    while flag:
        print("what is the stool")
        temp = input()
        check_end(temp)
        while not temp.isdigit():
            print("please try again,what is the stool,the input should be int")
            temp = input()
            check_end(temp)
        stool = int(temp)
        flag = False

    flag = True
    while flag:
        print("what is the cheese")
        temp = input()
        check_end(temp)
        while not temp.isdigit():
            print("please try again,what is the cheese,the input should be int")
            temp = input()
            check_end(temp)
        cheese = int(temp)
        flag = False

    c = ConsoleController(cheese, stool)
    c.play_loop()

    # Leave lines below as they are, so you will know what python_ta checks.
    # You will need consolecontroller_pyta.txt in the same folder.
    import python_ta
    python_ta.check_all(config="consolecontroller_pyta.txt")
