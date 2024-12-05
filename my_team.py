# baseline_team.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# baseline_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import util

from capture_agents import CaptureAgent
from game import Directions
from util import nearest_point
import time


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveReflexAgent', second='DefensiveReflexAgent', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = game_state.get_legal_actions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(game_state, a) for a in actions]
        # print('eval time for agent %d: %.4f' % (self.index, time.time() - start))

        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        food_left = len(self.get_food(game_state).as_list())

        if food_left <= 2:
            best_dist = 9999
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action

        return random.choice(best_actions)

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        """
        Normally, weights do not depend on the game state.  They can be either
        a counter or a dictionary.
        """
        return {'successor_score': 1.0}


class OffensiveReflexAgent(ReflexCaptureAgent):
    """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """
    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        # Get useful info about agent itself
        my_pos = game_state.get_agent_position(self.index)
        actions = game_state.get_legal_actions(self.index)

        # Get useful food info
        food_list = self.get_food(game_state).as_list()
        carried_food = game_state.get_agent_state(self.index).num_carrying
        # Define what is max food agent can carry based on how conservative the strategy should be
        max_carried_food = 3

        # Get info on ENEMIES!
        # Get positions of all enemies
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        enemy_positions = [enemy.get_position() for enemy in enemies if enemy.get_position() is not None]
        # Define a boolean to check if any enemy is a ghost and if they are scared
        enemy_is_ghost = any(not enemy.is_pacman for enemy in enemies)
        enemy_is_scared = any(enemy.scared_timer > 1 for enemy in enemies)
        # print(f"enemy is ghost: {enemy_is_ghost}")
        # print(f"enemy is scared: {enemy_is_scared}")
        # Get Manhattan distances to enemies
        distances_from_enemies = [self.get_maze_distance(my_pos, pos) for pos in enemy_positions]
        # print(f"enemy distance is {distances_from_enemies}")

        # OFFENSIVE STRATEGY OVERVIEW
        # Logic for determining the best action:
        # 
        # 1. Use evaluation function by default to maximize distance to enemy ghost, minimize distance to enemy pacman,
        # and discourage Stop and Reverse actions.
        # 2. IF enemy is not observed OR enemy is pacman OR enemy is scared THEN run a search to the closest food
        # or to starting position and take first action from generated path.
        # 2.1. IF a food bigger than `max_carried_food` is being carried, run a search to the starting position
        # 2.2. ELSE run a search to the closest food
        
        # 1. Use evaluation function by default
        # start = time.time()
        values = [self.evaluate(game_state, a) for a in actions]
        # print('eval time for agent %d: %.4f' % (self.index, time.time() - start))
        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]
        # print(f"Using eval function best action is {best_actions}\n")

        # 2. If no threating enemy nearby, run search to closest food or back home.
        if len(distances_from_enemies) == 0 or not enemy_is_ghost or enemy_is_scared:
            # print(f"Len distance from enemies is 0?: {distances_from_enemies}")
            # print(f"Enemy is pacman?: {not enemy_is_ghost}")
            # print(f"Enemy is scared?: {enemy_is_scared}")
            # 2.1. If carrying a lot of food, don't be greedy - go back home.
            if carried_food >= max_carried_food:
                # print(f"Carried food of {carried_food} exceeded max food of {max_carried_food}")
                search_goal_state = self.start
                # print(f"Run search to start coord {search_goal_state}")
                best_actions = SearchNode.breadth_first_search(my_pos, search_goal_state, game_state, self.index)
                # print(f"Best action is {best_actions}\n")
            # 2.2. Else run search to closest food.
            else:
                min_distance = 9999
                for food in food_list:
                    food_distance = self.get_maze_distance(my_pos, food)
                    if food_distance < min_distance:
                        closest_food_coord = food
                        min_distance = food_distance
                search_goal_state = closest_food_coord
                # print(f"Run search to closest food with coord {closest_food_coord}")
                best_actions = SearchNode.breadth_first_search(my_pos, search_goal_state, game_state, self.index)
                # print(f"Best action is {best_actions}\n")

        # If logic above failed, use all actions expect Stop and pick one at random
        # Better safe than sorry...
        if best_actions is None or not any(a in actions for a in best_actions):
            stop_action = 'Stop'
            if stop_action in actions:
                actions.remove(stop_action)
            # print(f"Failed strategy.\nbest_actions is {best_actions}")
            # print(f"Using random actions expect Stop best action is {actions}\n")
            return random.choice(actions)

        return random.choice(best_actions)

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        my_pos = successor.get_agent_state(self.index).get_position()

        # Ensure pacman avoids ghosts by penalizing ghost proximity
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        closest_ghost = []
        closest_pacman = []

        for enemy in enemies:
            if enemy.get_position() is not None and not enemy.is_pacman:
                distance_to_ghost = self.get_maze_distance(my_pos, enemy.get_position())
                closest_ghost.append(distance_to_ghost)
            elif enemy.get_position() is not None and enemy.is_pacman:
                distance_to_pacman = self.get_maze_distance(my_pos, enemy.get_position())
                closest_pacman.append(distance_to_pacman)                

        if len(closest_ghost) != 0:
            features['distance_to_ghost'] = min(closest_ghost)
        if len(closest_pacman) != 0:
            features['distance_to_pacman'] = min(closest_pacman)

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        return {
            'distance_to_ghost': 10,
            'distance_to_pacman': -10,
            'stop': -500,
            'reverse': -2,
            }

# class DefensiveReflexAgent(ReflexCaptureAgent):
#     def choose_action(self, game_state):
#         return 'Stop'

class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        # Get useful info about agent itself
        my_pos = game_state.get_agent_position(self.index)
        actions = game_state.get_legal_actions(self.index)

        # Get useful food info
        food_list = self.get_food(game_state).as_list()
        carried_food = game_state.get_agent_state(self.index).num_carrying
        # Define what is max food agent can carry based on how conservative the strategy should be
        max_carried_food = 3

        # Get info on ENEMIES!
        # Get positions of all enemies
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        enemy_positions = [enemy.get_position() for enemy in enemies if enemy.get_position() is not None]
        # Define a boolean to check if any enemy is a ghost and if they are scared
        enemy_is_ghost = any(not enemy.is_pacman for enemy in enemies)
        enemy_is_scared = any(enemy.scared_timer > 1 for enemy in enemies)
        # print(f"enemy is ghost: {enemy_is_ghost}")
        # print(f"enemy is scared: {enemy_is_scared}")
        # Get Manhattan distances to enemies
        distances_from_enemies = [self.get_maze_distance(my_pos, pos) for pos in enemy_positions]
        # print(f"enemy distance is {distances_from_enemies}")

        # DEFENSIVE STRATEGY OVERVIEW
        # Logic for determining the best action:
        # 
        # 1. Use evaluation function by default to encourage being a ghost and minimizing number of invader enemy pacmans,
        # and discourage Stop and Reverse actions.
        # 2. For each enemy, IF enemy is a pacman, and we can observe their location, run a search to their location and follow first action in path.

        # 1. Use evaluation function by default to encourage being a ghost and minimizing number of invader enemy pacmans.
        # start = time.time()
        values = [self.evaluate(game_state, a) for a in actions]
        # print('eval time for agent %d: %.4f' % (self.index, time.time() - start))
        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]
        print(f"Using eval function best action is {best_actions}\n")

        # 2. For each enemy, IF enemy is a pacman and location is not None, run search to enemy.
        for i in self.get_opponents(game_state):
            enemy = game_state.get_agent_state(i)
            enemy_position = enemy.get_position()
            enemy_is_pacman = enemy.is_pacman
            print(f"Enemy position: {enemy_position}")
            print(f"Enemy is pacman?: {not enemy_is_pacman}")
            if enemy_is_pacman and enemy_position is not None:
                print(f"Enemy {i} is pacman with a known location!")
                search_goal_state = enemy_position
                print(f"Run search to enemy with coord {enemy_position}")
                best_actions = SearchNode.breadth_first_search(my_pos, search_goal_state, game_state, self.index)
                print(f"Best action to enemy is {best_actions}\n")

        # If logic above failed, use all actions expect Stop and pick one at random
        # Better safe than sorry...
        if best_actions is None or not any(a in actions for a in best_actions):
            stop_action = 'Stop'
            if stop_action in actions:
                actions.remove(stop_action)
            print(f"Failed strategy.\nbest_actions is {best_actions}")
            print(f"Using random actions expect Stop best action is {actions}\n")
            return random.choice(actions)        

        return random.choice(best_actions)

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Computes whether we're on defense (1) or offense (0)
        features['on_defense'] = 1
        if my_state.is_pacman: features['on_defense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        return {'num_invaders': -1000, 'on_defense': 100, 'invader_distance': -10, 'stop': -100, 'reverse': -2}

class SearchNode:
    def __init__(self, parent, node_info):
        """
            parent: parent SearchNode.

            node_info: tuple with three elements => (coord, action, cost, game_state)

            coord: (x,y) coordinates of the node position

            action: Direction of movement required to reach node from
            parent node. Possible values are defined by class Directions from
            game.py

            cost: cost of reaching this node from the starting node.
        """

        self.__state = node_info[0]
        self.action = node_info[1]
        self.cost = node_info[2] if parent is None else node_info[2] + parent.cost
        self.game_state = node_info[3]
        self.parent = parent

    def breadth_first_search(search_initial_state, search_goal_state, game_state, self_index):
        expanded_nodes = set()
        initial_search_node = SearchNode(parent=None, node_info=(search_initial_state, None, 0, game_state))
        frontier = util.Queue()
        frontier.push(initial_search_node)

        if search_goal_state != search_initial_state:
            while not frontier.is_empty():
                node = frontier.pop()

                if node.state not in expanded_nodes:
                    expanded_nodes.add(node.state)

                    # Check if we have reached the goal state
                    if node.state == search_goal_state:
                        path = node.get_path()
                        if len(path) > 0:
                            best_actions = path[0]
                        else:
                            best_actions = path.pop(0)
                        return [best_actions]

                    # Expand the current node
                    actions = node.game_state.get_legal_actions(self_index)
                    for action in actions:
                        if action != 'Stop':
                            successor_state = node.game_state.generate_successor(self_index, action)
                            successor_coord = successor_state.get_agent_position(self_index)

                                # Ensure successors are valid and not revisited
                            if successor_coord not in expanded_nodes:
                                child_node = SearchNode(
                                    parent=node,
                                    node_info=(successor_coord, action, 1, successor_state)
                                )
                                frontier.push(child_node)

    def A_star_search(search_initial_state, search_goal_state, game_state, self_index, heuristic, agent):
        expanded_nodes = {}
        initial_search_node = SearchNode(parent=None, node_info=(search_initial_state, None, 0, game_state))
        frontier = util.PriorityQueue()
        frontier.push(initial_search_node, 0)
        if search_goal_state != search_initial_state:
            while not frontier.is_empty():
                node = frontier.pop()
                current_cost = node.cost
                current_state = node.state
                if current_state not in expanded_nodes or current_cost < expanded_nodes[current_state]:
                    expanded_nodes[current_state] = current_cost

                    # Check if we have reached the goal state
                    if current_state == search_goal_state:
                        path = node.get_path()
                        if len(path) > 0:
                            best_actions = path[0]
                        else:
                            best_actions = path.pop(0)
                        return [best_actions]

                    # Expand the current node
                    actions = node.game_state.get_legal_actions(self_index)
                    for action in actions:
                        if action != 'Stop':
                            successor_state = node.game_state.generate_successor(self_index, action)
                            successor_coord = successor_state.get_agent_position(self_index)
                            # Use formula for A⋆: f(n) = g(n) + h(n)
                            ucs_cost = 1
                            total_cost = ucs_cost + heuristic(self_index,successor_coord, game_state, agent)

                            # Ensure successors are valid and not revisited
                            if successor_coord not in expanded_nodes or total_cost < expanded_nodes[node.state]:
                                child_node = SearchNode(
                                    parent=node,
                                    node_info=(successor_coord, action, ucs_cost, successor_state)
                                )
                                frontier.push(child_node, total_cost)
    
    def null_heuristic(self, pos, game_state, agent):
        return 0
    
    def avg_heuristic(self, pos, game_state, agent):
        avg_dist = sum(round(agent.get_maze_distance(pos, food),1) for food in agent.get_food(game_state).as_list()) / len(agent.get_food(game_state).as_list())
        return round(1/avg_dist, 3)

    @property
    def state(self):
        return self.__state

    def get_path(self):
        path = []
        current_node = self
        while current_node.parent is not None:
            path.append(current_node.action)
            current_node = current_node.parent
        path.reverse()
        return path
