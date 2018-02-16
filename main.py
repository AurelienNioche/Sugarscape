# -*- coding: utf-8 -*-
import numpy as np
import itertools

from matplotlib.pyplot import show, figure, rcParams, grid
from matplotlib.ticker import NullFormatter
from matplotlib.animation import FuncAnimation, writers


class Economy(object):

    def __init__(self):

        self.t = 0

        ############################
        #  About the map           #
        ############################
        self.map_size = 50
        self.map = np.zeros((self.map_size, self.map_size), dtype=[("coord", int, 2),
                                                                   ("occupation", int, 1),
                                                                   ("sugar", int, 1)])
        ##############################
        #    About sugar mountains   #
        ##############################

        self.max_sugar = 4
        self.peaks = [[14, 34], [40.5, 10]]
        # self.peaks = [[2, 3]]
        self.peak_radius = 5

        #############################
        #   About agents            #
        #############################

        self.n = 400
        self.metabolism_range = np.arange(1, 5)  # Suppose metabolism values between 1 and 4
        self.vision_range = np.arange(1, 7)  # Suppose vision between 1 and 6 squares
        self.endowment_range = np.arange(5, 26)  # Suppose endowment between 5 and 25 units
        self.positions = np.zeros((self.n, 2), dtype=int)
        self.agents = list()
        self.selected_agent = None
        self.i = self.n
        self.index = np.arange(self.n)

    def prepare_environment(self):

        ##########################
        # Preparation of the map #
        ##########################

        # Each square on the map holds coordinates
        for i, j in itertools.product(range(self.map_size), range(self.map_size)):
            self.map["coord"][i, j][0] = i
            self.map["coord"][i, j][1] = j

        # Each map square is free at beginning
        self.map["occupation"] = -1

        # Compute level of sugar for each square
        self.compute_sugar_levels()

        ###################
        # Agents creation #
        ###################

        # Attribute a random place to every agent; for this create a random order of possible places
        all_positions = []
        for x, y in itertools.product(range(self.map_size), range(self.map_size)):
            all_positions.append([x, y])
        np.random.shuffle(all_positions)

        # Create a vision parameter for each agent
        all_visions = np.random.choice(self.vision_range, size=self.n)

        # Create a metabolism parameter for each agent
        all_metabolisms = np.random.choice(self.metabolism_range, size=self.n)

        # Create an initial endowment for each agent
        all_endowments = np.random.choice(self.endowment_range, size=self.n)

        # Create agents
        self.agents = []
        for i in range(self.n):

            new_agent = Agent(self,
                              name=i,
                              coord=all_positions[i],
                              vision=all_visions[i],
                              metabolism=all_metabolisms[i],
                              endowment=all_endowments[i])

            self.agents.append(new_agent)
            self.positions[i, :] = all_positions[i]
            self.map["occupation"][all_positions[i][1], all_positions[i][0]] = i

    def compute_sugar_levels(self):

        # For each peak...
        for peak in self.peaks:

            # For every position in the map...
            for x, y in itertools.product(range(self.map_size), range(self.map_size)):

                # Compute the distance between this point and the peak
                dist_x = ((peak[0] - x)**2)
                dist_y = ((peak[1] - y)**2)
                tot_dist = (dist_x + dist_y)**(1/2.)

                # Variable that will be decremented until the 'good' level of sugar
                sugar_level = self.max_sugar

                while sugar_level > 0:

                    # Compute the distance to the peak associated to the current sugar level
                    dist_to_check = ((self.max_sugar-sugar_level+1)*self.peak_radius)

                    # If the distance of the point is inferior to the distance to the peak associated to the
                    #   current sugar level, then attribute to this position the current sugar level
                    #   (excepted if its current sugar level is higher), else decrease the current sugar level
                    if tot_dist < dist_to_check:
                        if self.map["sugar"][y, x] < sugar_level:
                            self.map["sugar"][y, x] = sugar_level
                        break

                    else:
                        sugar_level -= 1
                        continue

    def select_an_agent(self):

        # Method called by 'do what needs to be done' method:

        # Agent to be selected should be the next in the list
        self.i += 1

        # If each agent has already played...
        if self.i >= self.n:

            # Increment time
            self.t += 1

            # Randomize the indexes order for not taking agents in the same order every time
            np.random.shuffle(self.index)

            # Agent to be selected will be the first of the list
            self.i = 0

        self.selected_agent = self.agents[self.index[self.i]]

    def do_what_needs_to_be_done(self):

        # Select an agent
        self.select_an_agent()

        # Make the selected agent define his goal and move if he is still alive
        if self.selected_agent.alive == 1:
            self.selected_agent.update_goal()
            self.selected_agent.move()

        # # Update positions
        # # # self.positions[:] = np.random.choice(range(self.map_size), size=(self.n, 2))

        # # Update sugar levels if necessary
        # # # self.sugar_levels = np.random.choice(range(0, self.max_sugar+1), size=(self.map_size, self.map_size))


class Agent(object):

    def __init__(self, eco, name, coord, vision, metabolism, endowment):

        self.eco = eco

        self.name = name

        self.vision = vision
        self.metabolism = metabolism

        self.sugar_stock = endowment

        self.coord = np.zeros(shape=2, dtype=int)
        self.coord[:] = coord

        self.alive = 1

        self.goal = np.zeros(shape=2, dtype=int)

    def update_goal(self):

        # Compute x position and y position possibles depending on the vision of the agent
        possible_x = np.arange(self.coord[0] - self.vision, self.coord[0] + self.vision + 1) % self.eco.map_size
        possible_y = np.arange(self.coord[1] - self.vision, self.coord[1] + self.vision + 1) % self.eco.map_size

        # Delete current x and y (agent has to move if there are free places)
        possible_x = np.delete(possible_x, np.where(possible_x == self.coord[0]))
        possible_y = np.delete(possible_y, np.where(possible_y == self.coord[1]))

        possibilities = np.zeros(self.vision*4, dtype=[("coord", int, 2),
                                                       ("occupation", int, 1),
                                                       ("sugar", int, 1),
                                                       ("distance", int, 1)])

        i = 0
        for x, y in zip(possible_x, possible_y):
            possibilities["coord"][i] = x, self.coord[1]
            possibilities["occupation"][i] = self.eco.map["occupation"][self.coord[1], x]
            possibilities["sugar"][i] = self.eco.map["sugar"][self.coord[1], x]
            possibilities["distance"][i] = np.absolute(self.coord[0] - x)

            i += 1

            possibilities["coord"][i] = self.coord[0], y
            possibilities["occupation"][i] = self.eco.map["occupation"][y, self.coord[0]]
            possibilities["sugar"][i] = self.eco.map["sugar"][y,  self.coord[0]]
            possibilities["distance"][i] = np.absolute(self.coord[1] - y)

            i += 1

        not_occupied = possibilities[possibilities["occupation"] == -1]

        # If there is a free place, move to the place with the most sugar
        if len(not_occupied) > 0:

            max_sugar = np.max(not_occupied["sugar"])
            most_sugar = not_occupied[not_occupied["sugar"] == max_sugar]

            # If there is only one possibility, go there; otherwise, choose the nearest one
            if len(most_sugar) == 1:
                self.goal[:] = most_sugar["coord"][0, :]

            else:
                min_distance = np.min(most_sugar["distance"])
                closest = most_sugar[most_sugar["distance"] == min_distance]
                if len(closest) == 1:
                    self.goal[:] = closest["coord"][0, :]
                else:

                    idx = np.arange(np.size(closest))
                    np.random.shuffle(idx)
                    self.goal[:] = closest["coord"][idx[0], :]
        else:

            self.goal[:] = self.coord[:]

    def move(self):

        # Reduce stock sugar as a consequence of moving
        self.sugar_stock -= self.metabolism

        # Free the actual place
        self.eco.map["occupation"][self.coord[1], self.coord[0]] = -1

        # Take the new position
        self.coord[:] = self.goal[:]
        self.eco.map["occupation"][self.coord[1], self.coord[0]] = self.name
        self.eco.positions[self.name, :] = self.coord[:]

        # Collect the sugar
        self.sugar_stock += self.eco.map["sugar"][self.coord[1], self.coord[0]]

        # If the stock of agent is not sufficient to move, kill him
        if self.sugar_stock <= 0:

            self.alive = 0

            # Put for him a arbitrary position out of the map (cemetery)
            self.eco.positions[self.name, :] = -1, -1
            # Publish a death notice.
            # # print "A poor agent died (may God rest his soul)."

        # Else, he actually moves and can collect sugar
        else:
            pass


class Mode:
    save_video = 0
    new_frame_on_key_press = 1
    display = 2


class Window(object):

    def __init__(self, eco, mode):

        ##########################
        #     Global variables   #
        ##########################

        # 'Economy' object
        self.eco = eco

        # Display, create a video or test mode with new frame on key press
        self.mode = mode

        ###########################
        #  Animation parameters   #
        ###########################

        self.grid = 0

        self.t_max_for_video = 10

        ####################
        #    For Agents    #
        ####################

        # Agents positions are given directly by the economy

        # Agents colors
        self.C = "r"  # predefined colors are :
        # ["blue", "green", "red", "cyan", "magenta", "yellow", "black", "white"]

        # Agents 'sizes'
        self.S = 10**5/(self.eco.map_size**2)

        ####################
        #    For sugar     #
        ####################

        # Sugar graphical size
        self.sugar_S = 511*10**3/(self.eco.map_size**2)

        # Sugar positions
        (x, y) = np.meshgrid(np.arange(self.eco.map_size), np.arange(self.eco.map_size))

        self.sugar_X = x.reshape(self.eco.map_size**2)
        self.sugar_Y = y.reshape(self.eco.map_size**2)

        # Sugar colors
        self.sugar_C = np.zeros((self.eco.map_size**2, 4))

        # The first three columns are RBG values
        self.sugar_C[:, :2] = 1.
        # * The third column is 0 by default, so let it as it is
        # * The fourth column needs to be your alphas (defined in 'prepare first plot')

        # Sugar shapes
        self.sugar_M = 's'  # 's' is for square

        #########################
        #     Containers        #
        #########################

        # Containers for figure, animation and scatter plot
        self.fig = None
        self.animation = None
        self.sugar_scatter = None
        self.agent_scatter = None
        self.ax = None

        # For time index
        self.t = 0

    def launch_animation(self):

        self.sugar_C[:, 3] = (self.eco.map["sugar"].reshape(self.eco.map_size**2)/self.eco.max_sugar)**2

        #########################
        #   Graphical part      #
        #########################

        # No toolbar
        rcParams['toolbar'] = 'None'

        # New figure with white background
        self.fig = figure(figsize=(10, 10), facecolor='white', dpi=72)

        # New self.axis over the whole figure and a 1:1 aspect ratio
        self.fig.subplots_adjust(top=.96, bottom=.02, left=.02, right=.98)
        self.ax = self.fig.add_subplot(111)

        # Plot sugar
        self.sugar_scatter = \
            self.ax.scatter(self.sugar_X, self.sugar_Y, s=self.sugar_S,
                            c=self.sugar_C, marker='s', lw=0)

        self.agent_scatter = \
            self.ax.scatter(self.eco.positions[:, 0], self.eco.positions[:, 1], s=self.S,
                            c=self.C, lw=0)

        if self.grid == 1:

            self.ax.set_xlim(0, self.eco.map_size-0.5), self.ax.set_xticks(np.arange(-0.5, self.eco.map_size, 1))
            self.ax.xaxis.set_major_formatter(NullFormatter())
            self.ax.set_ylim(0, self.eco.map_size-0.5), self.ax.set_yticks(np.arange(-0.5, self.eco.map_size, 1))
            self.ax.yaxis.set_major_formatter(NullFormatter())
            grid()
        else:
            self.ax.set_xlim(-0.5, self.eco.map_size-0.5), self.ax.set_xticks([])
            self.ax.set_ylim(-0.5, self.eco.map_size-0.5), self.ax.set_yticks([])

        self.ax.set_aspect('equal')
        self.fig.gca().invert_yaxis()

        if self.mode == Mode.new_frame_on_key_press:
            print("Testing mode! Press a key for a new frame!")
            self.animation = self.fig.canvas.mpl_connect('key_press_event', self.run_one_time_step)

        elif self.mode == Mode.save_video:
            print("Creating video! Could need some time to complete!")

            ffmpeg_writer = writers['ffmpeg']
            metadata = dict(title='Sugarscape', artist='Matplotlib',
                            comment='Reproduce Sugarspace')
            writer = ffmpeg_writer(fps=15, metadata=metadata)

            n_frames = self.eco.n * self.t_max_for_video

            with writer.saving(self.fig, "sugarscape.mp4", n_frames):
                for i in range(n_frames):
                    self.run_one_time_step()
                    writer.grab_frame()

        else:
            print("Display mode!")
            self.animation = FuncAnimation(self.fig, self.run_one_time_step, interval=2)

        if self.mode != Mode.save_video:

            try:
                # Display
                show()

            # For avoiding an error when closing
            except AttributeError:
                pass

        else:
            pass

    def run_one_time_step(self, event=0):

        # Launch the economy job
        self.eco.do_what_needs_to_be_done()

        # Update sugar levels
        self.sugar_C[:, 3] = (self.eco.map["sugar"].reshape(self.eco.map_size**2)/self.eco.max_sugar)**2
        self.sugar_scatter.set_edgecolors(self.sugar_C)

        # Update values for agent positions
        self.agent_scatter.set_offsets(self.eco.positions)

        # Update time
        self.ax.set_title("t = {0}".format(self.eco.t))

        # Update manually if necessary
        if self.mode == Mode.new_frame_on_key_press:
            self.fig.canvas.draw()


if __name__ == "__main__":

    m = Mode.display

    e = Economy()
    e.prepare_environment()

    w = Window(e, m)
    w.launch_animation()
