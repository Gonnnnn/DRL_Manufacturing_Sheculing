import random, os

import math
import sys
from gym import error, spaces
from gym import Env

import copy

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import animation

from utils.arg_parser import common_arg_parser

tester_mean = 50
tester_std = 10

class Map:
    '''
    1 : Start
    2 : Lift
    3 : Testers
    4 : Line
    5 : Exit
    0 : Invalid
    '''
    def __init__(self):
        self.map = [
            [1, 2, 4, 4, 4, 4, 4, 4, 2, 4, 4, 4, 4, 4, 4, 2, 5],
            [0, 2, 3, 3, 3, 3, 3, 0, 2, 3, 3, 3, 3, 3, 0, 2, 0],
            [0, 2, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 2, 0],
            [0, 2, 4, 4, 4, 4, 4, 4, 2, 4, 4, 4, 4, 4, 4, 2, 0],
            [0, 2, 3, 3, 3, 3, 3, 0, 2, 3, 3, 3, 3, 3, 0, 2, 0],
            [0, 2, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 2, 0],
            [0, 2, 4, 4, 4, 4, 4, 4, 2, 4, 4, 4, 4, 4, 4, 2, 0],
            [0, 2, 3, 3, 3, 3, 3, 0, 2, 3, 3, 3, 3, 3, 0, 2, 0],
            [0, 2, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 2, 0],
            [0, 2, 4, 4, 4, 4, 4, 4, 2, 4, 4, 4, 4, 4, 4, 2, 0],
            [0, 2, 3, 3, 3, 3, 3, 0, 2, 3, 3, 3, 3, 3, 0, 2, 0],
            [0, 2, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 2, 0],
            [0, 2, 4, 4, 4, 4, 4, 4, 2, 4, 4, 4, 4, 4, 4, 2, 0],
            [0, 0, 3, 3, 3, 3, 3, 0, 0, 3, 3, 3, 3, 3, 0, 0, 0],
        ]

        self.x_limit = len(self.map[0]) - 1
        self.y_limit = len(self.map) - 1
        self.agents = {}

        self.testers = {
            "a": {
                "x": [2, 3, 4, 5, 6],
                "y": [[0, 1], [3, 4], [6, 7], [9, 10], [12, 13]]
            }, 
            "b": {
                "x": [9, 10, 11, 12, 13],
                "y": [[0, 1], [3, 4], [6, 7], [9, 10], [12, 13]]
            }, 
        }

        self.lifts = {
            "a": {
                "x": [1],
                "y": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
            },
            "b": {
                "x": [8],
                "y": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
            },
            "c": {
                "x": [15],
                "y": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
            }
        }

    def map_value(self, state):
        if state is None:
            return "OUT"
        m = self.map[state[0]][state[1]]
        if m == 0:
            return "INVALID"
        elif m == 1:
            return "START"
        elif m == 2:
            return "LIFT"
        elif m == 3:
            return "TESTERS"
        elif m == 4:
            return "LINE"
        elif m == 5:
            return "EXIT"

    def tester_status(self, tester_type):
        # 층별 테스터기 점유 상태
        counts = [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ]

        for agent_idx in self.agents:
            agent = self.agents[agent_idx]

            if agent.target is not None:
                if agent.target[0] == tester_type:
                    counts[agent.target[1]][agent.target[2]] += 1

        return counts
    


    def lift_status(self, lift_type):
        count = 0

        # 리프트 점유 상태
        for agent_idx in self.agents:
            agent = self.agents[agent_idx]

            s, _, _ = agent.location()
            if s == "LIFT":
                x = agent.state[1]

                if x in self.lifts[lift_type]["x"]:
                    count += 1

        return count

    def entrance_status(self):
        # 입구 점유 상태
        for agent_idx in self.agents:
            agent = self.agents[agent_idx]

            s, _, _ = agent.location()
            if s == "START":
                return True
        
        return False

    def is_occupied(self, state):
        # 셀 점유 상태
        for agent_idx in self.agents:
            agent = self.agents[agent_idx]

            if agent.state == state:
                return True
        
        return False

    def render(self, buffers=None, save=False, show=True, movie_name="movie_name"):
        def createBackground(ax):
            for floor, y in enumerate(self.map):
                for pos, value in enumerate(y):
                    p = [floor, pos]
                    v = self.map_value(p)

                    if v == "INVALID":
                        edgecolor=None
                        facecolor='white'
                    elif v == "START":
                        edgecolor = None
                        facecolor= 'blueviolet'
                    elif v == "LIFT":
                        # edgecolor = 'darkorange'
                        edgecolor = None
                        facecolor= 'navajowhite'
                    elif v == "LINE":
                        edgecolor = None
                        facecolor= 'powderblue'
                    elif v == "TESTERS":
                        # TODO : 테스터기별 컬러 다르게하기
                        if pos < 7:
                            edgecolor = 'steelblue'
                            facecolor= 'deepskyblue'
                        else:
                            edgecolor = 'darkseagreen'
                            facecolor= 'palegreen'
                    elif v == "EXIT":
                        edgecolor = None
                        facecolor= 'darkblue'
                    node = patches.Rectangle((pos, floor), 1, 1, fill=True, edgecolor=edgecolor, facecolor=facecolor)
                            
                    # node.set_width(0.5)
                    # node.set_height(0.5)
                    # node.set_xy([x,y])

                    ax.add_artist(node)

            return ax

        fig = plt.figure(figsize=((1 + len(buffers)) * self.x_limit / 3, self.y_limit / 2))

        axes = {}
        for i, buffer_type in enumerate(buffers):
            ax = fig.add_subplot(101+i+10*len(buffers), aspect='equal', autoscale_on=True)
            ax.title.set_text(buffer_type.upper())

            ax.set_xlim(0, self.x_limit + 1)
            ax.set_ylim(0, self.y_limit + 1)

            ax = createBackground(ax)

            axes[buffer_type] = ax

        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
        tact_time_text = ax.text(0.02, 0.90, '', transform=ax.transAxes)

        if buffers is None == 0:
            
            for agent_idx in self.agents:
                agent = self.agents[agent_idx]

                p = agent.state
                if p is None:
                    p = (-100, -100)
                node = patches.Circle((p[1] + 0.5, p[0] + 0.5), radius=0.3, fill=True, facecolor="white")
                ax.add_artist(node)
                ax.annotate(str(agent.id), xy=(p[1] + 0.5, p[0] + 0.4), fontsize=8, ha="center")

            plt.show()     
        else: 
            pals = {}
            anns = {}
            for buffer_type in buffers:
                buffer = buffers[buffer_type]
                ax = axes[buffer_type]
                # Pallet 생성
                pals[buffer_type] = []
                anns[buffer_type] = []
                for agent_idx in buffer[0]:
                    agent = buffer[0][agent_idx]

                    # p = agent.state
                    # if p is None:
                    p = (-100, -100)
                    node = patches.Circle((p[1] + 0.5, p[0] + 0.5), radius=0.3, fill=True, facecolor="white")

                    pals[buffer_type].append(node)

                    ax.add_artist(node)
                    annotation = ax.annotate(str(agent.id), xy=(p[1] + 0.5, p[0] + 0.4), fontsize=8, ha="center")

                    anns[buffer_type].append(annotation)

            def init():
                """initialize animation"""
                time_text.set_text('')
                tact_time_text.set_text('')
                pallet_nodes = []
                annotations  = []

                for buffer_type in buffers:
                    pallet_nodes += pals[buffer_type]
                    annotations  += anns[buffer_type]

                # return tuple(pallet_nodes) + (time_text,) + (tact_time_text,) + tuple(annotations)
                return tuple(pallet_nodes) + tuple(annotations)

            def animate(i):
                pallet_nodes = []
                annotations  = []

                for buffer_type in buffers:
                    buffer = buffers[buffer_type]
                    b = buffer[i]

                    for pidx, agent_idx in enumerate(b):
                        agent = b[agent_idx]
                        
                        p = agent.state
                        if p is None:
                            p = (-100, -100)
                        # node = patches.Circle((p[1] + 0.5, p[0] + 0.5), radius=0.3, fill=True, facecolor="white")

                        pals[buffer_type][pidx].center = p[1] + 0.5, p[0] + 0.5
                        anns[buffer_type][pidx].set_position((p[1] + 0.5, p[0] + 0.4))
                        anns[buffer_type][pidx].xy = (p[1] + 0.5, p[0] + 0.4)

                for buffer_type in buffers:
                    pallet_nodes += pals[buffer_type]
                    annotations  += anns[buffer_type]

                return tuple(pallet_nodes) + tuple(annotations)

            interval = 0.1 * 1000
            anim = animation.FuncAnimation(fig, animate, frames=len(buffer),
                                            interval=interval, blit=True, init_func=init)

            if save == True:
                anim.save(movie_name+'.mp4')
            if show == True:
                plt.show()

class Agent:
    def __init__(self, map, id, enter, env):
        # Start Point
        if enter == False:
            self.state = None
        else:
            self.state = (0, 0)
        self.map = map
        self.id = id
        self.target = None
        self.test_count = 0
        self.done = False
        self.test_time = 0

        self.actions = []

    def enter(self):
        # 입장시 다른 팔레트가 이미 있는지 확인
        if self.map.entrance_status() == False and self.test_count == 0:
            self.state = (0, 0)

    def exit(self):
        self.state = None
        self.target = None
        self.test_count += 1

    def move(self, action):
        """
        action: up, down, left, right
        -------------
        0 | 1 | 2| 3|
        1 |
        2 |
        return next position on board
        """

        if action == "down" or action == "d":
            next_state = (self.state[0] - 1, self.state[1])
        elif action == "up" or action == "u":
            next_state = (self.state[0] + 1, self.state[1])
        elif action == "left" or action == "l":
            next_state = (self.state[0], self.state[1] - 1)
        else:
            next_state = (self.state[0], self.state[1] + 1)
        
        # if next state is legal
        # if (next_state[0] >= 0) and (next_state[0] <= self.map.y_limit):
        #     if (next_state[1] >= 0) and (next_state[1] <= self.map.x_limit):
        #         if self.map.map_value(next_state) != "INVALID":
        #             self.state = next_state
        # 다음 위치가 valid한지 확인
        # 리프트 사용중, 충돌 여부
        moveable = True
        for lift_type in self.map.lifts:
            lift = self.map.lifts[lift_type]
            if next_state[1] in lift["x"]:
                # 이 리프트에 타려고함
                c = self.map.lift_status(lift_type)
                if c > 0:
                    # 리프트 사용중 -> 대기
                    # 그게 자기 자신이면 -> 이동
                    s, _, _ = self.location()
                    if s == "LIFT":
                        pass
                    else:
                        moveable = False

        # 충돌 여부 확인
        if self.map.is_occupied(next_state) == True:
            moveable = False

        # 검사 로직
        s_before = self.map.map_value(self.state)
        s_after  = self.map.map_value(next_state)

        if s_before == "LINE" and s_after == "TESTERS":
            # 검사기에 진입
            self.test_count += 1
            self.test_time = 0
            # self.test_target_time = random.randint(200,300) # 4 ~ 7 timesteps 만큼 랜덤으로 머무름 # Uniform
            self.test_target_time = np.random.normal(loc=tester_mean, scale=tester_std, size=1).astype(int)[0]  # Normal Distribution


        if s_before == "TESTERS" and s_after == "LINE":
            # 검사기에서 탈출
            self.test_time += 1
            if self.test_target_time > self.test_time:
                moveable = False

        if moveable == True:
            
            self.state = next_state
            self.actions.pop(0)

            if s_after == "EXIT":
                self.exit()
                self.done = True

        return self.state

    def setTarget(self, tester_type, floor):
        # 해당 층에 몇개의 테스터기가 점유되어 있는가
        counts = self.map.tester_status(tester_type)[floor]

        if sum(counts) >= 5:
            # 꽉 참
            return False

        # 경로 계산
        actions = ["r"]

        # 검사기 앞까지 (y)
        target_y = self.map.testers[tester_type]["y"][floor][0] - self.state[0]
        if target_y > 0:
            actions += ["u"] * target_y
        elif target_y < 0:
            actions += ["d"] * abs(target_y)

        # 검사기 앞까지 (x) / 오른쪽부터 채움
        t_idx = list(reversed(counts)).index(0)
        target_x = len(counts) - t_idx - 1

        actions += ["r"] * (target_x + 1)

        # 검사기 진입 후 탈출
        actions += ["u"]
        actions += ["d"]

        # Lift로 이동
        actions += ["r"] * (1 + t_idx)

        if tester_type == "b":
            # 종료로
            actions += ["r"]
            actions += ["d"] * self.map.testers[tester_type]["y"][floor][0]
            actions += ["r"]

        self.actions = actions
        self.target = [tester_type, floor, target_x]
        
        return actions

    def tester_type(self):
        if self.test_count == 0:
            return "a"
        elif self.test_count == 1:
            return "b"
        else:
            return None

    def autopilot(self, flag="min", floor=None, return_floor=False):
        if self.test_count == 0:
            tester_type = "a"
        elif self.test_count == 1:
            tester_type = "b"
        else:
            # 테스트 완료함
            return []

        '''
        flag : min or fcfs
        '''
        if flag == "min":
            # 각 층의 최소
            tester_status = self.map.tester_status(tester_type)
            floor = tester_status.index(min(tester_status))
        elif flag == "fcfs":
            # 앞부터 채움
            tester_status = self.map.tester_status(tester_type)
            for floor, testers in enumerate(tester_status):
                if sum(testers) < 5:
                    break
        elif flag == "rl":
            floor = floor

        r = self.setTarget(tester_type, floor)

        if return_floor == True:
            # 액션 빼앗기용
            return floor

        if r == False:
            # 대상이 꽉 참
            return False
        else:
            return r 

    def location(self):
        s = self.map.map_value(self.state)
        if s == "LINE" or s == "TESTERS":
            p = self.state
            x = p[1]
            y = p[0]

            for tester_type in self.map.testers:

                xs = self.map.testers[tester_type]["x"]
                ys = self.map.testers[tester_type]["y"]

                floor = None
                status = False
                if x in xs:
                    # x 범위에 들어옴
                    for f, _y in enumerate(ys):
                        if y in _y:
                            # 위치 찾음
                            status = True
                            floor = f

                return s, tester_type, floor                    
        else:
            return s, None, None

class FloorEnv(Env):
    def __init__(self, args=None, dim=2):
        self.agent_counts = args.agent_counts
        self.title = "RL"
        self.dim = dim
        self.args = args

        obs = self.reset()
        self.resetBuffer()


        self.action_space = spaces.Discrete(5)

        if self.dim == 2:
            if self.args.window_size > 1:
                obs_shape = obs.shape
            else:
                obs_shape = obs.shape
        elif self.dim == 1:
            if self.args.window_size > 1:
                obs_shape = (self.args.window_size, len(obs),)
            else:
                obs_shape = (len(obs),)

        self.observation_space = spaces.Box(low=0, high=4, shape=obs_shape)

    def reset(self):
        self.map = Map()
        self.agent_idx = 0
        self.agents = {}

        self.cursor = 0 # Agent IDX
        self.done_count = 0
        self.done = False

        self.sim_time = 0

        if self.dim == 2:
            result, _, _ = self.empty_obs("a")
            self.memory = [result] * self.args.window_size
        elif self.dim == 1:
            result, _, _ = self.empty_obs("a")
            r = self.flatten_obs(result)

            self.memory = [copy.deepcopy(r)] * self.args.window_size

        self.resetBuffer()
        
        for agent_idx in range(self.agent_counts):
            if agent_idx == 0:
                enter = True
            else:
                enter = False
            a = self.createAgent(enter=enter)

        self.saveBuffer(self.title)

        obs = self.obs(tester_type=self.agents[0].tester_type())

        return obs

    def get_action_meanings(self):
        return ["F1", "F2", "F3", "F4", "F5"] + ['NOOP']

    def createAgent(self, enter):
        a = Agent(self.map, self.agent_idx, enter=enter, env=self)
        self.agents[self.agent_idx] = a
        self.agent_idx += 1

        return a

    def resetBuffer(self):
        self.buffers = {}

    def saveBuffer(self, buffer_type="a"):
        if not buffer_type in self.buffers:
            self.buffers[buffer_type] = []

        self.map.agents = self.agents
        self.buffers[buffer_type].append(copy.deepcopy(self.agents))

        self.sim_time += 1

    def render(self, buffers=None, save=False, show=True, still=False, movie_name="movie_name"):
        self.map.agents = self.agents
        if still == True:
            self.map.render(buffers=None)
        else:
            if buffers is None:
                buffers = self.buffers
            self.map.render(buffers=buffers, save=save, show=show, movie_name=movie_name)

    def step(self, action):
        '''
        이미 Route가 Assign된 애들은 simulate하고, 액션이 필요한 애만 지정해야함.
        따라서 Return하는 State는 다음에 Action이 필요한 agent가 바라본 현 상태여야함.
        Action : 1~5층 중 어디에 넣을까!
        가만히 있는 action이 필요할까? 만약 모든 검사기가 꽉 찼으면.. Penalize를 하면 안됨
        '''
        a = self.agents[self.cursor]

        if action == 5:
            # 대기
            reward = np.count_nonzero(self.obs(tester_type=a.tester_type()) == 2) / 25
        else:
            # 대기가 아님
            routes = a.autopilot(flag='rl', floor=action)
            
            if routes == False:
                # 해당 검사기가 꽉참. Penalize!
                # print("FULL", a.id)
                reward = -1
            else:
                # print("RUN RL ACTION ID", a.id, a.state, a.target, a.test_count, self.done_count)
                a.move(a.actions[0])
                
                # Assign된 검사기의 수 리턴
                reward = np.count_nonzero(self.obs(tester_type=a.tester_type()) == 2) / 25

        if self.cursor == self.agent_counts -1:
            # 한바퀴를 다 수행하였을 때만 현화면 저장
            self.saveBuffer(self.title)

        # 대기 혹은 wrong assign 때 다른 agent에 대해 simulate 진행
        # 어차피 다음 차례에 이 agent로 돌아옴
        self.cursor += 1
        
        while True:
            # Simulation
            self.cursor = self.cursor % self.agent_counts
            a = self.agents[self.cursor]
            # 입장 처리
            if a.state == None:
                a.enter()

            # 입장이 됨
            
            if a.state is not None:
                if a.done == False:                
                    if len(a.actions) == 0:
                        # Action이 필요한 애가 선정.
                        # print("#####")
                        # print("ID", a.id, a.state, a.target)
                        # print("BREAK")
                        break
                        
                    if len(a.actions) > 0:
                        
                        a.move(a.actions[0]) 

                if a.done == True:
                    # print("DONE: ", a.id)
                    self.done_count += 1
                
                if self.done_count == self.agent_counts:
                    print("ALL DONE, SIMTIME:", self.sim_time)
                    break

            self.cursor += 1

            if self.cursor == self.agent_counts -1:
                # 한바퀴를 다 수행하였을 때만 현화면 저장
                self.saveBuffer(self.title)

        obs = self.obs(tester_type=a.tester_type()) # 현상태의 state           

        if self.done_count == self.agent_counts:
            self.done = True

        return obs, reward, self.done, {"buffers": self.buffers}

    def current_agent(self):
        return self.agents[self.cursor]

    def empty_obs(self, tester_type):
        ys = [0,1,2,3,4,5,6,7,8,9,10,11,12,13]
        if tester_type == "a":
            xs = [1,2,3,4,5,6,7]
        else: # tester_type == "b":
            xs = [8,9,10,11,12,13,14]

        result = np.zeros((len(ys), len(xs)))

        # 맵 구성
        for i, y in enumerate(ys):
            for j, x in enumerate(xs):
                map_type = self.map.map[y][x]
                if map_type == 0:
                    # 빈칸
                    result[i][j] = -1
                elif map_type == 2:
                    # Lift
                    result[i][j] = 0
                elif map_type == 3:
                    # Tester
                    result[i][j] = 0
                elif map_type == 4:
                    # Line
                    result[i][j] = 0

        return result, ys, xs

    def flatten_obs(self, result):
        r = result.flatten()
        r = np.delete(r, np.where(r < 0))

        return r

    def obs(self, tester_type):
        result, ys, xs = self.empty_obs(tester_type)

        # Agent 분포
        for agent_idx in self.agents:
            a = self.agents[agent_idx]
            if a.target is not None:
                if a.target[0] == tester_type:
                    i = 3 * a.target[1] + 1 # floor
                    j = a.target[2] + 1

                    result[i][j] = 2 # Occupied / Reserved
            if a.state is not None:
                if a.state[0] in ys and a.state[1] in xs:
                    i = ys.index(a.state[0])
                    j = xs.index(a.state[1])

                    if result[i][j] != 2:
                        result[i][j] = 1 # Agent Located
                    if a.test_count > 0:
                        result[i][j] = 2 + a.test_count / tester_mean

        if self.dim == 1:
            r = self.flatten_obs(result)
        elif self.dim == 2:
            r = result
        
        if self.args.window_size > 0:
            del self.memory[-1]
            self.memory.insert(0, r)

            return np.array(self.memory).flatten()
        else:
            return r


if __name__ == "__main__":
    arg_parser = common_arg_parser()
    args, unknown_args = arg_parser.parse_known_args(sys.argv)

    env = FloorEnv(args=args)
    
    random.seed(1234)

    def simulate(env, autopilot_flag="min"):
        env.reset()
        env.title = autopilot_flag

        while True:
            action = env.current_agent().autopilot(flag=autopilot_flag, return_floor=True)
            # print("")
            # print("ITERATION:",i)
            obs, reward, done, info = env.step(action)

            # print(reward)
            if done == True:
                print("ELAPSED SIM-TIME: ", env.sim_time, " | RL")
                break

        return env.buffers

    def rl_test(env):
        env.reset()

        while True:
            action = random.choice([0,1,2,3,4,5])
            # print("")
            # print("ITERATION:",i)
            obs, reward, done, info = env.step(action)
            # print(reward)
            if done == True:
                print("ELAPSED SIM-TIME: ", env.sim_time, " | RL")
                break

        return env.buffers


    buffers = []
    for autopilot_flag in ["fcfs"]:
        env.title = autopilot_flag
        buf = simulate(env, autopilot_flag)
        buffers.append(buf)

        # env.render(movie_name="fcfs_200", save=True)

    # buf = rl_test(env)
    # buffers.append(buf)

    buffers = dict(pair for d in buffers for pair in d.items())

    env.render(buffers=buffers,movie_name="autopilots_200", save=True, show=False)
