import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=5, linewidth=120, suppress=True)
import bisect
from PIL import Image
import random

class Astar():
    def __init__(self,maze_list,Heuristic,robot_num,names):
        self.maze_list=maze_list
        self.Heuristic=Heuristic
        self.robot_num=robot_num
        # self.robot_list=robot_list
        # self.colors=list(np.random.random(size=robot_num) * 256)
        self.colors =[(255,0,0),(0,255,128),(102,0,204),(255,102,255),(255,255,51),(153,255,255),(255,153,51)]
        self.names=names

    def find_path(self, end_node,f):
        paths = [[] for _ in self.robot_list]
        for i in range(len(paths)):
            el = end_node
            while (el != np.inf):
                paths[i].append(el)
                el = f[i][int(el)]
        return paths

    def create_graph(self,world_mat):
        N = world_mat.shape[0]
        graph = []
        for i in range(N):
            for j in range(N):
                neigh = []
                if (i != N - 1):
                    if (world_mat[i + 1, j] == 0):
                        neigh.append(N * (i + 1) + j)
                if (i != 0):
                    if (world_mat[i - 1, j] == 0):
                        neigh.append(N * (i - 1) + j)
                if (j != N - 1):
                    if (world_mat[i, j + 1] == 0):
                        neigh.append(N * i + j + 1)
                if (j != 0):
                    if (world_mat[i, j - 1] == 0):
                        neigh.append(N * i + j - 1)

                graph.append(neigh)

        cost = np.ones([N * N, N * N]) * np.inf

        for i in range(N * N):
            for j in graph[i]:
                cost[i, j] = 1

        return graph, cost

    def display_result(self,world_mat, paths,action_list,index):
        N = world_mat.shape[0]
        display_mat = 1-world_mat.copy()/10
        image_list = []
        current_loc=[]
        img_back = np.repeat(display_mat[:, :, np.newaxis] * 255, 3, axis=2)
        img_back = np.array(img_back, dtype=np.uint8)
        for i,path in enumerate(paths):
            a = path.pop()
            current_loc.append(convert_to_matrixindex(a,N))
            x, y = convert_to_matrixindex(a, N)
            cv2.circle(img_back, (y, x), radius=1, color=self.colors[i])
            img = cv2.cvtColor(img_back, cv2.COLOR_BGR2RGB)
            img=cv2.resize(img,(300,300))
            im_pil = Image.fromarray(img)
            image_list.append(im_pil)
        for k in action_list:
            img_back = np.repeat(display_mat[:, :, np.newaxis] * 255, 3, axis=2)
            img_back = np.array(img_back, dtype=np.uint8)
            for i in range(len(paths)):
                if (i==k)and(len(paths[i])>0):
                    a=paths[i].pop()
                    current_loc[i]=convert_to_matrixindex(a,N)
                x, y=current_loc[i]
                cv2.circle(img_back, (y, x), radius=1, color=self.colors[i])
            img = cv2.cvtColor(img_back, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (300, 300))
            im_pil = Image.fromarray(img)
            image_list.append(im_pil)
        image_list[0].save(self.names[index],
                        save_all=True, append_images=image_list[1:], optimize=False, duration=10,
                        loop=0)

    def collision_detection(self):

        pass

    def Manhattan_Distance(self, current_robot, other_robot,n):
        x_,y_=convert_to_matrixindex(current_robot, n)
        result=0
        for i in other_robot:
            x, y = convert_to_matrixindex(i, n)
            result+=abs(x_-x)+abs(y_-y)
        return result

    def random_loc(self,maze):
        robot_list=np.array((np.where(maze==0))).transpose().tolist()
        return random.sample(robot_list,self.robot_num)

    def forward(self):
        for heuristic in self.Heuristic.keys():
            func = self.Heuristic[heuristic]['funcs']
            equa = self.Heuristic[heuristic]['equa']
            print("-------------------------------------------")
            print("By using Astar method with {} :{}:".format(heuristic,equa))
            print("-------------------------------------------")
            for i in range(3):
                self.robot_list=self.random_loc(self.maze_list[i])
                N = self.maze_list[i].shape[0]
                current_loc = [convert_to_listindex(x, y, N) for x, y in self.robot_list]
                graph, cost = self.create_graph(self.maze_list[i])
                OPEN = [[] for _ in self.robot_list]
                CLOSE = [[] for _ in self.robot_list]
                UPPER = [np.inf for _ in self.robot_list]
                d = [np.ones([N * N]) * np.inf for _ in self.robot_list]
                f = [np.ones([N * N]) * np.inf for _ in self.robot_list]
                action_list=[]
                v = [np.zeros([N * N]) for _ in self.robot_list]
                h = [func(np.zeros([N * N]),N) for _ in self.robot_list]
                V = [[] for _ in self.robot_list]
                w = [0 for _ in self.robot_list]
                for j,robot in enumerate(current_loc):
                    d[j][robot] = 0
                    OPEN[j].append(robot)
                while any(OPEN):
                    for j,robot in enumerate(current_loc):
                        if OPEN[j] != []:
                            w[j] = w[j] + 1
                            a = OPEN[j][0]
                            CLOSE[j].append(a)
                            OPEN[j].remove(a)
                            other_robot_loc=[rr for k,rr in enumerate(current_loc) if k!=j]
                            for ch in graph[a]:
                                if (ch not in CLOSE) and (cost[a, ch] + d[j][a] < d[j][ch]) and (ch not in other_robot_loc):
                                    current_loc[j]=ch
                                    d[j][ch] = cost[a, ch] + d[j][a]
                                    f[j][ch] = a
                                    v[j][ch] = d[j][ch] + h[j][ch]+ self.Manhattan_Distance(ch,other_robot_loc,N)
                                    if ch == N * N - 1:
                                        UPPER[j] = d[j][ch]
                                        continue
                                    if ch in OPEN[j]:
                                        t = OPEN[j].index(ch)
                                        V[j].pop(t)
                                        OPEN[j].remove(ch)
                                    action_list.append(j)
                                    bisect.insort_left(V[j], v[j][ch])
                                    OPEN[j].insert(V[j].index(v[j][ch]), ch)
                            if V[j] != []:
                                V[j].pop(0)
                            if ch == N * N - 1:
                                continue
                # for j, robot in enumerate(robot_list):
                print("For maze", i, ":")
                print("The length of shortest path is:")
                print(sum(UPPER))
                print("Nodes are tested:")
                print(sum(w))
                pas = self.find_path(N * N - 1,f)
                self.display_result(self.maze_list[i], pas,action_list,i)
                plt.show()

def convert_to_listindex(i, j, N):
    return N * i + j

def convert_to_matrixindex(a, N):
    i = int(a / N)
    j = int(a % N)
    return i, j

def No_heuristic(h, n):
    return h

def Manhattan_Distance(h, n):
    for i in range(n * n):
        x, y = convert_to_matrixindex(i, n)
        h[i] = 2 * n - 2 - x - y
    return h

def Euclidean_Distance(h, n):
    for i in range(n * n):
        x, y = convert_to_matrixindex(i, n)
        h[i] = 2 * n - 2 - x - y
        h[i] = np.sqrt((n-1-x)**2+(n-1-y)**2)
    return h


data_root='/home/endeleze/Documents/Networked_Robotics_Systems/data/HW1-2'
files=os.listdir(data_root)

robot_num=7
maze_list = [np.load(os.path.join(data_root,file)) for file in files if file[-3:]=='npy']
Heuristic={'Manhattan Distance':{'funcs':Manhattan_Distance,'equa':'h(i,j)=|i-goali|+|j-goalj|'}}
module=Astar(maze_list,Heuristic,robot_num,[os.path.join(data_root,i.replace('npy','gif')) for i in files])
module.forward()