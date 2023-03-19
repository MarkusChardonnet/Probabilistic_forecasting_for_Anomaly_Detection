import numpy as np
import math
import copy
import pickle

class RMDF():
    def __init__(self,depth=10,ascent_rate=20,start=np.array([0,0]),end=np.array([1,0])):
        self.control_points = [[] for i in range(depth+1)]
        self.control_points_copy = [[] for i in range(depth+1)]
        self.anchor = [[] for i in range(depth+1)]
        self.depth = depth
        self.start = start
        self.end = end
        self.ascent_rate = ascent_rate

    def gen_anchor(self):
        start = self.start
        end = self.end
        self.control_points[0].append([[start[0],end[0]],start,end])
        for d in range(self.depth):
            for e in self.control_points[d]:
                start = e[1]
                end = e[2]
                l = self.__length(start,end)
                pmid = self.__mid(start,end)
                h = np.random.normal(0,l/self.ascent_rate)

                zeta = math.atan(h/(l/2))
                l2 = math.sqrt(h*h+(l/2)*(l/2))
                T = np.matrix([[math.cos(zeta),-math.sin(zeta)],[math.sin(zeta),math.cos(zeta)]])
                a = np.matrix([[pmid[0]-start[0]],[pmid[1]-start[1]]])
                b = np.matmul(T,a)*(l2/l*2)
                p = np.array([start[0]+b[0,0],start[1]+b[1,0]])

                self.control_points[d+1].append([[start[0],p[0]],start,p])
                self.control_points[d+1].append([[p[0],end[0]],p,end])

        self.anchor = self.control_points.copy()
    
    def clear_all(self):
        self.__clear(self.depth+1)
        self.gen_anchor()

    def gen(self, forking_depth, length):
        self.__rebase_anchor()
        self.__clear(forking_depth)
        self.__forking(forking_depth)
        self.__std()
        x_ = np.arange(0,1,1/length)
        # Replaced 10 by self.depth
        y = np.array([self.__expression(x,self.depth) for x in x_])
        return y
    
    def __rebase_anchor(self):
        self.control_points = self.anchor.copy()

    def __std_anchor(self):
        # Replaced 10 by self.depth
        point_list = list(map(lambda x:x[2],self.anchor[self.depth]))
        y_value_list = list(map(lambda x:x[1], point_list))
        max_y = np.max(y_value_list)
        min_y = np.min(y_value_list)
        height = max_y-min_y
        for i in range(len(self.anchor[self.depth])):
            self.anchor[self.depth][i][2][1]=self.anchor[self.depth][i][2][1]/height

    def __std(self):
        # Replaced 10 by self.depth
        point_list = list(map(lambda x:x[2],self.control_points_copy[self.depth]))
        y_value_list = list(map(lambda x:x[1], point_list))
        max_y = np.max(y_value_list)
        min_y = np.min(y_value_list)
        height = max_y-min_y
        for i in range(len(self.control_points_copy[self.depth])):
            self.control_points_copy[self.depth][i][2][1] = self.control_points_copy[self.depth][i][2][1]/height

    def __expression(self,x,depth):
        expression = self.control_points_copy[depth]
        for e in expression:
            if x>=e[0][0] and x<=e[0][1]:
                p1 = e[1]
                p2 = e[2]
                k = (p2[1]-p1[1])/(p2[0]-p1[0])
                b = p1[1]-k*p1[0]
                return k*x+b

    def __forking(self,forking_depth):
        shared_depth = self.depth - forking_depth
        for d in range(shared_depth,self.depth):
            for e in self.control_points[d]:
                start = e[1]
                end = e[2]
                l = self.__length(start,end)
                pmid = self.__mid(start,end)
                h = np.random.normal(0,l/self.ascent_rate)

                zeta = math.atan(h/(l/2))
                l2 = math.sqrt(h*h+(l/2)*(l/2))
                T = np.matrix([[math.cos(zeta),-math.sin(zeta)],[math.sin(zeta),math.cos(zeta)]])
                a = np.matrix([[pmid[0]-start[0]],[pmid[1]-start[1]]])
                b = np.matmul(T,a)*(l2/l*2)
                p = np.array([start[0]+b[0,0],start[1]+b[1,0]])

                self.control_points[d+1].append([[start[0],p[0]],start,p])
                self.control_points[d+1].append([[p[0],end[0]],p,end])
        self.control_points_copy = copy.deepcopy(self.control_points)

    def __clear(self,forking_depth):
        # clear the latest forking_depth layer.
        shared_depth = self.depth - forking_depth
        for i in range(shared_depth,self.depth):
            self.control_points[i+1]=[]

    def __length(self,p1,p2):
        # length of line (p1,p2)
        # p = [x,y]
        # L2-norm
        return np.linalg.norm(p1-p2)
    
    def __mid(self,p1,p2):
        # mid point of line(p1,p2)
        x = (p2[0]+p1[0])/2
        y = (p2[1]+p1[1])/2
        return np.array([x,y])
    
    def save_anchor(self,path):
        with open(path, 'wb') as f:
            pickle.dump(self.anchor, f)

    def load_anchor(self,path):
        with open(path, 'rb') as f:
            self.anchor = pickle.load(f)