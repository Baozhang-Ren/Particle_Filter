#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 07:34:08 2018

@author: baozhang, juntao
"""

###Read in Map, scan data, heading, distance,noises, and ground truth to implement particle 
###filter and sequential importance sampling
from matplotlib import pyplot
from shapely.geometry import LineString
from shapely.geometry import Polygon, Point
import pickle 
import numpy as np
import math
import map_1, map_2, map_3, map_4, map_5
import matplotlib.pyplot as plt
from functools import reduce
import random
import time

map_num = 4
MAP_DATA = [map_4.BOUND, map_4.OBSTACLES]
START = map_4.START  # the init point of turtlebot
START.append(0)
SAMPLE_NUM = 100 # number of particles
with open('trajectories/output4_1.pkl', 'rb') as pickle_file:
    PATH_DATA = pickle.load(pickle_file)[:5]  # [heading,dis,position,orientation,scan]

NOISE = [0.1, 0.1]  # [trans_noise, rotate_noise]


class SIR:
    def __init__(self, map_data, path_data, noises, start, sample_num):
        self.start = start  # init point of the turtlebot
        self.sample_num = sample_num  # number of particles
        self.map_data = map_data
        self.obs = self.get_obs()  # obstacles
        self.trans_nois = noises[0]
        self.rotat_nois = noises[1]
        self.noises = noises
        self.all_data = path_data
        self.particles = []  # particle list
        self.weights = []  # the weight for each particle
        self.exp = []  # expectation

    def check_collision(self,point):
        point = Point(point)
        if point.within(self.obs[0])==False:
            return True
        
        for obj in self.obs[1:5]:
            if point.within(obj)==True:
                if obj.distance(point)<1e-8:
                    return True
        for obj in self.obs[5:]:
            if point.within(obj)==True:   
                return True
        return False
            
        
    def get_obs(self,):

        coords = [self.map_data[0]]
        for i in range(len(self.map_data[1])):
            coords.append(self.map_data[1][i])
        # print(coords)
        obs = []
        for i in range(len(coords)):
            if i == 0:
                polygon = coords[i]
                polygon.append(polygon[0])
                shapely_poly = Polygon(polygon)
                obs.append(shapely_poly)
                for k in range(len(coords[i])):
                    
                    if k!=len(coords[i])-1:
                        line = [coords[i][k],coords[i][k+1]]
                        shapely_line = LineString(line)
                    else:
                        line = [coords[i][0],coords[i][k]]
                        shapely_line = LineString(line)

                    obs.append(shapely_line)
            else:
                polygon = coords[i]
                polygon.append(polygon[0])
                shapely_poly = Polygon(polygon)
                obs.append(shapely_poly)
        return obs

    def get_next_state(self, real_control, xt, noises):
        #compute noisy xt+1 given real control and position of xt
        collision = True
        while collision==True:
            l, theta = self.get_control(real_control, noises)
            dx = math.cos(theta)*l
            dy = math.sin(theta)*l
            x_new = [xt[0]+dx, xt[1]+dy, theta]
            #print [x_new[0][0],x_new[1][0]]
            collision = self.check_collision((x_new[0][0],x_new[1][0]))
        return x_new

    def get_control(self, real_control, noise):
        #use real_control as mean, noise as std as gaussian distribution parameters
        l = np.random.normal(real_control[0], noise[0], 1)
        theta = np.random.normal(real_control[1], noise[1], 1)
        return [l,theta]

    def get_dist(self, p1, p2):
        dist = math.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)
        return dist

    def get_scan(self, xt):

        laser_range = [-30.0/180.0*math.pi,30.0/180.0*math.pi]
        incre = 1.11/180.0*math.pi
        max_l = 100.0
        theta = xt[2]+laser_range[0]
        scan = []
        for i in range(54):
            theta = theta+incre
            dx = math.cos(theta)*max_l
            dy = math.sin(theta)*max_l
            point = [xt[0]+dx,xt[1]+dy,theta]
            line = [(xt[0], xt[1]), (point[0], point[1])]
            shapely_line = LineString(line)
            all_points = []
            all_dist = []
            for obj in self.obs[1:]:
                if shapely_line.intersects(obj):
                    inter_p = list(shapely_line.intersection(obj).coords)[0]
                    dist = self.get_dist([xt[0], xt[1]], inter_p)
                    all_points.append(all_points)
                    all_dist.append(dist)

            if len(all_dist) == 0:  # out of map
                scan = []
                for n in range(54):
                    scan.append(np.nan)
                scan = np.asarray(scan)
                return scan

            index = all_dist.index(min(all_dist))
            if min(all_dist)<10 and min(all_dist)>0.45:
                scan.append(min(all_dist)-0.40)
            else:
                scan.append(np.nan)
        scan = np.asarray(scan)
        return scan

    def remove_nan(self, scan):
        """turn nan to 10"""
        for i in range(scan.shape[0]):
            if np.isnan(scan[i]):
                scan[i] = 10
        return scan

    def first_sample(self):
        """first sample some particles with the start sate and control"""
        for i in range(self.sample_num):
            self.particles.append(self.get_next_state((self.all_data[0][1], self.all_data[0][0]), self.start, self.noises))
        self.particles = np.asarray(self.particles)
        return self.particles

    def resample(self):
        """resample"""
        resample_particles_list = []
        weights = np.asarray(self.weights)
        weights_sum = np.sum(weights)
        prob = []  # normalized probability of each particle
        for i in range(len(weights)):
            prob.append(weights[i] / weights_sum)
        eff_num = sum([x**2 for x in prob])
        eff_num = 1/eff_num
        print eff_num
        if eff_num<self.sample_num/2:
            for i in range(self.sample_num):  # resample particles
                c = random.random()
                index = 0
                for j in range(len(self.particles)):
                    if c > prob[j]:
                        c = c - prob[j]
                    else:
                        index = j
                        break
                x_mean = self.particles[index][0]
                y_mean = self.particles[index][1]
                theta_mean = self.particles[index][2]
                x = np.random.normal(x_mean, 0.01, 1)
                y = np.random.normal(y_mean, 0.01, 1)
                theta = np.random.normal(theta_mean, 0.01, 1)
                resample_particles_list.append(np.asarray([x, y, theta]))
    
            self.particles = np.asarray(resample_particles_list)


    def get_weight(self, step, particle):
        """get the scan dist of a particular particle"""
        particle_scan = self.get_scan(particle)
        real_scan = np.asarray(self.all_data[step][4])
        particle_scan = self.remove_nan(particle_scan)
        real_scan = self.remove_nan(real_scan)
        # print(len(particle_scan))
        dist = np.linalg.norm(particle_scan-real_scan, ord=1)
        if dist==0:
            weight = 1
            return weight
        weight = 1 / dist
        return weight

    def get_weights(self, step):
        for i in range(len(self.particles)):
            weight = self.get_weight(step, self.particles[i])
            self.weights.append(weight)

    def get_predict_x(self):
        #print(self.particles.shape)
        x = np.mean(self.particles[:, 0])
        y = np.mean(self.particles[:, 1])
        theta = np.mean(self.particles[:, 2])
        return [x, y, theta]

    def move(self, step):
        for i in range(len(self.particles)):
            #print('i',self.particles[i])
            next_state = self.get_next_state([self.all_data[step][1], self.all_data[step][0]], self.particles[i], self.noises)
            self.particles[i] = next_state
            #self.particles[i] = [next_state[0], next_state[1], next_state[2]]

    def get_next_particles(self, step):
        self.get_weights(step)
        self.resample()
        self.move(step)

    def main(self):
        all_dist = 0
        self.first_sample()
        for step in range(len(self.all_data)):
            self.get_next_particles(step)
            self.exp = self.get_predict_x()
            self.draw_plot(step)
            x_distance = abs(self.exp[0] - self.all_data[step][2][0])
            y_distance = abs(self.exp[1] - self.all_data[step][2][1])
            distance = x_distance + y_distance
            all_dist = all_dist+distance
        mean_error = all_dist/len(self.all_data)
        #x_accuracy = 1 - abs(x_distance/self.all_data[-1][2][0])
        #y_accuracy = 1 - abs(y_distance/self.all_data[-1][2][1])
        #accuracy = (x_accuracy + y_accuracy) / 2
        print('mean_error =', mean_error)
        #print('accuracy =', accuracy)

        # for step in range(5):
        #     self.get_next_particles(step)


    def draw_plot(self, figure_num):

        figure = plt.figure(num=figure_num, figsize=(10, 10))
        # draw particles
        for i in range(len(self.particles)):
            plt.scatter(self.particles[i][0], self.particles[i][1], color='blue', marker='o', s=1)

        # draw expectation
        exp_x = self.exp
        plt.scatter(exp_x[0], exp_x[1], color='red')

        # draw true position
        plt.scatter(self.all_data[figure_num][2][0], self.all_data[figure_num][2][1], color='black')

        # draw bound
        plt.plot(np.append(np.array(self.map_data[0])[:, 0], np.array(self.map_data[0])[0, 0]),
                 np.append(np.array(self.map_data[0])[:, 1], np.array(self.map_data[0])[0, 1]), 'black', label="bound")

        # draw obstacles

        for i in range(len(self.map_data[1])):
            plt.plot(np.append(np.array(self.map_data[1][i])[:, 0], np.array(self.map_data[1][i])[0, 0]),
                     np.append(np.array(self.map_data[1][i])[:, 1], np.array(self.map_data[1][i])[0, 1]), 'brown',
                     label="obstacles1")

        plt.savefig('plots_int/map%d'%map_num + '_' + str(SAMPLE_NUM)+'_'+str(int(self.noises[0]*100))+'/' + 'map%d_'%map_num + str(figure_num))


if __name__ == "__main__":
    start_time = time.time()
    sir = SIR(MAP_DATA, PATH_DATA, NOISE, START, SAMPLE_NUM)
    # # print('next', sir.get_next_state([sir.all_data[0][1], sir.all_data[0][0]], sir.start, sir.noises))
    # sir.first_sample()
    # sir.get_weights(0)
    # #sir.draw_plot()
    # sir.resample()
    # #sir.draw_plot()
    sir.main()
    end_time = time.time()
    print('time', end_time - start_time)



