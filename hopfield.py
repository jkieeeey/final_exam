#%%
import numpy as np
import random
import matplotlib.pyplot as plt

#%%
city_num = 10
np.random.seed(2)
city_loc = np.random.rand(city_num, 2)
dist = np.zeros([city_num, city_num])

def calc_point_dist(x, y):
    r = (x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2
    return np.sqrt(r)

for i in range(city_num - 1):
    illegal = np.random.randint(low = i + 1, high = city_num, size = 3)
    for j in range(i + 1, city_num):
        if j in illegal:
            dist[i, j] = 100 # 路不通
        else:
            dist[i, j] = calc_point_dist(city_loc[i, :], city_loc[j, :])
for i in range(city_num):
    for j in range(i):
        dist[i, j] = dist[j, i]

plt.scatter(city_loc[:, 0], city_loc[:, 1])
for i in range(city_num - 1):
    for j in range(i + 1, city_num):
        if 0 < dist[i,j] < 10:
            plt.plot([city_loc[i, 0], city_loc[j, 0]], [city_loc[i, 1], city_loc[j, 1]])
plt.show()

#%%
city_travelled = np.array([False] * city_num)
dist_travel = 0
best_dist_travel = 1000
best_route_travel = [] 
def find_road(route_travel):
    global dist_travel, city_travelled, best_dist_travel, best_route_travel
    if dist_travel > best_dist_travel:
        return 0
    if len(route_travel) == city_num:
        if dist_travel < best_dist_travel:
            best_dist_travel = dist_travel
            best_route_travel = route_travel[:]
        return 0
    else:
        for i in range(city_num):
            if not city_travelled[i]:
                if len(route_travel) > 0:
                    dist_travel += dist[route_travel[-1], i]
                city_travelled[i] = True
                _ = find_road(route_travel + [i])
                city_travelled[i] = False
                if len(route_travel) > 0:
                    dist_travel -= dist[route_travel[-1], i]
    return 0
_ = find_road([])                
best_path_travel = []
for i in range(len(best_route_travel) - 1):
    best_path_travel.append((best_route_travel[i], best_route_travel[i + 1]))
plt.scatter(city_loc[:, 0], city_loc[:, 1])
for k in range(len(best_path_travel)):
    i = best_path_travel[k][0]
    j = best_path_travel[k][1]
    plt.plot([city_loc[i, 0], city_loc[j, 0]], [city_loc[i, 1], city_loc[j, 1]])
plt.show()
print(best_dist_travel)

#%%
para_a = city_num ** 2
para_d = city_num / 2
para_u0 = 0.001
para_alpha = 0.0001
para_iter = 10000
energys = np.zeros(para_iter)

#%%
def calc_du(v, dist):
    row_num = np.sum(v, axis = 0) - 1
    col_num = np.sum(v, axis = 1) - 1
    sum1 = np.zeros((city_num, city_num))
    sum2 = np.zeros((city_num, city_num))
    for i in range(city_num):
        for j in range(city_num):
            sum1[i, j] = row_num[j]
    for i in range(city_num):
        for j in range(city_num):
            sum2[j, i] = col_num[j]
    c_1 = v[:, 1:city_num]
    c_0 = np.zeros((city_num, 1))
    c_0[:, 0] = v[:, 0]
    c = np.concatenate((c_1, c_0), axis=1)
    sum3 = np.dot(dist, c)
    return - para_a * (sum1 + sum2) - para_d * sum3

def calc_u(u, du):
    return u + du * para_alpha

def calc_v(u, u0):
    return 0.5 * (1 + np.tanh(u / u0))

def calc_energy(v, dist):
    sum1 = np.sum(np.power(np.sum(v, axis = 0) - 1, 2))
    sum2 = np.sum(np.power(np.sum(v, axis = 1) - 1, 2))
    idx = [i for i in range(1, city_num)]
    idx = idx + [0]
    vt = v[:, idx]
    sum3 = dist * vt
    sum3 = np.sum(np.sum(np.multiply(v, sum3)))
    e = 0.5 * (para_a * (sum1 + sum2) + para_d * sum3)
    return e

def check_path(v):
    ansv = np.zeros([city_num, city_num])
    route = []
    for i in range(city_num):
        m = np.max(v[:, i])
        for j in range(city_num):
            if abs(v[j, i] - m) < 1e-16:
                ansv[j, i] = 1
                route += [j]
                break
    return route, ansv

def calc_dist(route):
    ans = 0
    for i in range(len(route) - 1):
        ans += dist[route[i], route[i + 1]]
    return ans

#%%
best_dist = 1000
u = np.random.rand(city_num, city_num) * 2 - 1 + para_u0 * np.log(city_num - 1) / 2
v = calc_v(u,para_u0)
for n in range(para_iter):
    du = calc_du(v, dist)
    u = calc_u(u, du)
    v = calc_v(u, para_u0)
    energys[n] = calc_energy(v, dist)
    route, newv = check_path(v)
    if len(np.unique(route)) == city_num:
        new_dist = calc_dist(route)
        if new_dist < best_dist:
            best_dist = new_dist
            best_route = route[:]
            
best_path = []
for i in range(len(best_route) - 1):
    best_path.append((best_route[i], best_route[i + 1]))
plt.scatter(city_loc[:, 0], city_loc[:, 1])
for k in range(len(best_path)):
    i = best_path[k][0]
    j = best_path[k][1]
    plt.plot([city_loc[i, 0], city_loc[j, 0]], [city_loc[i, 1], city_loc[j, 1]])
plt.show()
print(best_dist)
plt.plot(energys)


    