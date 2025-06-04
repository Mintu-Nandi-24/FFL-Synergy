import numpy as np
# import matplotlib.pyplot as plt

# Production function and derivatives
def cal_func_f(Kxy,Kxz,Kyz,x,y,z):
    fx = 1
    fy = x/(Kxy+x)
    fz = (Kyz/(Kyz+y)) * (x/(Kxz+x)) 
    
    return fx, fy, fz
    
def cal_func_fp(a_y,a_z,Kxy,Kxz,Kyz,x,y,z):
    fyxp = a_y * Kxy/(Kxy+x)**2
    fzxp = a_z * (Kyz/(Kyz+y)) * (Kxz/(Kxz+x)**2)
    fzyp = a_z * (x/(Kxz+x)) * (-Kyz/(Kyz+y)**2)
    
    return fyxp, fzxp, fzyp

# Function to calculate propensities
def calculate_propensities(a_x,b_x,a_y,b_y,a_z,b_z,Kxy,Kxz,Kyz,x,y,z):
    fx, fy, fz = cal_func_f(Kxy, Kxz, Kyz, x, y, z)
    
    psx = a_x * fx
    pdx = b_x * x 
    psy = a_y * fy 
    pdy = b_y * y 
    psz = a_z * fz 
    pdz = b_z * z           
    return [psx,pdx,psy,pdy,psz,pdz]

#Common parameters
simulation_time = 10000

zcv2_array = []
m1_array = []
n1_array = []
m2_array = []
n2_array = []
mn1_array = []
mn2_array = []
z_cross_array = []
z_path_array = []
rel_z_cross_array = []
Hz_array = []
Hz_path_array = []
Hz_cross_array = []
Ixz_array = []
I_path_array = []
I_cross_array = []

intx_array = [0.001, 0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.075, 0.1]

for it_intx in intx_array:
    intx = it_intx
    xav_param = 1/intx
    yav_param = 100
    zav_param = 100
    Kxy = 100
    Kxz = 100
    Kyz = 100
    b_x = 0.1
    b_y = 1.0
    b_z = 10.0
    
    fx, fy, fz = cal_func_f(Kxy,Kxz,Kyz,xav_param,yav_param,zav_param)
    
    a_x = b_x * xav_param / fx  
    a_y = b_y * yav_param / fy
    a_z = b_z * zav_param / fz
    
    x_ss_array = []
    y_ss_array = []   
    z_ss_array = []
    
    
    x = xav_param
    y = yav_param
    z = zav_param
    
    time = 0
    nu = 6   
    
    while time < simulation_time:
        
        propensities = calculate_propensities(a_x,b_x,a_y,b_y,a_z,b_z,Kxy,Kxz,Kyz,x,y,z)
        total_propensity = sum(propensities)

        # Calculate time until the next event
        delta_t = -np.log(np.random.random()) / total_propensity
        time += delta_t

        # Choose which event occurs
        r2 = np.random.random()
        r2a0 = r2 * total_propensity
        sum2 = 0.0
        event = -1

        for j in range(nu):
            sum2 += propensities[j]
            if sum2 >= r2a0:
                event = j
                break

        if event == 0:  
            x += 1
        elif event == 1:
            x -= 1
        elif event == 2:
            y += 1
        elif event == 3:
            y -= 1
        elif event == 4:  
            z += 1 
        else:
            z -= 1
        
        x_ss_array.append(x)
        y_ss_array.append(y)
        z_ss_array.append(z)
        
    xav = np.mean(x_ss_array)
    yav = np.mean(y_ss_array)
    zav = np.mean(z_ss_array)
    
    xzcov_matrix = np.cov(x_ss_array, z_ss_array, ddof=0) #ddof=0: population covariance ddof=1: sample covariance
    xzcov = xzcov_matrix[0,1]
    xzcv2 = xzcov / (xav*zav)
    
    xvar = np.var(x_ss_array) #xzcov_matrix[0,0]
    zvar = np.var(z_ss_array) #xzcov_matrix[1,1]
    # zvar = np.var(z_ss_array)
    xcv2 = xvar/(xav**2)
    zcv2 = zvar/(zav**2)
    zcv2_array.append(zcv2)
    
    fyxp, fzxp, fzyp = cal_func_fp(a_y, a_z, Kxy, Kxz, Kyz, xav, yav, zav)
    m1 = fzxp * xav / (b_z * zav) 
    n1 = (1/((b_x+b_y)*(b_x+b_z))) * (fyxp * fzyp / zav) #cov(xz,ind)
    mn1 = m1 * n1
    m2 = fzyp * yav / (b_z*zav)
    n2 = ((2*b_x+b_y+b_z)/((b_x+b_y)*(b_x+b_z)*(b_y+b_z))) * (xav*fyxp*fzxp/(yav*zav)) # cov(yz,cross)
    mn2 = m2 * n2
    z_cross = mn1 + mn2
    m1_array.append(m1)
    n1_array.append(n1)
    m2_array.append(m2)
    n2_array.append(n2)
    mn1_array.append(mn1)
    mn2_array.append(mn2)
    z_cross_array.append(z_cross)
    
    z_path = zcv2 - z_cross
    z_path_array.append(z_path)
    
    rel_z_cross = z_cross / z_path
    rel_z_cross_array.append(rel_z_cross)

    
np.savetxt('i1ffl-intx-etaz.dat', np.column_stack((intx_array, zcv2_array)), comments='', fmt="%.6f", delimiter="\t")
np.savetxt('i1ffl-intx-etaz_cross.dat', np.column_stack((intx_array, z_cross_array)), comments='', fmt="%.6f", delimiter="\t")
np.savetxt('i1ffl-intx-mn1-mn2.dat', np.column_stack((intx_array, mn1_array, mn2_array)), comments='', fmt="%.6f", delimiter="\t")
np.savetxt('i1ffl-intx-m1-n1-m2-n2.dat', np.column_stack((intx_array, m1_array, n1_array, m2_array, n2_array)), comments='', fmt="%.6f", delimiter="\t")
np.savetxt('i1ffl-intx-rel_etaz_cross.dat', np.column_stack((intx_array, rel_z_cross_array)), comments='', fmt="%.6f", delimiter="\t")
