#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 13:18:19 2025

@author: ron.teichner
"""

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from astroquery.jplhorizons import Horizons
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import pickle
import os
import torch
from scipy.optimize import minimize
from IRAS import IRAS_train_script
from skimage import measure
from sklearn.preprocessing import PolynomialFeatures
from scipy.optimize import least_squares


def get_orbital_observations(planets, epoch, obs_epochs):
    # Initialize plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Colors for each planet
    colors = {
        'Mercury': 'gray',
        'Venus': 'orange',
        'Earth': 'blue',
        'Mars': 'red'
    }
    
    all_data = []
    elements_data = []
    true_anomaly_values = []
    # Loop through each planet
    for name, planet_id in planets.items():
        print(f'{name}')
        # Query orbital elements
        obj = Horizons(id=planet_id, location='500@10', epochs=epoch, id_type='id')
        elements = obj.elements()
    
        # Append desired parameters
        elements_data.append({
            'target': name,
            'a': elements['a'][0],       # semi-major axis (au)
            'e': elements['e'][0],       # eccentricity
            'i': elements['incl'][0],    # inclination (deg)
            'Omega': elements['Omega'][0], # longitude of ascending node (deg)
            'w': elements['w'][0],       # argument of periapsis (deg)
            'M': elements['M'][0],        # mean anomaly (deg)
            'T': 360/elements['n'][0]     # period
        })
        
        # Extract orbital parameters
        a = float(elements['a'][0])  # semi-major axis in AU
        e = float(elements['e'][0])  # eccentricity
        i = np.radians(float(elements['incl'][0]))  # inclination
        Omega = np.radians(float(elements['Omega'][0]))  # longitude of ascending node
        w = np.radians(float(elements['w'][0]))  # argument of periapsis
    
        # Generate true anomaly values
        theta = np.linspace(0, 2 * np.pi, 1000)
        r = a * (1 - e**2) / (1 + e * np.cos(theta))
    
        # Compute 3D coordinates of the orbit
        cos_O = np.cos(Omega)
        sin_O = np.sin(Omega)
        cos_i = np.cos(i)
        sin_i = np.sin(i)
        cos_wt = np.cos(w + theta)
        sin_wt = np.sin(w + theta)
    
        x = r * (cos_O * cos_wt - sin_O * sin_wt * cos_i)
        y = r * (sin_O * cos_wt + cos_O * sin_wt * cos_i)
        z = r * (sin_wt * sin_i)
    
        true_anomaly_values = true_anomaly_values + [{'target': name, 'x':x[i], 'y':y[i], 'z':z[i]} for i in range(len(x))]
    
        # Plot orbit
        ax.plot(x, y, z, label=f"{name}", color=colors[name])
    
        # Query sparse observational data
        obs = Horizons(id=planet_id, location='500@10', epochs=obs_epochs, id_type='id')
        vec = obs.vectors()
        obs_x = vec['x'].astype(float)
        obs_y = vec['y'].astype(float)
        obs_z = vec['z'].astype(float)
    
        vec_df = vec.to_pandas() #Convert Astropy Table to pandas DataFrame
        vec_df['targetname'] = name# Add planet name as a column
        all_data.append(vec_df)
    
        # Plot observations
        indices=np.random.randint(0,obs_x.shape[0],20)
        ax.scatter(obs_x[indices], obs_y[indices], obs_z[indices], color=colors[name], marker='o', s=15)#, label=f"{name} Obs")
    
    # Plot the Sun at the origin
    ax.scatter(0, 0, 0, color='yellow', label='Sun', s=100)
    
    # Set labels and title
    ax.set_xlabel("X (AU)")
    ax.set_ylabel("Y (AU)")
    ax.set_zlabel("Z (AU)")
    ax.set_zlim(-0.1, 0.1)
    #ax.set_title("3D Orbits and Observations of Inner Planets")
    ax.legend(loc='upper right', bbox_to_anchor=(0.95, 0.78))
    plt.tight_layout()
    plt.show(block=False)

    #plt.savefig('/Users/ron.teichner/Library/CloudStorage/OneDrive-Technion/Kepler/orbits.png', dpi=300)
    
    
    
    
    
    true_anomaly_values_df = pd.DataFrame(true_anomaly_values)
    
    # Convert to DataFrame for easy handling
    orbitalParams_df = pd.DataFrame(elements_data)
    
    
    # Combine all into a single DataFrame
    orbitalObs_df = pd.concat(all_data, ignore_index=True)
    orbitalObs_df.rename(columns={'targetname': 'target'}, inplace=True)
    
    # creating rotation matrix for each planet and noisy measurements


    # Apply to your DataFrame
    orbitalParams_df['rotation_matrix'] = orbitalParams_df.apply(compute_rotation_matrix, axis=1)
    orbitalParams_df['plane eq'] = orbitalParams_df.apply(plane_eq_str, axis=1)
    orbitalParams_df['orbitalPlaneNormal'] = orbitalParams_df.apply(add_orbitalPlaneNormal, axis=1)
    
    orbitalObs_df['r'] = orbitalObs_df.apply(lambda row: convert_to_r(row, orbitalObs_df, orbitalParams_df, 0), axis=1)
    orbitalObs_df['r_proj2OrbitalPlane'] = orbitalObs_df.apply(lambda row: proj_r_2OrbitalPlane(row, orbitalParams_df), axis=1)
    orbitalObs_df['r_2D'] = orbitalObs_df.apply(lambda row: transform_2_2D(row, orbitalParams_df, np.array([[1], [0], [0]])), axis=1)
    
    
    
    orbitalObs_df['v'] = orbitalObs_df.apply(lambda row: convert_to_r(row, orbitalObs_df, orbitalParams_df, 0, to_v=True), axis=1)
    orbitalObs_df['v_proj2OrbitalPlane'] = orbitalObs_df.apply(lambda row: proj_r_2OrbitalPlane(row, orbitalParams_df, to_v=True), axis=1)
    orbitalObs_df['v_2D'] = orbitalObs_df.apply(lambda row: transform_2_2D(row, orbitalParams_df, np.array([[1], [0], [0]]), to_v=True), axis=1)
    
    orbitalObs_df['L'] = orbitalObs_df.apply(lambda row: calc_L(row), axis=1)
    orbitalObs_df['L_2D'] = orbitalObs_df.apply(lambda row: calc_L(row, twoD=True), axis=1)
    
    alpha = 1e-2
    orbitalObs_df['rNoisy'] = orbitalObs_df.apply(lambda row: convert_to_r(row, orbitalObs_df, orbitalParams_df, alpha), axis=1)
    orbitalObs_df['vNoisy'] = orbitalObs_df.apply(lambda row: convert_to_r(row, orbitalObs_df, orbitalParams_df, alpha, to_v=True), axis=1)
    
    orbitalObs_df['r_2D_noisy'] = orbitalObs_df.apply(lambda row: convert_to_r(row, orbitalObs_df, orbitalParams_df, alpha, workOn2D_est=True), axis=1)
    orbitalObs_df['v_2D_noisy'] = orbitalObs_df.apply(lambda row: convert_to_r(row, orbitalObs_df, orbitalParams_df, alpha, workOn2D_est=True, to_v=True), axis=1)
    
    orbitalObs_df['L_Noisy'] = orbitalObs_df.apply(lambda row: calc_L(row, noisy=True), axis=1)
    orbitalObs_df['L_2D_noisy'] = orbitalObs_df.apply(lambda row: calc_L(row, twoD=True, noisy=True), axis=1)
    
    orbitalObs_df['r_tag'] = orbitalObs_df.apply(lambda row: rotate_r_to_r_tag(row, orbitalParams_df, False), axis=1)
    
    orbitalObs_df['r_tagNoisy'] = orbitalObs_df.apply(lambda row: rotate_r_to_r_tag(row, orbitalParams_df, True), axis=1)
    
    true_anomaly_values_df['r'] = true_anomaly_values_df.apply(lambda row: convert_to_r(row, true_anomaly_values_df, orbitalParams_df, 0), axis=1)
    true_anomaly_values_df['r_tag'] = true_anomaly_values_df.apply(lambda row: rotate_r_to_r_tag(row, orbitalParams_df, False), axis=1)
    
    #true_anomaly_values_df['v'] = true_anomaly_values_df.apply(lambda row: convert_to_r(row, true_anomaly_values_df, orbitalParams_df, 0, to_v=True), axis=1)
    
    if False:#plot velocities
        planet = 'Mercury'
        Obs = orbitalObs_df[orbitalObs_df['target'] == planet]
        plt.figure()
        plt.scatter(x=Obs['vx'], y=Obs['vy'], s=1, label=planet)
        plt.xlabel('vx')
        plt.ylabel('vy')
        plt.show(block=False)

        
    ######### many orbital params #########


    # List of target IDs or names — you can get IDs from https://ssd.jpl.nasa.gov/tools/sbdb_lookup.html
    # Here's an example list of asteroids' IDs (elliptical orbits)
    target_ids = (100000 + np.arange(500)).tolist()
    target_ids = target_ids + [199, 299, 399, 499, 599, 699, 799, 899, 999,  # Planets
     #301, 401, 402, 501, 502, 503, 504, 601, 602, 606, 801, 901,  # Moons
     2000001, 2000002, 2000004, 2000433, 2025143, 1000001, 1000002]  # Asteroids & Comets

    # Storage for results
    elements_data = []
    all_data = []
    for target in target_ids:
        try:
            print(f'{target}')
            # Set up query — `id_type='id'` for numeric IDs
            obj = Horizons(id=target, id_type='id', location='500@0', epochs='2458849.5')  # JD for 2020-01-01

            # Query orbital elements
            elements = obj.elements()

            # Append desired parameters
            elements_data.append({
                'target': target,
                'a': elements['a'][0],       # semi-major axis (au)
                'e': elements['e'][0],       # eccentricity
                'i': elements['incl'][0],    # inclination (deg)
                'Omega': elements['Omega'][0], # longitude of ascending node (deg)
                'w': elements['w'][0],       # argument of periapsis (deg)
                'M': elements['M'][0],        # mean anomaly (deg)
                'T': 360/elements['n'][0]     # period
            })
            if False:
                vec = obs.vectors()
                
            
                vec_df = vec.to_pandas() #Convert Astropy Table to pandas DataFrame
                vec_df['target'] = target# Add planet name as a column
                all_data.append(vec_df)
        except Exception as e:
            print(f"Failed to fetch data for target {target}: {e}")

    # Convert to DataFrame for easy handling
    multi_orbitalParams_df = pd.DataFrame(elements_data)
    
    
    multi_orbitalParams_df['rotation_matrix'] = multi_orbitalParams_df.apply(compute_rotation_matrix, axis=1)
    multi_orbitalParams_df['plane eq'] = multi_orbitalParams_df.apply(plane_eq_str, axis=1)
    multi_orbitalParams_df['orbitalPlaneNormal'] = multi_orbitalParams_df.apply(add_orbitalPlaneNormal, axis=1)
    
    if False:
        multi_orbitalObs_df = pd.concat(all_data, ignore_index=True)
        multi_orbitalObs_df['r'] = multi_orbitalObs_df.apply(lambda row: convert_to_r(row, multi_orbitalObs_df, multi_orbitalParams_df, 0), axis=1)
        multi_orbitalObs_df['r_proj2OrbitalPlane'] = multi_orbitalObs_df.apply(lambda row: proj_r_2OrbitalPlane(row, multi_orbitalParams_df), axis=1)
        multi_orbitalObs_df['r_2D'] = multi_orbitalObs_df.apply(lambda row: transform_2_2D(row, multi_orbitalParams_df, np.array([[1], [0], [0]])), axis=1)
        
        
        
        multi_orbitalObs_df['v'] = multi_orbitalObs_df.apply(lambda row: convert_to_r(row, multi_orbitalObs_df, multi_orbitalParams_df, 0, to_v=True), axis=1)
        multi_orbitalObs_df['v_proj2OrbitalPlane'] = multi_orbitalObs_df.apply(lambda row: proj_r_2OrbitalPlane(row, multi_orbitalParams_df, to_v=True), axis=1)
        multi_orbitalObs_df['v_2D'] = multi_orbitalObs_df.apply(lambda row: transform_2_2D(row, multi_orbitalParams_df, np.array([[1], [0], [0]]), to_v=True), axis=1)
        
        multi_orbitalObs_df['L'] = multi_orbitalObs_df.apply(lambda row: calc_L(row), axis=1)
        multi_orbitalObs_df['L_2D'] = multi_orbitalObs_df.apply(lambda row: calc_L(row, twoD=True), axis=1)
        
        alpha = 1e-3
        multi_orbitalObs_df['rNoisy'] = multi_orbitalObs_df.apply(lambda row: convert_to_r(row, multi_orbitalObs_df, multi_orbitalParams_df, alpha), axis=1)
        multi_orbitalObs_df['vNoisy'] = multi_orbitalObs_df.apply(lambda row: convert_to_r(row, multi_orbitalObs_df, multi_orbitalParams_df, alpha, to_v=True), axis=1)
        
        multi_orbitalObs_df['r_2D_noisy'] = multi_orbitalObs_df.apply(lambda row: convert_to_r(row, multi_orbitalObs_df, multi_orbitalParams_df, alpha, workOn2D_est=True), axis=1)
        multi_orbitalObs_df['v_2D_noisy'] = multi_orbitalObs_df.apply(lambda row: convert_to_r(row, multi_orbitalObs_df, multi_orbitalParams_df, alpha, workOn2D_est=True, to_v=True), axis=1)
        
        multi_orbitalObs_df['L_Noisy'] = multi_orbitalObs_df.apply(lambda row: calc_L(row, noisy=True), axis=1)
        multi_orbitalObs_df['L_2D_noisy'] = multi_orbitalObs_df.apply(lambda row: calc_L(row, twoD=True, noisy=True), axis=1)
        
        multi_orbitalObs_df['r_tag'] = multi_orbitalObs_df.apply(lambda row: rotate_r_to_r_tag(row, multi_orbitalParams_df, False), axis=1)
        
        multi_orbitalObs_df['r_tagNoisy'] = multi_orbitalObs_df.apply(lambda row: rotate_r_to_r_tag(row, multi_orbitalParams_df, True), axis=1)
    
        # Save to CSV
        #df.to_csv('orbital_elements.csv', index=False)
    else:
        multi_orbitalObs_df = None

    return orbitalObs_df, orbitalParams_df, true_anomaly_values_df, multi_orbitalParams_df, multi_orbitalObs_df


def angularMomentum2D(r,v):
    return np.array([r[0]*v[1] - r[1]*v[0]])

def calc_L(row, noisy=False, twoD=False, est=False, proj2Ellipse=False, sign=False, ellipse_params_r=None, ellipse_params_v=None):
    
    if not noisy:
        if twoD:
            if proj2Ellipse:
                if est:
                    return angularMomentum2D(row['r_2D_est_proj2estEllipse'][:,0], row['v_2D_est_proj2estEllipse'][:,0])[:,None]
                else:
                    return angularMomentum2D(row['r_2D_proj2estEllipse'][:,0], row['v_2D_proj2estEllipse'][:,0])[:,None]
            elif sign:
                if est:
                    Ar, Br, Cr, hr, kr = ellipse_params_r['A'], ellipse_params_r['B'], ellipse_params_r['C'], ellipse_params_r['h'], ellipse_params_r['k']
                    Av, Bv, Cv, hv, kv = ellipse_params_v['A'], ellipse_params_v['B'], ellipse_params_v['C'], ellipse_params_v['h'], ellipse_params_v['k']
                    
                    x, sign_y = row['r_2D_est_sign_y'][0,0], row['r_2D_est_sign_y'][1,0]
                    sign_vx, vy = row['v_2D_est_sign_x'][0,0], row['v_2D_est_sign_x'][1,0]
                    sol_y = solve_ellipse_y_minus_k(Ar, Br, Cr, hr, kr, x)
                    sol_vx = solve_ellipse_x_minus_h(Av, Bv, Cv, hv, kv, vy)
                    
                    if not(len(sol_y)==2) or not(len(sol_vx)==2):
                        return np.ones((2,1))*np.nan
                    else:
                        if sign_y == 1:
                            y = sol_y[0]
                        else:
                            y = sol_y[1]
                        if sign_vx == 1:
                            vx = sol_vx[0]
                        else:
                            vx = sol_vx[1]
                        return angularMomentum2D(np.array([x,y]), np.array([vx,vy]))[:,None]
                    
            else:
                 if est:
                     return angularMomentum2D(row['r_2D_est'][:,0], row['v_2D_est'][:,0])[:,None]
                 else:
                     return angularMomentum2D(row['r_2D'][:,0], row['v_2D'][:,0])[:,None]   
        else:
            return np.cross(row['r'][:,0], row['v'][:,0])[:,None]
    else:
        if twoD:
            if proj2Ellipse:
                if est:
                    return angularMomentum2D(row['r_2D_est_proj2estEllipse_noisy'][:,0], row['v_2D_est_proj2estEllipse_noisy'][:,0])[:,None]
                else:
                    return angularMomentum2D(row['r_2D_proj2estEllipse_noisy'][:,0], row['v_2D_proj2estEllipse_noisy'][:,0])[:,None]
            elif sign:
                if est:
                    Ar, Br, Cr, hr, kr = ellipse_params_r['A'], ellipse_params_r['B'], ellipse_params_r['C'], ellipse_params_r['h'], ellipse_params_r['k']
                    Av, Bv, Cv, hv, kv = ellipse_params_v['A'], ellipse_params_v['B'], ellipse_params_v['C'], ellipse_params_v['h'], ellipse_params_v['k']
                    
                    x, sign_y = row['r_2D_est_sign_y_noisy'][0,0], row['r_2D_est_sign_y_noisy'][1,0]
                    sign_vx, vy = row['v_2D_est_sign_x_noisy'][0,0], row['v_2D_est_sign_x_noisy'][1,0]
                    sol_y = solve_ellipse_y_minus_k(Ar, Br, Cr, hr, kr, x)
                    sol_vx = solve_ellipse_x_minus_h(Av, Bv, Cv, hv, kv, vy)
                    
                    if not(len(sol_y)==2) or not(len(sol_vx)==2):
                        return np.ones((2,1))*np.nan
                    else:
                        if sign_y == 1:
                            y = sol_y[0]
                        else:
                            y = sol_y[1]
                        if sign_vx == 1:
                            vx = sol_vx[0]
                        else:
                            vx = sol_vx[1]
                        return angularMomentum2D(np.array([x,y]), np.array([vx,vy]))[:,None]
            else:
                if est:
                    return angularMomentum2D(row['r_2D_est_noisy'][:,0], row['v_2D_est_noisy'][:,0])[:,None]
                else:
                    return angularMomentum2D(row['r_2D_noisy'][:,0], row['v_2D_noisy'][:,0])[:,None]
        else:
            return np.cross(row['rNoisy'][:,0], row['vNoisy'][:,0])[:,None]
    
    
    

def add_orbitalPlaneNormal(row):
    R = row['rotation_matrix']
    normal = np.asarray([R[0,2], R[1,2], R[2,2]])[:,None]
    return normal / np.linalg.norm(normal)
    
    
def add_est_orbitalPlaneNormal(row, planet, hat_n, columnName):
    
    if row['target'] == planet:
        return hat_n/np.linalg.norm(hat_n)
    else:
        if columnName in row.index:
            return row[columnName].to_numpy()[0]
        else: 
            return np.array([[0], [0], [0]])
            

# Define rotation matrices
def R_x(theta):
    return np.array([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta), np.cos(theta)]
    ])

def R_z(theta):
    return np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])

# Function to compute the full rotation matrix
def compute_rotation_matrix(row):
    #print(f'{row.shape}')
    Omega = np.deg2rad(row['Omega'])
    i = np.deg2rad(row['i'])
    w = np.deg2rad(row['w'])
    R = R_z(Omega) @ R_x(i) @ R_z(w)
    #print(f'{np.random.randn()};{R.shape}')
    return R

def transform_2_2D(row, orbitalParams_df, a, est=False, to_v=False):
    planet = row['target']
    OrbitParams = orbitalParams_df[orbitalParams_df['target'] == planet]
    
    
    
    
    if est:
        normal = OrbitParams['est_orbitalPlaneNormal'].to_numpy()[0]
        if to_v:
            r_proj = row['v_proj2EstOrbitalPlane']
        else:
            r_proj = row['r_proj2EstOrbitalPlane']
    else:
        normal = OrbitParams['orbitalPlaneNormal'].to_numpy()[0]
        if to_v:
            r_proj = row['v_proj2OrbitalPlane']
        else:
            r_proj = row['r_proj2OrbitalPlane']
    
    
    u = (np.cross(normal[:,0],a[:,0]) / np.linalg.norm(np.cross(normal[:,0],a[:,0])))[:,None]
    v = (np.cross(normal[:,0],u[:,0]))[:,None]
    
    alpha = (u.T@r_proj)[0,0]
    beta = (v.T@r_proj)[0,0]
    
    return np.array([[alpha], [beta]])
    
    

def convert_to_r(row, orbitalObs_df, orbitalParams_df, alpha, workOn2D_est=False, to_v=False, est=False, workOn2D_Proj=False):
    assert not(workOn2D_est and workOn2D_Proj)
    planet = row['target']
    Obs = orbitalObs_df[orbitalObs_df['target'] == planet]
    OrbitParams = orbitalParams_df[orbitalParams_df['target'] == planet]
    if alpha > 0:
        if workOn2D_est:
            if to_v:
                if est:
                    alpha_c = [Obs['v_2D_est'].to_numpy()[i][0,0] for i in range(Obs.shape[0])]
                    beta_c = [Obs['v_2D_est'].to_numpy()[i][1,0] for i in range(Obs.shape[0])]  
                    r_2D_est = row['v_2D_est']
                else:
                    alpha_c = [Obs['v_2D'].to_numpy()[i][0,0] for i in range(Obs.shape[0])]
                    beta_c = [Obs['v_2D'].to_numpy()[i][1,0] for i in range(Obs.shape[0])]  
                    r_2D_est = row['v_2D']
            else:
                if est:
                    alpha_c = [Obs['r_2D_est'].to_numpy()[i][0,0] for i in range(Obs.shape[0])]
                    beta_c = [Obs['r_2D_est'].to_numpy()[i][1,0] for i in range(Obs.shape[0])]  
                    r_2D_est = row['r_2D_est']
                else:
                    alpha_c = [Obs['r_2D'].to_numpy()[i][0,0] for i in range(Obs.shape[0])]
                    beta_c = [Obs['r_2D'].to_numpy()[i][1,0] for i in range(Obs.shape[0])]  
                    r_2D_est = row['r_2D']
            
            return np.array([[r_2D_est[0,0] + alpha*np.asarray(alpha_c).std()*np.random.randn()], [r_2D_est[1,0] + alpha*np.asarray(beta_c).std()*np.random.randn()]])
        elif workOn2D_Proj:
            if to_v:
                if est:
                    alpha_c = [Obs['v_2D_est_proj2estEllipse'].to_numpy()[i][0,0] for i in range(Obs.shape[0])]
                    beta_c = [Obs['v_2D_est_proj2estEllipse'].to_numpy()[i][1,0] for i in range(Obs.shape[0])]  
                    r_2D_est = row['v_2D_est_proj2estEllipse']
                else:
                    alpha_c = [Obs['v_2D_proj2estEllipse'].to_numpy()[i][0,0] for i in range(Obs.shape[0])]
                    beta_c = [Obs['v_2D_proj2estEllipse'].to_numpy()[i][1,0] for i in range(Obs.shape[0])]  
                    r_2D_est = row['v_2D_proj2estEllipse']
            else:
                if est:
                    alpha_c = [Obs['r_2D_est_proj2estEllipse'].to_numpy()[i][0,0] for i in range(Obs.shape[0])]
                    beta_c = [Obs['r_2D_est_proj2estEllipse'].to_numpy()[i][1,0] for i in range(Obs.shape[0])]  
                    r_2D_est = row['r_2D_est_proj2estEllipse']
                else:
                    alpha_c = [Obs['r_2D_proj2estEllipse'].to_numpy()[i][0,0] for i in range(Obs.shape[0])]
                    beta_c = [Obs['r_2D_proj2estEllipse'].to_numpy()[i][1,0] for i in range(Obs.shape[0])]  
                    r_2D_est = row['r_2D_proj2estEllipse']
            
            return np.array([[r_2D_est[0,0] + alpha*np.asarray(alpha_c).std()*np.random.randn()], [r_2D_est[1,0] + alpha*np.asarray(beta_c).std()*np.random.randn()]])
        else:
            if to_v:
                vx = [Obs['v'].to_numpy()[i][0,0] for i in range(Obs.shape[0])]
                vy = [Obs['v'].to_numpy()[i][1,0] for i in range(Obs.shape[0])]
                vz = [Obs['v'].to_numpy()[i][2,0] for i in range(Obs.shape[0])]
            
                return np.array([[row['vx'] + alpha*np.asarray(vx).std()*np.random.randn()], [row['vy'] + alpha*np.asarray(vy).std()*np.random.randn()], [row['vz'] + alpha*np.asarray(vz).std()*np.random.randn()]])
            else:
                x = [Obs['r'].to_numpy()[i][0,0] for i in range(Obs.shape[0])]
                y = [Obs['r'].to_numpy()[i][1,0] for i in range(Obs.shape[0])]
                z = [Obs['r'].to_numpy()[i][2,0] for i in range(Obs.shape[0])]
            
                return np.array([[row['x'] + alpha*np.asarray(x).std()*np.random.randn()], [row['y'] + alpha*np.asarray(y).std()*np.random.randn()], [row['z'] + alpha*np.asarray(z).std()*np.random.randn()]])
    else:       
        if workOn2D_est:
            return np.array([[r_2D_est[0,0]], [r_2D_est[1,0]]])
        else:
            if to_v:
                return np.array([[row['vx']], [row['vy']], [row['vz']]])
            else:
                return np.array([[row['x']], [row['y']], [row['z']]])

def rotate_r_to_r_tag(row, orbitalParams_df, noisyObs):
    planet = row['target']
    planetParams = orbitalParams_df[orbitalParams_df['target'] == planet]
    R = planetParams['rotation_matrix'].to_numpy()[0]
    if noisyObs:
        return R.T@row['rNoisy']
    else:
        return R.T@row['r']

def project_point_2_plane(r,n):
    return r - (n.T@r / np.power(np.linalg.norm(n), 2)) * n
    
def proj_r_2OrbitalPlane(row, orbitalParams_df, to_v=False):
    planet = row['target']
    planetParams = orbitalParams_df[orbitalParams_df['target'] == planet]
    normal = planetParams['orbitalPlaneNormal'].to_numpy()[0]
    if to_v:
        return project_point_2_plane(row['v'],normal)
    else:
        return project_point_2_plane(row['r'],normal)
    
def proj_r_2EstOrbitalPlane(row, orbitalParams_df, to_v=False):
    planet = row['target']
    planetParams = orbitalParams_df[orbitalParams_df['target'] == planet]
    hat_n = planetParams['est_orbitalPlaneNormal'].to_numpy()[0]
    if to_v:
        return project_point_2_plane(row['v'],hat_n)
    else:
        return project_point_2_plane(row['r'],hat_n)
    
    

def poly_eq_str(intercept, coefficients, feature_names):
    # Construct the polynomial equation as a string
    terms = [f"{intercept:.3f}"]
    for coef, name in zip(coefficients[1:], feature_names[1:]):
        terms.append(f"{coef:.3f}*{name}")
    
    polynomial_equation = " + ".join(terms)
    return polynomial_equation
    
def plane_eq_str(row):
    R = row['rotation_matrix']
    R_t = R.T
    coefficients = [0, R_t[2,0], R_t[2,1], R_t[2,2]]
    feature_names = ['0','x','y','z']
    intercept = 0
    
    return poly_eq_str(intercept, coefficients, feature_names)

def different_plots(planets, true_anomaly_values_df, orbitalObs_df):
    
    # Colors for each planet
    colors = {
        'Mercury': 'gray',
        'Venus': 'orange',
        'Earth': 'blue',
        'Mars': 'red'
    }
    
    fig = plt.figure(figsize=(20, 16))
    ax, bx = fig.add_subplot(321), fig.add_subplot(322)
    cx, dx = fig.add_subplot(323), fig.add_subplot(324)
    ex, fx = fig.add_subplot(325), fig.add_subplot(326)
    for planet in planets:
        model = true_anomaly_values_df[true_anomaly_values_df['target'] == planet]
        Obs = orbitalObs_df[orbitalObs_df['target'] == planet]
    
        x_tag_model = [model['r_tag'].to_numpy()[i][0,0] for i in range(model.shape[0])]
        y_tag_model = [model['r_tag'].to_numpy()[i][1,0] for i in range(model.shape[0])]
        z_tag_model = [model['r_tag'].to_numpy()[i][2,0] for i in range(model.shape[0])]
    
        x_tag_noisyObs = [Obs['r_tagNoisy'].to_numpy()[i][0,0] for i in range(Obs.shape[0])]
        y_tag_noisyObs = [Obs['r_tagNoisy'].to_numpy()[i][1,0] for i in range(Obs.shape[0])]
        z_tag_noisyObs = [Obs['r_tagNoisy'].to_numpy()[i][2,0] for i in range(Obs.shape[0])]
    
        x_model = [model['r'].to_numpy()[i][0,0] for i in range(model.shape[0])]
        y_model = [model['r'].to_numpy()[i][1,0] for i in range(model.shape[0])]
        z_model = [model['r'].to_numpy()[i][2,0] for i in range(model.shape[0])]
    
        x_noisyObs = [Obs['rNoisy'].to_numpy()[i][0,0] for i in range(Obs.shape[0])]
        y_noisyObs = [Obs['rNoisy'].to_numpy()[i][1,0] for i in range(Obs.shape[0])]
        z_noisyObs = [Obs['rNoisy'].to_numpy()[i][2,0] for i in range(Obs.shape[0])]
    
        # Plot orbit
        ax.plot(x_tag_model, y_tag_model, label=f"{planet}", color=colors[planet])
        ax.scatter(x_tag_noisyObs, y_tag_noisyObs, color=colors[planet], marker='o', s=15, label=f"{planet} Noisy Obs")
    
        bx.plot(x_model, y_model, label=f"{planet}", color=colors[planet])
        bx.scatter(x_noisyObs, y_noisyObs, color=colors[planet], marker='o', s=15, label=f"{planet} Noisy Obs")
        
        # Plot orbit
        cx.plot(x_tag_model, z_tag_model, label=f"{planet}", color=colors[planet])
        cx.scatter(x_tag_noisyObs, z_tag_noisyObs, color=colors[planet], marker='o', s=15, label=f"{planet} Noisy Obs")
    
        dx.plot(x_model, z_model, label=f"{planet}", color=colors[planet])
        dx.scatter(x_noisyObs, z_noisyObs, color=colors[planet], marker='o', s=15, label=f"{planet} Noisy Obs")
    
        # Plot orbit
        ex.plot(x_tag_model, z_tag_model, label=f"{planet}", color=colors[planet])
        ex.scatter(x_tag_noisyObs, z_tag_noisyObs, color=colors[planet], marker='o', s=15, label=f"{planet} Noisy Obs")
    
        fx.plot(y_model, z_model, label=f"{planet}", color=colors[planet])
        fx.scatter(y_noisyObs, z_noisyObs, color=colors[planet], marker='o', s=15, label=f"{planet} Noisy Obs")
    
    # Plot the Sun at the origin
    ax.scatter(0, 0, color='yellow', label='Sun', s=100)
    bx.scatter(0, 0, color='yellow', label='Sun', s=100)
    cx.scatter(0, 0, color='yellow', label='Sun', s=100)
    dx.scatter(0, 0, color='yellow', label='Sun', s=100)
    ex.scatter(0, 0, color='yellow', label='Sun', s=100)
    fx.scatter(0, 0, color='yellow', label='Sun', s=100)
    
    # Set labels and title
    ax.set_xlabel(r"$x_t$ (AU)")
    ax.set_ylabel(r"$y_t$ (AU)")
    #ax.set_title("3D Orbits and Observations of Inner Planets")
    ax.legend(loc='upper right', bbox_to_anchor=(0.95, 0.78))
    ax.grid()
    ax.set_aspect('equal', adjustable='box')
    
    
    bx.set_xlabel(r"$x$ (AU)")
    bx.set_ylabel(r"$y$ (AU)")
    #ax.set_title("3D Orbits and Observations of Inner Planets")
    bx.legend(loc='upper right', bbox_to_anchor=(0.95, 0.78))
    bx.grid()
    bx.set_aspect('equal', adjustable='box')
    
    
    cx.set_xlabel(r"$x_t$ (AU)")
    cx.set_ylabel(r"$z_t$ (AU)")
    #ax.set_title("3D Orbits and Observations of Inner Planets")
    cx.legend(loc='upper right', bbox_to_anchor=(0.95, 0.78))
    cx.grid()
    #cx.set_aspect('equal', adjustable='box')
    
    dx.set_xlabel(r"$x$ (AU)")
    dx.set_ylabel(r"$z$ (AU)")
    #ax.set_title("3D Orbits and Observations of Inner Planets")
    dx.legend(loc='upper right', bbox_to_anchor=(0.95, 0.78))
    dx.grid()
    #dx.set_aspect('equal', adjustable='box')
    
    
    ex.set_xlabel(r"$y_t$ (AU)")
    ex.set_ylabel(r"$z_t$ (AU)")
    #ax.set_title("3D Orbits and Observations of Inner Planets")
    ex.legend(loc='upper right', bbox_to_anchor=(0.95, 0.78))
    ex.grid()
    #ex.set_aspect('equal', adjustable='box')
    
    
    fx.set_xlabel(r"$y$ (AU)")
    fx.set_ylabel(r"$z$ (AU)")
    #ax.set_title("3D Orbits and Observations of Inner Planets")
    fx.legend(loc='upper right', bbox_to_anchor=(0.95, 0.78))
    fx.grid()
    #fx.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    plt.show(block=False)

    


def project_point_to_ellipse(row, ellipse_params, to_v=False):
    """
    Projects a point (x0, y0) onto the ellipse defined by:
    A(x - h)^2 + B(x - h)(y - k) + C(y - k)^2 = 1

    Returns the closest point (x, y) on the ellipse.
    """
    A, B, C, h, k = ellipse_params['A'], ellipse_params['B'], ellipse_params['C'], ellipse_params['h'], ellipse_params['k']
    if to_v:
        x0, y0 = row['v_2D_est'][0,0], row['v_2D_est'][1,0]
    else:
        x0, y0 = row['r_2D_est'][0,0], row['r_2D_est'][1,0]
    if np.isnan(x0) or np.isnan(y0):
        return np.ones((2,1))*np.nan
    def ellipse_constraint(p):
        x, y = p
        return A * (x - h)**2 + B * (x - h)*(y - k) + C * (y - k)**2 - 1

    def objective(p):
        x, y = p
        return (x - x0)**2 + (y - y0)**2

    initial_guess = np.array([x0, y0])
    constraint = {'type': 'eq', 'fun': ellipse_constraint}

    #result = minimize(objective, initial_guess, constraints=[constraint])
    
    result = minimize(
            objective,
            initial_guess,
            constraints=[constraint],
            method='SLSQP',  # or 'trust-constr'
            options={'maxiter': 1000, 'ftol': 1e-9}
        )

    
    if result.success:
        x, y = result.x[0], result.x[1] 
        #print(f'init=({str(round(x0,4))},{str(round(y0,4))}), final = ({str(round(x,4))},{str(round(y,4))}); init = {str(round(ellipse_constraint([x0,y0]),4))}, final = {str(round(ellipse_constraint([x,y]),4))}, ratio = {str(round(ellipse_constraint([x0,y0])/ellipse_constraint([x,y]),4))}')
        return result.x[:,None]
    else:
        #print(f"Optimization failed on point ({str(round(x0,4))},{str(round(y0,4))}) with init = {str(round(ellipse_constraint([x0,y0]),4))}: ")
        if to_v:
            return project_point_to_ellipse({'v_2D_est': row['v_2D_est'] + 1e-5*np.random.randn(2,1)}, ellipse_params, to_v=True)
        else:
            return project_point_to_ellipse({'r_2D_est': row['r_2D_est'] + 1e-5*np.random.randn(2,1)}, ellipse_params)
        #raise RuntimeError(f"Optimization failed on point ({str(round(x0,4))},{str(round(y0,4))}) with init = {str(round(ellipse_constraint([x0,y0]),4))}: " + result.message)


def plot_est2D_ellipse(IRAS_runOnCoordinatesResultsDict, planet, orbitalParams_df, orbitalObs_df, to_v=False, title=''):
    colors = {
        'Mercury': 'gray',
        'Venus': 'orange',
        'Earth': 'blue',
        'Mars': 'red'
    }
    #Mercury2D_v
    if to_v:
        ellipse_params = IRAS_runOnCoordinatesResultsDict[planet + '2D_v']['implicitPolyDictList'][0]['ellipse_fit']['ellipse_params']
        focci = IRAS_runOnCoordinatesResultsDict[planet + '2D_v']['implicitPolyDictList'][0]['ellipse_fit']['focci']
        axes_eccentricity = IRAS_runOnCoordinatesResultsDict[planet + '2D_v']['implicitPolyDictList'][0]['ellipse_fit']['axes_eccentricity']
        corr = np.abs(IRAS_runOnCoordinatesResultsDict[planet + '2D_v']['implicitPolyDictList'][0]['ellipse_fit']['corr'])
    else:
        ellipse_params = IRAS_runOnCoordinatesResultsDict[planet + '2D']['implicitPolyDictList'][0]['ellipse_fit']['ellipse_params']
        focci = IRAS_runOnCoordinatesResultsDict[planet + '2D']['implicitPolyDictList'][0]['ellipse_fit']['focci']
        axes_eccentricity = IRAS_runOnCoordinatesResultsDict[planet + '2D']['implicitPolyDictList'][0]['ellipse_fit']['axes_eccentricity']
        corr = np.abs(IRAS_runOnCoordinatesResultsDict[planet + '2D']['implicitPolyDictList'][0]['ellipse_fit']['corr'])
    
    print(f'Pearson correlation between g() and estimated ellipse is {str(round(corr,3))}')
    
    proj_sun_on_plane = proj_r_2EstOrbitalPlane(pd.Series({'target': planet, 'r': np.array([[0], [0], [0]])}), orbitalParams_df)
    sun_coordinates_on_plane = transform_2_2D(pd.Series({'target': planet, 'r_proj2OrbitalPlane': proj_sun_on_plane}), orbitalParams_df, np.array([[1], [0], [0]]))
    
    OrbitParams = orbitalParams_df[orbitalParams_df['target'] == planet]
    
    A, B, C, h, k = ellipse_params['A'], ellipse_params['B'], ellipse_params['C'], ellipse_params['h'], ellipse_params['k']
    
    # Create a grid of x and y values
    if not to_v:
        x_vals = np.linspace(h - 5, h + 5, 400)
        y_vals = np.linspace(k - 5, k + 5, 400)
    else:
        x_vals = np.linspace(h - 0.05, h + 0.05, 400)
        y_vals = np.linspace(k - 0.05, k + 0.05, 400)
    X, Y = np.meshgrid(x_vals, y_vals)
    
    # Compute the left-hand side of the ellipse equation
    Z = A * (X - h)**2 + B * (X - h) * (Y - k) + C * (Y - k)**2
    
    # Plot the contour where the equation equals 1
    plt.contour(X, Y, Z, levels=[1], colors='blue', label='fitted ellipse for IRAS g()')
    plt.scatter(sun_coordinates_on_plane[0,0], sun_coordinates_on_plane[1,0], color='yellow', label='Sun', s=100)
    plt.scatter(x=[focci[0][0], focci[1][0]], y=[focci[0][1], focci[1][1]], marker='+', s=100, color='blue', label='focci')
    #plt.title("Ellipse: A(x−h)^2 + B(x−h)(y−k) + C(y−k)^2 = 1")
    if not to_v:
        plt.text(-0.3, 0.2, r'$\hat{a} = $' + f'{str(round(axes_eccentricity["a"],3))}   ', fontsize=14)# + r'$a = $' + f'{str(round(OrbitParams["a"][0], 3))}', fontsize=12)
        plt.text(-0.3, 0.1, r'$\hat{e} = $' + f'{str(round(axes_eccentricity["e"],3))}   ', fontsize=14)# + r'$e = $' + f'{str(round(OrbitParams["e"][0], 3))}', fontsize=12)
    #plt.title("IRAS-estimated ellipse")
    plt.xlabel(r"$x_{\hat{n}}$")
    plt.ylabel(r"$y_{\hat{n}}$")
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True)
    
    Obs = orbitalObs_df[orbitalObs_df['target'] == 'Mercury']
    if to_v:
        alpha = np.asarray([Obs['v_2D'].to_numpy()[i][0,0] for i in range(Obs.shape[0])])
        beta = np.asarray([Obs['v_2D'].to_numpy()[i][1,0] for i in range(Obs.shape[0])])
        alpha_hat = np.asarray([Obs['v_2D_est'].to_numpy()[i][0,0] for i in range(Obs.shape[0])])
        beta_hat = np.asarray([Obs['v_2D_est'].to_numpy()[i][1,0] for i in range(Obs.shape[0])])
        if False:
            alpha_hat_proj = np.asarray([Obs['v_2D_est_proj2estEllipse'].to_numpy()[i][0,0] for i in range(Obs.shape[0])])
            beta_hat_proj = np.asarray([Obs['v_2D_est_proj2estEllipse'].to_numpy()[i][1,0] for i in range(Obs.shape[0])])
    else:
        alpha = np.asarray([Obs['r_2D'].to_numpy()[i][0,0] for i in range(Obs.shape[0])])
        beta = np.asarray([Obs['r_2D'].to_numpy()[i][1,0] for i in range(Obs.shape[0])])
        alpha_hat = np.asarray([Obs['r_2D_est'].to_numpy()[i][0,0] for i in range(Obs.shape[0])])
        beta_hat = np.asarray([Obs['r_2D_est'].to_numpy()[i][1,0] for i in range(Obs.shape[0])])
        if False:
            alpha_hat_proj = np.asarray([Obs['r_2D_est_proj2estEllipse'].to_numpy()[i][0,0] for i in range(Obs.shape[0])])
            beta_hat_proj = np.asarray([Obs['r_2D_est_proj2estEllipse'].to_numpy()[i][1,0] for i in range(Obs.shape[0])])
        
    indices = np.random.permutation(np.arange(len(alpha_hat)))[:30]
    plt.scatter(x=alpha_hat[indices], y=beta_hat[indices], marker='x', s=80, color=colors[planet], label=planet+' coordinates projected\nto IRAS estimated orbital plane')
    
    #plt.legend(loc='lower left')
    
    
    # Create proxy artists for legend
    legend_elements = [
    Line2D([0], [0], color='blue', lw=2, label='fitted ellipse for IRAS '+title),
    Line2D([0], [0], marker='o', color='yellow', linestyle='None', markersize=10, label='Sun'),
    Line2D([0], [0], marker='+', color='blue', linestyle='None', markersize=10, label='focci'),
    Line2D([0], [0], marker='x', color=colors[planet], linestyle='None', markersize=10, label= r'$r_{\mathrm{2D}}$: ' + planet + ' coordinates projected\nto IRAS estimated orbital plane')
    ]

    plt.legend(handles=legend_elements, loc='lower left')

    if not to_v:
        plt.xlim([-0.5, 0.5])
        plt.ylim([-0.5, 0.5])
        plt.show(block=False)

        #plt.savefig('/Users/ron.teichner/Library/CloudStorage/OneDrive-Technion/Kepler/'+planet+'2D_corr_'+f'{str(round(corr,3))}'+'_.png', dpi=300)
    else:
        plt.show(block=False)

        #plt.savefig('/Users/ron.teichner/Library/CloudStorage/OneDrive-Technion/Kepler/'+planet+'_v_2D_corr_'+f'{str(round(corr,3))}'+'_.png', dpi=300)
    
    if False:
        r_2D_est_deviation = []
        for x,y in zip(alpha_hat, beta_hat):
            r_2D_est_deviation.append(ellipse_model([A,B,C,h,k], x, y))
            
        
        
        r_2D_est_proj_deviation = []
        for x,y in zip(alpha_hat_proj, beta_hat_proj):
            r_2D_est_proj_deviation.append(ellipse_model([A,B,C,h,k], x, y))
        
        plt.figure()
        if to_v:
            #plt.plot(r_2D_est_deviation, label='v_2D_est_deviation')
            plt.plot(r_2D_est_proj_deviation, label='v_2D_est_proj_deviation')
        else:
            #plt.plot(r_2D_est_deviation, label='r_2D_est_deviation')
            plt.plot(r_2D_est_proj_deviation, label='r_2D_est_proj_deviation')
        plt.legend()
        plt.show(block=False)

    
    
def perifocal_trans_plot(planets, orbitalObs_df, orbitalParams_df):
    for planet in planets:
    
        #Obs = true_anomaly_values_df[true_anomaly_values_df['target'] == planet]
        Obs = orbitalObs_df[orbitalObs_df['target'] == planet]
        OrbitParams = orbitalParams_df[orbitalParams_df['target'] == planet]
        
        x = [Obs['r'].to_numpy()[i][0,0] for i in range(Obs.shape[0])]
        y = [Obs['r'].to_numpy()[i][1,0] for i in range(Obs.shape[0])]
        z = [Obs['r'].to_numpy()[i][2,0] for i in range(Obs.shape[0])]
        
        x_tag = [Obs['r_tag'].to_numpy()[i][0,0] for i in range(Obs.shape[0])]
        y_tag = [Obs['r_tag'].to_numpy()[i][1,0] for i in range(Obs.shape[0])]
        z_tag = [Obs['r_tag'].to_numpy()[i][2,0] for i in range(Obs.shape[0])]
        
        a = OrbitParams['a'].to_numpy()[0]
        e = OrbitParams['e'].to_numpy()[0]
        b = a*np.sqrt(1 - np.power(e,2))
        
        if False:
            plt.figure()
            plt.subplot(4,1,1)
            plt.title(planet + ' samples')
            plt.plot(x, label=r'$x$')
            plt.plot(x_tag, label=r'$x_t$')
            plt.legend(loc='lower left')
            plt.subplot(4,1,2)
            plt.plot(y, label=r'$y$')
            plt.plot(y_tag, label=r'$y_t$')
            plt.legend(loc='lower left')
            plt.subplot(4,1,3)
            plt.plot(z, label=r'$z$')
            plt.plot(z_tag, label=r'$z_t$')
            plt.legend(loc='lower left')
            plt.subplot(4,1,4)
            plt.plot(np.power(x + a*e, 2) / np.power(a, 2) + np.power(y, 2) / np.power(b, 2), label=r'$\frac{(x+ae)^2}{a^2}+\frac{y^2}{a^2(1-e^2)}$')
            plt.plot(np.power(x_tag + a*e, 2) / np.power(a, 2) + np.power(y_tag, 2) / np.power(b, 2),'--',  label=r'$\frac{(x_t+ae)^2}{a^2}+\frac{y_t^2}{a^2(1-e^2)}$')
            plt.legend()
            plt.tight_layout()
            plt.show(block=False)

    
        plt.figure()
        plt.suptitle(planet + ' ' + OrbitParams["plane eq"].values[0])
        plt.subplot(1,3,1)
        #plt.plot(z, label=r'$z$')
        plt.plot(z_tag, label=r'$z_t$')
        plt.legend(loc='lower left')
        plt.subplot(1,3,2)
        #plt.plot(np.power(x + a*e, 2) / np.power(a, 2) + np.power(y, 2) / np.power(b, 2), label=r'$\frac{(x+ae)^2}{a^2}+\frac{y^2}{a^2(1-e^2)}$')
        ellipse = np.power(x_tag + a*e, 2) / np.power(a, 2) + np.power(y_tag, 2) / np.power(b, 2)
        plt.plot(ellipse,'--',  label=r'$\frac{(x_t+ae)^2}{a^2}+\frac{y_t^2}{a^2(1-e^2)}$')
        plt.legend()
        plt.subplot(1,3,3)
        plt.scatter(x=z_tag, y=ellipse, s=1)
        plt.xlabel(r'$z_t$')
        plt.ylabel(r'$\frac{(x_t+ae)^2}{a^2}+\frac{y_t^2}{a^2(1-e^2)}$')
        plt.tight_layout()
        plt.show(block=False)

        

def solve_ellipse_y_minus_k(A, B, C, h, k, x):
    u = x - h
    a = C
    b = B * u
    c = A * u**2 - 1

    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        return []  # No real solutions
    elif discriminant == 0:
        return [-b / (2 * a)]
    else:
        sqrt_disc = np.sqrt(discriminant)
        return [(-b + sqrt_disc) / (2 * a), (-b - sqrt_disc) / (2 * a)]
    
def solve_ellipse_x_minus_h(A, B, C, h, k, y):
    v = y - k
    a = A
    b = B * v
    c = C * v**2 - 1

    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        return []  # No real solutions
    elif discriminant == 0:
        return [-b / (2 * a)]
    else:
        sqrt_disc = np.sqrt(discriminant)
        return [(-b + sqrt_disc) / (2 * a), (-b - sqrt_disc) / (2 * a)]


def replace_y_with_sign(row, ellipse_params, replace_x=False, to_v=False, noisy=False):
    A, B, C, h, k = ellipse_params['A'], ellipse_params['B'], ellipse_params['C'], ellipse_params['h'], ellipse_params['k']
    if to_v:
        if noisy:
            x0, y0 = row['v_2D_est_noisy'][0,0], row['v_2D_est_noisy'][1,0]
        else:
            x0, y0 = row['v_2D_est'][0,0], row['v_2D_est'][1,0]
    else:
        if noisy:
            x0, y0 = row['r_2D_est_noisy'][0,0], row['r_2D_est_noisy'][1,0]
        else:
            x0, y0 = row['r_2D_est'][0,0], row['r_2D_est'][1,0]
    if np.isnan(x0) or np.isnan(y0):
        return np.ones((2,1))*np.nan
    
    if not replace_x:
        sol = solve_ellipse_y_minus_k(A, B, C, h, k, x0)
        if len(sol) == 0:
            return np.ones((2,1))*np.nan
        elif len(sol) == 1:
            y1, y2 = sol[0], sol[0]
        else:
            y1, y2 = sol[0], sol[1]
        
        if np.abs(y0-y1) < np.abs(y0-y2):
            y_new = 1
        else:
            y_new = -1
        return np.array([[x0], [y_new]])
    
    else:
        sol = solve_ellipse_x_minus_h(A, B, C, h, k, y0)
        if len(sol) == 0:
            return np.ones((2,1))*np.nan
        elif len(sol) == 1:
            x1, x2 = sol[0], sol[0]
        else:
            x1, x2 = sol[0], sol[1]
        
        if np.abs(x0-x1) < np.abs(x0-x2):
            x_new = 1
        else:
            x_new = -1
        return np.array([[x_new], [y0]])
        
    
    
    

def runIRAS(planets, true_anomaly_values_df, orbitalObs_df, orbitalParams_df, runOn2D=False, runOn2D_v=False, runCoordinates_n_Velocities=False, externalReport=False):
    if not type(planets) is list:
        planets = [planets]
    assert sum([runOn2D, runOn2D_v, runCoordinates_n_Velocities]) <= 1
    
    model, Obs, OrbitParams = dict(), dict(), dict()
    for planet in planets:
        if not(true_anomaly_values_df is None):
            model[planet] = true_anomaly_values_df[true_anomaly_values_df['target'] == planet] 
        Obs[planet] = orbitalObs_df[orbitalObs_df['target'] == planet]
        OrbitParams[planet] = orbitalParams_df[orbitalParams_df['target'] == planet]
    
    if not(true_anomaly_values_df is None):
        observations_model = np.zeros((0, model[planets[0]].shape[0], 3))
        for planet in planets:
            x_model = [model[planet]['r'].to_numpy()[i][0,0] for i in range(model[planet].shape[0])]
            y_model = [model[planet]['r'].to_numpy()[i][1,0] for i in range(model[planet].shape[0])]
            z_model = [model[planet]['r'].to_numpy()[i][2,0] for i in range(model[planet].shape[0])]
            observations_single_model = np.concatenate((np.asarray(x_model)[:,None], np.asarray(y_model)[:,None], np.asarray(z_model)[:,None]), axis=1)[None]
            observations_model = np.concatenate((observations_model,observations_single_model), axis=0)
    
    a, e, b = dict(), dict(), dict()
    for planet in planets:
        a[planet] = OrbitParams[planet]['a'].to_numpy()[0]
        e[planet] = OrbitParams[planet]['e'].to_numpy()[0]
        b[planet] = a[planet]*np.sqrt(1 - np.power(e[planet],2))
    
    if not(true_anomaly_values_df is None):
        hypotheses_regulations_model = np.zeros((0, model[planets[0]].shape[0], 1))
        for planet in planets:
            x_model_tag = [model[planet]['r_tag'].to_numpy()[i][0,0] for i in range(model[planet].shape[0])]
            y_model_tag = [model[planet]['r_tag'].to_numpy()[i][1,0] for i in range(model[planet].shape[0])]
            z_model_tag = [model[planet]['r_tag'].to_numpy()[i][2,0] for i in range(model[planet].shape[0])]
            hypotheses_regulations_single_model = (np.power(x_model_tag + a[planet]*e[planet], 2) / np.power(a[planet], 2) + np.power(y_model_tag, 2) / np.power(b[planet], 2))[None,:,None]
            hypotheses_regulations_model = np.concatenate((hypotheses_regulations_model, hypotheses_regulations_single_model), axis=0)
    
    observations = np.zeros((0, Obs[planets[0]].shape[0], 3))
    for planet in planets:
        x = [Obs[planet]['r'].to_numpy()[i][0,0] for i in range(Obs[planet].shape[0])]
        y = [Obs[planet]['r'].to_numpy()[i][1,0] for i in range(Obs[planet].shape[0])]
        z = [Obs[planet]['r'].to_numpy()[i][2,0] for i in range(Obs[planet].shape[0])]
        observations_single = np.concatenate((np.asarray(x)[:,None], np.asarray(y)[:,None], np.asarray(z)[:,None]), axis=1)[None]
        observations = np.concatenate((observations, observations_single), axis=0)
    
    v_observations = np.zeros((0, Obs[planets[0]].shape[0], 3))
    for planet in planets:
        vx = [Obs[planet]['v'].to_numpy()[i][0,0] for i in range(Obs[planet].shape[0])]
        vy = [Obs[planet]['v'].to_numpy()[i][1,0] for i in range(Obs[planet].shape[0])]
        vz = [Obs[planet]['v'].to_numpy()[i][2,0] for i in range(Obs[planet].shape[0])]
        v_observations_single = np.concatenate((np.asarray(vx)[:,None], np.asarray(vy)[:,None], np.asarray(vz)[:,None]), axis=1)[None]
        v_observations = np.concatenate((v_observations, v_observations_single), axis=0)
    
    if runOn2D or runOn2D_v or runCoordinates_n_Velocities:
        coordinate_observations2D_est = np.zeros((0, Obs[planets[0]].shape[0], 2))
        coordinate_observations2D_est_sign_y = np.zeros((0, Obs[planets[0]].shape[0], 2))
        coordinate_observations2D_est_sign_y_noisy = np.zeros((0, Obs[planets[0]].shape[0], 2))
        coordinate_observations2D_estProj2Ellipse = np.zeros((0, Obs[planets[0]].shape[0], 2))
        coordinate_observations2D = np.zeros((0, Obs[planets[0]].shape[0], 2))
        for planet in planets:
            if 'r_2D_est' in Obs[planet].columns:
                alpha = [Obs[planet]['r_2D_est'].to_numpy()[i][0,0] for i in range(Obs[planet].shape[0])]
                beta = [Obs[planet]['r_2D_est'].to_numpy()[i][1,0] for i in range(Obs[planet].shape[0])]
                observations2D_est_single = np.concatenate((np.asarray(alpha)[:,None], np.asarray(beta)[:,None]), axis=1)[None]
                coordinate_observations2D_est = np.concatenate((coordinate_observations2D_est, observations2D_est_single), axis=0)
                
                if False:
                    alpha = [Obs[planet]['r_2D_est_sign_y'].to_numpy()[i][0,0] for i in range(Obs[planet].shape[0])]
                    beta = [Obs[planet]['r_2D_est_sign_y'].to_numpy()[i][1,0] for i in range(Obs[planet].shape[0])]
                    observations2D_est_single = np.concatenate((np.asarray(alpha)[:,None], np.asarray(beta)[:,None]), axis=1)[None]
                    coordinate_observations2D_est_sign_y = np.concatenate((coordinate_observations2D_est_sign_y, observations2D_est_single), axis=0)
                    
                    alpha = [Obs[planet]['r_2D_est_sign_y_noisy'].to_numpy()[i][0,0] for i in range(Obs[planet].shape[0])]
                    beta = [Obs[planet]['r_2D_est_sign_y_noisy'].to_numpy()[i][1,0] for i in range(Obs[planet].shape[0])]
                    observations2D_est_single = np.concatenate((np.asarray(alpha)[:,None], np.asarray(beta)[:,None]), axis=1)[None]
                    coordinate_observations2D_est_sign_y_noisy = np.concatenate((coordinate_observations2D_est_sign_y_noisy, observations2D_est_single), axis=0)
                    
                    
                    
                    alpha = [Obs[planet]['r_2D_est_proj2estEllipse'].to_numpy()[i][0,0] for i in range(Obs[planet].shape[0])]
                    beta = [Obs[planet]['r_2D_est_proj2estEllipse'].to_numpy()[i][1,0] for i in range(Obs[planet].shape[0])]
                    observations2D_est_single = np.concatenate((np.asarray(alpha)[:,None], np.asarray(beta)[:,None]), axis=1)[None]
                    coordinate_observations2D_estProj2Ellipse = np.concatenate((coordinate_observations2D_estProj2Ellipse, observations2D_est_single), axis=0)
            
            alpha = [Obs[planet]['r_2D'].to_numpy()[i][0,0] for i in range(Obs[planet].shape[0])]
            beta = [Obs[planet]['r_2D'].to_numpy()[i][1,0] for i in range(Obs[planet].shape[0])]
            observations2D_est_single = np.concatenate((np.asarray(alpha)[:,None], np.asarray(beta)[:,None]), axis=1)[None]
            coordinate_observations2D = np.concatenate((coordinate_observations2D, observations2D_est_single), axis=0)
            
    if runCoordinates_n_Velocities or runOn2D_v:
        v_observations2D_est = np.zeros((0, Obs[planets[0]].shape[0], 2))
        v_observations2D_est_sign_x = np.zeros((0, Obs[planets[0]].shape[0], 2))
        v_observations2D_est_sign_x_noisy = np.zeros((0, Obs[planets[0]].shape[0], 2))
        v_observations2D_estProj2Ellipse = np.zeros((0, Obs[planets[0]].shape[0], 2))
        v_observations2D = np.zeros((0, Obs[planets[0]].shape[0], 2))
        for planet in planets:
            if 'v_2D_est' in Obs[planet].columns:
                alpha = [Obs[planet]['v_2D_est'].to_numpy()[i][0,0] for i in range(Obs[planet].shape[0])]
                beta = [Obs[planet]['v_2D_est'].to_numpy()[i][1,0] for i in range(Obs[planet].shape[0])]
                observations2D_est_single = np.concatenate((np.asarray(alpha)[:,None], np.asarray(beta)[:,None]), axis=1)[None]
                v_observations2D_est = np.concatenate((v_observations2D_est, observations2D_est_single), axis=0)
                
                alpha = [Obs[planet]['v_2D_est_sign_x'].to_numpy()[i][0,0] for i in range(Obs[planet].shape[0])]
                beta = [Obs[planet]['v_2D_est_sign_x'].to_numpy()[i][1,0] for i in range(Obs[planet].shape[0])]
                observations2D_est_single = np.concatenate((np.asarray(alpha)[:,None], np.asarray(beta)[:,None]), axis=1)[None]
                v_observations2D_est_sign_x = np.concatenate((v_observations2D_est_sign_x, observations2D_est_single), axis=0)
                
                alpha = [Obs[planet]['v_2D_est_sign_x_noisy'].to_numpy()[i][0,0] for i in range(Obs[planet].shape[0])]
                beta = [Obs[planet]['v_2D_est_sign_x_noisy'].to_numpy()[i][1,0] for i in range(Obs[planet].shape[0])]
                observations2D_est_single = np.concatenate((np.asarray(alpha)[:,None], np.asarray(beta)[:,None]), axis=1)[None]
                v_observations2D_est_sign_x_noisy = np.concatenate((v_observations2D_est_sign_x_noisy, observations2D_est_single), axis=0)
                
                alpha = [Obs[planet]['v_2D_est_proj2estEllipse'].to_numpy()[i][0,0] for i in range(Obs[planet].shape[0])]
                beta = [Obs[planet]['v_2D_est_proj2estEllipse'].to_numpy()[i][1,0] for i in range(Obs[planet].shape[0])]
                observations2D_est_single = np.concatenate((np.asarray(alpha)[:,None], np.asarray(beta)[:,None]), axis=1)[None]
                v_observations2D_estProj2Ellipse = np.concatenate((v_observations2D_estProj2Ellipse, observations2D_est_single), axis=0)
            
            alpha = [Obs[planet]['v_2D'].to_numpy()[i][0,0] for i in range(Obs[planet].shape[0])]
            beta = [Obs[planet]['v_2D'].to_numpy()[i][1,0] for i in range(Obs[planet].shape[0])]
            observations2D_est_single = np.concatenate((np.asarray(alpha)[:,None], np.asarray(beta)[:,None]), axis=1)[None]
            v_observations2D = np.concatenate((v_observations2D, observations2D_est_single), axis=0)
  
    observations_tVec = np.repeat(np.arange(len(x))[None,:,None], observations.shape[0], 0)
    
    observations_tag = np.zeros((0, Obs[planets[0]].shape[0], 3))
    hypotheses_regulation_plane = np.zeros((0, Obs[planets[0]].shape[0], 1))
    hypotheses_regulation_ellipse = np.zeros((0, Obs[planets[0]].shape[0], 1))
    for planet in planets:
        x_tag = [Obs[planet]['r_tag'].to_numpy()[i][0,0] for i in range(Obs[planet].shape[0])]
        y_tag = [Obs[planet]['r_tag'].to_numpy()[i][1,0] for i in range(Obs[planet].shape[0])]
        z_tag = [Obs[planet]['r_tag'].to_numpy()[i][2,0] for i in range(Obs[planet].shape[0])]
        observations_tag_single = np.concatenate((np.asarray(x_tag)[:,None], np.asarray(y_tag)[:,None], np.asarray(z_tag)[:,None]), axis=1)[None]
        observations_tag = np.concatenate((observations_tag, observations_tag_single), axis=0)
    
        hypotheses_regulation_plane_single = np.asarray(z_tag)[None,:,None]
        hypotheses_regulation_plane = np.concatenate((hypotheses_regulation_plane, hypotheses_regulation_plane_single), axis=0)
    
        hypotheses_regulation_ellipse_single = (np.power(x_tag + a[planet]*e[planet], 2) / np.power(a[planet], 2) + np.power(y_tag, 2) / np.power(b[planet], 2))[None,:,None]
        hypotheses_regulation_ellipse = np.concatenate((hypotheses_regulation_ellipse, hypotheses_regulation_ellipse_single), axis=0)

    
    coordinate_observations_noisy = np.zeros((0, Obs[planets[0]].shape[0], 3))
    hypotheses_regulation_plane_noisy = np.zeros((0, Obs[planets[0]].shape[0], 1))
    coordinate_observations2D_est_noisy = np.zeros((0, Obs[planets[0]].shape[0], 2))
    coordinate_observations2D_estProj2Ellipse_noisy = np.zeros((0, Obs[planets[0]].shape[0], 2))
    coordinate_observations2D_noisy = np.zeros((0, Obs[planets[0]].shape[0], 2))
    v_observations_noisy = np.zeros((0, Obs[planets[0]].shape[0], 3))
    v_observations2D_est_noisy = np.zeros((0, Obs[planets[0]].shape[0], 2))
    v_observations2D_estProj2Ellipse_noisy = np.zeros((0, Obs[planets[0]].shape[0], 2))
    Lx_observations_noisy = np.zeros((0, Obs[planets[0]].shape[0], 1))
    for planet in planets:
        x_noisy = [Obs[planet]['rNoisy'].to_numpy()[i][0,0] for i in range(Obs[planet].shape[0])]
        y_noisy = [Obs[planet]['rNoisy'].to_numpy()[i][1,0] for i in range(Obs[planet].shape[0])]
        z_noisy = [Obs[planet]['rNoisy'].to_numpy()[i][2,0] for i in range(Obs[planet].shape[0])]
        coordinate_observations_noisy_single = np.concatenate((np.asarray(x_noisy)[:,None], np.asarray(y_noisy)[:,None], np.asarray(z_noisy)[:,None]), axis=1)[None]
        coordinate_observations_noisy = np.concatenate((coordinate_observations_noisy, coordinate_observations_noisy_single), axis=0)
        
        
        x_tag_noisy = [Obs[planet]['r_tagNoisy'].to_numpy()[i][0,0] for i in range(Obs[planet].shape[0])]
        y_tag_noisy = [Obs[planet]['r_tagNoisy'].to_numpy()[i][1,0] for i in range(Obs[planet].shape[0])]
        z_tag_noisy = [Obs[planet]['r_tagNoisy'].to_numpy()[i][2,0] for i in range(Obs[planet].shape[0])]
        hypotheses_regulation_plane_single = np.asarray(z_tag_noisy)[None,:,None]
        hypotheses_regulation_plane_noisy = np.concatenate((hypotheses_regulation_plane_noisy, hypotheses_regulation_plane_single), axis=0)
        
        if runOn2D or runCoordinates_n_Velocities or runOn2D_v:
            x_noisy = [Obs[planet]['r_2D_noisy'].to_numpy()[i][0,0] for i in range(Obs[planet].shape[0])]
            y_noisy = [Obs[planet]['r_2D_noisy'].to_numpy()[i][1,0] for i in range(Obs[planet].shape[0])]
            coordinate_observations_noisy_single = np.concatenate((np.asarray(x_noisy)[:,None], np.asarray(y_noisy)[:,None]), axis=1)[None]
            coordinate_observations2D_noisy = np.concatenate((coordinate_observations2D_noisy, coordinate_observations_noisy_single), axis=0)
            
            if 'r_2D_est_noisy' in Obs[planet].columns:
                x_noisy = [Obs[planet]['r_2D_est_noisy'].to_numpy()[i][0,0] for i in range(Obs[planet].shape[0])]
                y_noisy = [Obs[planet]['r_2D_est_noisy'].to_numpy()[i][1,0] for i in range(Obs[planet].shape[0])]
                coordinate_observations_noisy_single = np.concatenate((np.asarray(x_noisy)[:,None], np.asarray(y_noisy)[:,None]), axis=1)[None]
                coordinate_observations2D_est_noisy = np.concatenate((coordinate_observations2D_est_noisy, coordinate_observations_noisy_single), axis=0)
                
                if False:
                    x_noisy = [Obs[planet]['r_2D_est_proj2estEllipse_noisy'].to_numpy()[i][0,0] for i in range(Obs[planet].shape[0])]
                    y_noisy = [Obs[planet]['r_2D_est_proj2estEllipse_noisy'].to_numpy()[i][1,0] for i in range(Obs[planet].shape[0])]
                    coordinate_observations_noisy_single = np.concatenate((np.asarray(x_noisy)[:,None], np.asarray(y_noisy)[:,None]), axis=1)[None]
                    coordinate_observations2D_estProj2Ellipse_noisy = np.concatenate((coordinate_observations2D_estProj2Ellipse_noisy, coordinate_observations_noisy_single), axis=0)
                
                
        
        Lx_noisy = [Obs[planet]['L_Noisy'].to_numpy()[i][0,0] for i in range(Obs[planet].shape[0])]
        #print(f'Lx_noisy.shape={np.asarray(Lx_noisy)[:,None].shape}')
        Lx_observations_single_noisy = np.asarray(Lx_noisy)[:,None][None]#np.concatenate((np.asarray(Lx_noisy)[:,None]), axis=1)[None]
        Lx_observations_noisy = np.concatenate((Lx_observations_noisy, Lx_observations_single_noisy), axis=0)
        
        
        vx_noisy = [Obs[planet]['vNoisy'].to_numpy()[i][0,0] for i in range(Obs[planet].shape[0])]
        vy_noisy = [Obs[planet]['vNoisy'].to_numpy()[i][1,0] for i in range(Obs[planet].shape[0])]
        vz_noisy = [Obs[planet]['vNoisy'].to_numpy()[i][2,0] for i in range(Obs[planet].shape[0])]
        v_observations_noisy_single = np.concatenate((np.asarray(vx_noisy)[:,None], np.asarray(vy_noisy)[:,None], np.asarray(vz_noisy)[:,None]), axis=1)[None]
        v_observations_noisy = np.concatenate((v_observations_noisy, v_observations_noisy_single), axis=0)
        
        if runCoordinates_n_Velocities or runOn2D_v:
            if 'v_2D_est_noisy' in Obs[planet]:
                vx_noisy = [Obs[planet]['v_2D_est_noisy'].to_numpy()[i][0,0] for i in range(Obs[planet].shape[0])]
                vy_noisy = [Obs[planet]['v_2D_est_noisy'].to_numpy()[i][1,0] for i in range(Obs[planet].shape[0])]
                v_observations_noisy_single = np.concatenate((np.asarray(vx_noisy)[:,None], np.asarray(vy_noisy)[:,None]), axis=1)[None]
                v_observations2D_est_noisy = np.concatenate((v_observations2D_est_noisy, v_observations_noisy_single), axis=0)
                
                vx_noisy = [Obs[planet]['v_2D_est_proj2estEllipse_noisy'].to_numpy()[i][0,0] for i in range(Obs[planet].shape[0])]
                vy_noisy = [Obs[planet]['v_2D_est_proj2estEllipse_noisy'].to_numpy()[i][1,0] for i in range(Obs[planet].shape[0])]
                v_observations_noisy_single = np.concatenate((np.asarray(vx_noisy)[:,None], np.asarray(vy_noisy)[:,None]), axis=1)[None]
                v_observations2D_estProj2Ellipse_noisy = np.concatenate((v_observations2D_estProj2Ellipse_noisy, v_observations_noisy_single), axis=0)
    
    if runOn2D or runCoordinates_n_Velocities or runOn2D_v:
        observations2D_est_noisy = np.zeros((0, Obs[planets[0]].shape[0], 2))
        v_observations2D_noisy = np.zeros((0, Obs[planets[0]].shape[0], 2))
        L_2D_noisy = np.zeros((0, Obs[planets[0]].shape[0], 1))
        L_2D = np.zeros((0, Obs[planets[0]].shape[0], 1))
        L_2D_proj2Ellipse_noisy = np.zeros((0, Obs[planets[0]].shape[0], 1))
        L_2D_est_sign = np.zeros((0, Obs[planets[0]].shape[0], 1))
        L_2D_est_sign_noisy = np.zeros((0, Obs[planets[0]].shape[0], 1))
        L_2D_proj2Ellipse = np.zeros((0, Obs[planets[0]].shape[0], 1))
        for planet in planets:
            observations2D_est_noisy_single = np.asarray([Obs[planet]['L_2D'].to_numpy()[i][0,0] for i in range(Obs[planet].shape[0])])[None,:,None]
            L_2D = np.concatenate((L_2D, observations2D_est_noisy_single), axis=0)
            
            alpha = [Obs[planet]['v_2D_noisy'].to_numpy()[i][0,0] for i in range(Obs[planet].shape[0])]
            beta = [Obs[planet]['v_2D_noisy'].to_numpy()[i][1,0] for i in range(Obs[planet].shape[0])]
            observations2D_est_noisy_single = np.concatenate((np.asarray(alpha)[:,None], np.asarray(beta)[:,None]), axis=1)[None]
            v_observations2D_noisy = np.concatenate((v_observations2D_noisy, observations2D_est_noisy_single), axis=0)
            
            observations2D_est_noisy_single = np.asarray([Obs[planet]['L_2D_noisy'].to_numpy()[i][0,0] for i in range(Obs[planet].shape[0])])[None,:,None]
            L_2D_noisy = np.concatenate((L_2D_noisy, observations2D_est_noisy_single), axis=0)
            
            if 'r_2D_est_noisy' in Obs[planet]:
                alpha = [Obs[planet]['r_2D_est_noisy'].to_numpy()[i][0,0] for i in range(Obs[planet].shape[0])]
                beta = [Obs[planet]['r_2D_est_noisy'].to_numpy()[i][1,0] for i in range(Obs[planet].shape[0])]
                observations2D_est_noisy_single = np.concatenate((np.asarray(alpha)[:,None], np.asarray(beta)[:,None]), axis=1)[None]
                observations2D_est_noisy = np.concatenate((observations2D_est_noisy, observations2D_est_noisy_single), axis=0)
                
                if False:
                    observations2D_est_noisy_single = np.asarray([Obs[planet]['L_2D_est_proj2estEllipse_noisy'].to_numpy()[i][0,0] for i in range(Obs[planet].shape[0])])[None,:,None]
                    L_2D_proj2Ellipse_noisy = np.concatenate((L_2D_proj2Ellipse_noisy, observations2D_est_noisy_single), axis=0)
                    
                    observations2D_est_noisy_single = np.asarray([Obs[planet]['L_2D_est_proj2estEllipse'].to_numpy()[i][0,0] for i in range(Obs[planet].shape[0])])[None,:,None]
                    L_2D_proj2Ellipse = np.concatenate((L_2D_proj2Ellipse, observations2D_est_noisy_single), axis=0)
                    
                    observations2D_est_noisy_single = np.asarray([Obs[planet]['L_2D_est_sign'].to_numpy()[i][0,0] for i in range(Obs[planet].shape[0])])[None,:,None]
                    L_2D_est_sign = np.concatenate((L_2D_est_sign, observations2D_est_noisy_single), axis=0)
                    
                    observations2D_est_noisy_single = np.asarray([Obs[planet]['L_2D_est_sign_noisy'].to_numpy()[i][0,0] for i in range(Obs[planet].shape[0])])[None,:,None]
                    L_2D_est_sign_noisy = np.concatenate((L_2D_est_sign_noisy, observations2D_est_noisy_single), axis=0)
            
    
    
    hypotheses_regulations_plane_noisy, hypotheses_regulations_ellipse_noisy = None, None
    
    
    
    onlyPolyMixTerms = False
    playerPerPatient = False
    hypotheses_regulations_pearson = None
    hypotheses_regulations = None
    if runOn2D:
        observations_noisy = coordinate_observations2D_est_noisy
        observations2IRAS = coordinate_observations2D_est
        degreeOfPolyFit_pearson = [2]
        features2ShuffleTogether = None#[[0], [1]]
    elif runCoordinates_n_Velocities:
        
        #observations_noisy = np.concatenate((coordinate_observations2D_estProj2Ellipse_noisy, coordinate_observations2D_estProj2Ellipse_noisy), axis=-1) #np.concatenate((coordinate_observations2D_est_noisy, v_observations2D_est_noisy), axis=-1)
        #observations2IRAS = np.concatenate((coordinate_observations2D_estProj2Ellipse, v_observations2D_estProj2Ellipse), axis=-1) #np.concatenate((coordinate_observations2D_est, v_observations2D_est), axis=-1)#observations_noisy
        #L_2D_est_noisy = [Obs[planet]['L_2D_est_noisy'].to_numpy()[i][0,0] for i in range(Obs[planet].shape[0])]
        #hypotheses_regulations_pearson = np.asarray(L_2D_est_noisy)[None,:,None]
        
        observations2IRAS = np.concatenate((coordinate_observations2D, v_observations2D), axis=-1)
        observations_noisy = np.concatenate((coordinate_observations2D_noisy, v_observations2D_noisy), axis=-1)
        
        #observations2IRAS = np.concatenate((coordinate_observations2D_est, v_observations2D_est[:1]), axis=-1)#observations_noisy
        #observations_noisy = np.concatenate((coordinate_observations2D_est_noisy, v_observations2D_est_noisy[:1]), axis=-1)
        
        #observations2IRAS = np.concatenate((coordinate_observations2D_est_sign_y, v_observations2D_est_sign_x), axis=-1)
        #observations_noisy = np.concatenate((coordinate_observations2D_est_sign_y_noisy, v_observations2D_est_sign_x_noisy), axis=-1)
        
        
        hypotheses_regulations_pearson = L_2D_noisy
        hypotheses_regulations = L_2D
        
        #indices = np.logical_not(np.logical_or(np.isnan(hypotheses_regulations).any(axis=-1), np.logical_or(np.isnan(observations_noisy).any(axis=-1), np.logical_or(np.isnan(observations2IRAS).any(axis=-1), np.isnan(hypotheses_regulations_pearson).any(axis=-1))))[0])
        #observations2IRAS = observations2IRAS[:,indices]
        #observations_noisy = observations_noisy[:,indices]
        
        #hypotheses_regulations = hypotheses_regulations[:,indices]
        #hypotheses_regulations_pearson = hypotheses_regulations_pearson[:,indices]
        
        print(f'observations2IRAS.shape={observations2IRAS.shape}')
        print(f'observations_noisy.shape={observations_noisy.shape}')
        #return
        
        #hypotheses_regulations = L_2D
        
        degreeOfPolyFit_pearson = [2]
        onlyPolyMixTerms = True
        playerPerPatient = True
        features2ShuffleTogether = [[0,1],[2,3]]
    elif runOn2D_v:
        observations_noisy = v_observations2D_est_noisy
        observations2IRAS = v_observations2D_est
        degreeOfPolyFit_pearson = [2]
        features2ShuffleTogether = [[0], [1]]
    else:
        observations_noisy = coordinate_observations_noisy
        hypotheses_regulations_pearson = hypotheses_regulation_plane_noisy
        observations2IRAS = observations
        degreeOfPolyFit_pearson = [1]
        features2ShuffleTogether = None#[[0], [1], [2]]
        
    observations_tVec = np.repeat(np.arange(observations2IRAS.shape[1])[None,:,None], observations2IRAS.shape[0], 0)
    if False:
        data={'time': observations_tVec.flatten(),
         'Id': np.repeat(np.arange(observations_tVec.shape[0]), observations_tVec.shape[1]),
         'batch': np.zeros(observations_tVec.shape[0]*observations_tVec.shape[1]),
         'x': observations2IRAS[:,:,0].flatten(),
         'y': observations2IRAS[:,:,1].flatten(),
         'vx': observations2IRAS[:,:,2].flatten(),
         'vy': observations2IRAS[:,:,3].flatten(),
         'L': hypotheses_regulations.flatten()}
        pd.DataFrame(data).to_csv('/Users/ron.teichner/Library/CloudStorage/OneDrive-Technion/Kepler/'+'orbit2D'+'.csv')
        
        data={'time': observations_tVec.flatten(),
         'Id': np.repeat(np.arange(observations_tVec.shape[0]), observations_tVec.shape[1]),
         'batch': np.zeros(observations_tVec.shape[0]*observations_tVec.shape[1]),
         'x': observations_noisy[:,:,0].flatten(),
         'y': observations_noisy[:,:,1].flatten(),
         'vx': observations_noisy[:,:,2].flatten(),
         'vy': observations_noisy[:,:,3].flatten(),
         'L': hypotheses_regulations_pearson.flatten()}
        pd.DataFrame(data).to_csv('/Users/ron.teichner/Library/CloudStorage/OneDrive-Technion/Kepler/'+'orbit2DNoisy'+'.csv')
        
        
        print(f'hypotheses_regulations_pearson.shape={hypotheses_regulations_pearson.shape}')
        print(f'observations_noisy.shape={observations_noisy.shape}')
        return
        
        plt.figure()
        plt.subplot(1,2,1)
        for hyp, planet in zip(hypotheses_regulations_pearson, planets):
            plt.plot(hyp[:,0], label=planet)
        plt.legend()
        plt.subplot(1,2,2)
        for hyp, planet in zip(hypotheses_regulations, planets):
            plt.plot(hyp[:,0], label=planet)
        plt.legend()
        plt.show(block=False)

        return
    
        
    
    nIRAS_iter = 1
    min_CR_naive = np.inf

    enableSim = False
    if enableSim:
        small_noise = 10+1*np.random.randn(1,observations2IRAS.shape[1],1)
        small_x = np.random.randn(1,observations2IRAS.shape[1],1)
        small_y = 10*np.random.randn(1,observations2IRAS.shape[1],1)
        small_z = small_x - small_y + small_noise
        observations2IRAS = np.concatenate((small_x, small_y, small_z), axis=2)
        hypotheses_regulations_plane_noisy = small_noise[None]
        observations_noisy = observations2IRAS
        hypotheses_regulations_pearson = np.concatenate((small_noise[None], small_noise[None]), axis=0)
        nIRAS_iter = 1
        
    enableScaleFeatures = False
    if enableScaleFeatures:
        # Example: assuming observations2IRAS and observations_noisy are already defined
        # Shape: (batch, time, features)
        
        # Reshape to 2D for scaling: (batch*time, features)
        batch, time, features = observations2IRAS.shape
        reshaped_obs = observations2IRAS.reshape(-1, features)
        
        # Fit the scaler on observations2IRAS
        scaler = StandardScaler()
        scaled_obs = scaler.fit_transform(reshaped_obs)
        
        # Apply the same scaler to observations_noisy
        reshaped_noisy = observations_noisy.reshape(-1, features)
        scaled_noisy = scaler.transform(reshaped_noisy)
        
        # Reshape back to original 3D shape
        observations2IRAS = scaled_obs.reshape(batch, time, features)
        observations_noisy = scaled_noisy.reshape(batch, time, features)
    
    if runCoordinates_n_Velocities:
        MercuryIdx = np.where(np.asarray(planets)==199)[0][0]
        hypotheses_regulations_pearson = hypotheses_regulations_pearson[MercuryIdx:MercuryIdx+1]
        observations_noisy = observations_noisy[MercuryIdx:MercuryIdx+1]
        

        
        
        if False:
            # Step 1: Reshape by flattening the first dimension
            observations_noisy_reshaped = observations_noisy.reshape(-1, observations_noisy.shape[2])  # Resulting shape: (50000, 4)
            hypotheses_regulations_pearson_reshaped = hypotheses_regulations_pearson.reshape(-1, hypotheses_regulations_pearson.shape[2]) 
            
            # Step 2: Permute the first dimension
            permuted_indices = np.random.permutation(observations_noisy_reshaped.shape[0])
            observations_noisy_permuted = observations_noisy_reshaped[permuted_indices]
            hypotheses_regulations_pearson_permuted = hypotheses_regulations_pearson_reshaped[permuted_indices]
            
            #observations_noisy = observations_noisy_permuted[None]
            #hypotheses_regulations_pearson = hypotheses_regulations_pearson_permuted[None]
            
            observations_noisy = observations_noisy_permuted[:3*observations_tVec.shape[1]][None]
            hypotheses_regulations_pearson = hypotheses_regulations_pearson_permuted[:3*observations_tVec.shape[1]][None]
        
    else:
        if runOn2D:
            hypotheses_regulations_pearson = None
        else:
            hypotheses_regulations_pearson = hypotheses_regulations_pearson[:1]
        observations_noisy = observations_noisy[:1]
    hypotheses_regulations = None

    #print(f'{observations2IRAS.shape}')        
    #print(f'{observations_noisy.shape}')    
    #print(f'{hypotheses_regulations_pearson.shape}')  
    #print(f'{hypotheses_regulations}')   
    #return
    if not planet is str:
        planet_title = str(planet)
    else:
        planet_title = planet
    
    for i in range(nIRAS_iter):
        implicitPolyDictList = IRAS_train_script(observations2IRAS, observations_tVec, hypotheses_regulations, seriesForPearson=observations_noisy, hypothesesForPearson=hypotheses_regulations_pearson, titleStr=planet_title, nativeIRAS=True, nEpochs=500, degreeOfPolyFit=degreeOfPolyFit_pearson, onlyPolyMixTerms=onlyPolyMixTerms, externalReport=externalReport, features2ShuffleTogether=features2ShuffleTogether, playerPerPatient=playerPerPatient)
        if implicitPolyDictList[0]['CR_zeta1'] < min_CR_naive:
            minCR_implicitPolyDictList = implicitPolyDictList
            min_CR_naive = implicitPolyDictList[0]['CR_zeta1']
    
    if runOn2D or runOn2D_v:
        for minCR_implicitPolyDict,degreeOfPolyFit in zip(minCR_implicitPolyDictList, degreeOfPolyFit_pearson):
            if degreeOfPolyFit == 2:
                 #ellipse_fit_1 = fit_ellipse(minCR_implicitPolyDict['singleBatch'], minCR_implicitPolyDict['combination'], minCR_implicitPolyDict['intercept'], minCR_implicitPolyDict['coefficients'])
                 #ellipse_fit_2 = fit_ellipse(minCR_implicitPolyDict['singleBatch'], -minCR_implicitPolyDict['combination'], minCR_implicitPolyDict['intercept'], minCR_implicitPolyDict['coefficients'])
                 
                 ellipse_fit_1 = fit_ellipse_analytic(observations2IRAS, minCR_implicitPolyDict['singleBatch'], minCR_implicitPolyDict['combination'], minCR_implicitPolyDict['intercept'], minCR_implicitPolyDict['coefficients'])
                 ellipse_fit_2 = fit_ellipse_analytic(observations2IRAS, minCR_implicitPolyDict['singleBatch'], -minCR_implicitPolyDict['combination'], minCR_implicitPolyDict['intercept'], minCR_implicitPolyDict['coefficients'])
                 
                 if np.abs(ellipse_fit_1['corr']) > np.abs(ellipse_fit_2['corr']):
                     minCR_implicitPolyDict['ellipse_fit'] = ellipse_fit_1
                 else:
                     minCR_implicitPolyDict['ellipse_fit'] = ellipse_fit_2
                     
                 #theta = minCR_implicitPolyDict['ellipse_fit']['ellipse_rot_angle']
                 #R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta),  np.cos(theta)]])
                 
                 #minCR_implicitPolyDict['ellipse_fit']['rotated_ellipse'] = fit_ellipse_analytic(torch.tensor((minCR_implicitPolyDict['singleBatch'] @ minCR_implicitPolyDict['singleBatch'].numpy()[:,:,:,None])[:,:,:,0]), minCR_implicitPolyDict['combination'], minCR_implicitPolyDict['intercept'], minCR_implicitPolyDict['coefficients']
            
             

    
    return {'planet': planet, 'observations2IRAS': observations2IRAS, 'degreeOfPolyFit': degreeOfPolyFit_pearson, 'implicitPolyDictList': minCR_implicitPolyDictList}

############################################

def extract_ellipse_parameters(coefficients):
    """
    Given polynomial coefficients for features ['1', 'x0', 'x1', 'x0^2', 'x0 x1', 'x1^2'],
    return the corresponding ellipse parameters A, B, C, h, k and the reconstructed constant term.
    """
    print(f'coefficients={coefficients}')
    intercept, coef_x0, coef_x1, coef_x0_sq, coef_x0x1, coef_x1_sq = coefficients

    A = coef_x0_sq
    B = coef_x0x1
    C = coef_x1_sq

    # Solve the linear system for h and k
    M = np.array([[2*A, B], [B, 2*C]])
    rhs = -np.array([coef_x0, coef_x1])

    if np.linalg.det(M) == 0:
        raise ValueError("The coefficient matrix is singular and cannot be inverted.")

    h, k = np.linalg.solve(M, rhs)

    # Reconstruct the constant term
    constant_term = A*h**2 + B*h*k + C*k**2 + coef_x0*h + coef_x1*k + intercept

    return {
        'A': A,
        'B': B,
        'C': C,
        'h': h,
        'k': k,
        'reconstructed_constant': constant_term
    }



def compute_foci(A, B, C, h, k):
    """
    Compute the coordinates of the foci of an ellipse defined by:
    A(x−h)^2 + B(x−h)(y−k) + C(y−k)^2 = 1
    """
    # Step 1: Construct the matrix of the quadratic form
    M = np.array([[A, B / 2],
                  [B / 2, C]])

    # Step 2: Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(M)

    # Step 3: Sort eigenvalues to identify semi-major and semi-minor axes
    idx = np.argsort(eigenvalues)
    lambda1, lambda2 = eigenvalues[idx[0]], eigenvalues[idx[1]]
    v1, v2 = eigenvectors[:, idx[0]], eigenvectors[:, idx[1]]

    # Step 4: Compute semi-axes lengths
    a = 1 / np.sqrt(lambda1)
    b = 1 / np.sqrt(lambda2)

    # Ensure a is the semi-major axis
    if a < b:
        a, b = b, a
        v1, v2 = v2, v1

    # Step 5: Compute distance to foci
    c = np.sqrt(a**2 - b**2)

    # Step 6: Compute foci coordinates
    focus1 = (h + c * v1[0], k + c * v1[1])
    focus2 = (h - c * v1[0], k - c * v1[1])

    return focus1, focus2


def ellipse_axes_and_eccentricity(A, B, C, h, k):
    """
    Given the parameters of an ellipse in the form:
    A(x−h)^2 + B(x−h)(y−k) + C(y−k)^2 = 1,
    compute the semi-major axis, semi-minor axis, and eccentricity.
    """
    # Construct the matrix of the quadratic form
    M = np.array([[A, B / 2],
                  [B / 2, C]])

    # Compute eigenvalues of the matrix
    eigenvalues = np.linalg.eigvalsh(M)

    # Sort eigenvalues to identify major and minor axes
    lambda1, lambda2 = np.sort(eigenvalues)

    # Semi-axes lengths
    a = 1 / np.sqrt(lambda1)  # semi-major axis
    b = 1 / np.sqrt(lambda2)  # semi-minor axis

    # Ensure a is the larger axis
    if a < b:
        a, b = b, a

    # Eccentricity
    eccentricity = np.sqrt(1 - (b**2 / a**2))

    return {'a':a, 'b':b, 'e':eccentricity}


def fit_ellipse_analytic(observations2IRAS, singleBatch, combination, intercept, coefficients):
    #print(f'singleBatch.shape{singleBatch.shape}, combination.shape={combination.shape}')
    observations2IRAS, singleBatch, combination = observations2IRAS.reshape(-1, 2), singleBatch.reshape(-1, 2).numpy(), combination.reshape(-1, 1).detach().numpy()
    #print(f'singleBatch.shape{singleBatch.shape}, combination.shape={combination.shape}')
    
    alpha, beta = observations2IRAS[:,0], observations2IRAS[:,1]
    alpha_noisy, beta_noisy, l_data = singleBatch[:,0], singleBatch[:,1], combination[:,0]

    # Design matrix for conic fitting: Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0
    D = np.vstack([alpha**2, alpha * beta, beta**2, alpha, beta, np.ones_like(alpha)]).T
    
    # Solve the normal equations using least squares
    _, _, V = np.linalg.svd(D, full_matrices=False)
    conic_params = V[-1, :]  # solution is the last row of V
    
    A, B, C, D_, E, F = conic_params
    
    # Compute center (h, k) of the ellipse
    M = np.array([[2*A, B], [B, 2*C]])
    rhs = np.array([-D_, -E])
    center = np.linalg.solve(M, rhs)
    h, k = center
    
    # Translate coordinates
    x_shift = alpha - h
    y_shift = beta - k
    
    # Evaluate the ellipse equation in shifted coordinates
    Z = A * x_shift**2 + B * x_shift * y_shift + C * y_shift**2
    
    # Normalize so that the ellipse equation equals 1
    scale = np.mean(Z)
    A /= scale
    B /= scale
    C /= scale
    
    # Generate the ellipse equation string
    optimized_params = [A, B, C, h, k]
    equation_str = ellipse_equation_string(optimized_params)
    
    # Predict l_data using the optimized parameters
    predicted_l_data = ellipse_model(optimized_params, alpha_noisy, beta_noisy)
    
    corr = pd.Series(predicted_l_data).corr(pd.Series(l_data))
    
    return {'ellipse_eq': equation_str, 'corr': corr, 'ellipse_params': {'A':A, 'B':B, 'C':C, 'h':h, 'k':  k}, 'ellipse_rot_angle': clc_ellipse_theta(A,B,C), 'axes_eccentricity': ellipse_axes_and_eccentricity(A, B, C, h, k), 'focci': compute_foci(A, B, C, h, k)}
    
def clc_ellipse_theta(A,B,C):
    # Compute tan(2θ)
    tan_2theta = B / (A - C)
    
    # Compute 2θ
    two_theta = np.arctan(tan_2theta)
    
    # Compute θ in radians and degrees
    theta_rad = 0.5 * two_theta
    return theta_rad
    


def fit_ellipse(singleBatch, combination, intercept, coefficients):
    #print(f'singleBatch.shape{singleBatch.shape}, combination.shape={combination.shape}')
    singleBatch, combination = singleBatch.reshape(-1, 2).numpy(), combination.reshape(-1, 1).detach().numpy()
    #print(f'singleBatch.shape{singleBatch.shape}, combination.shape={combination.shape}')
    
    alpha, beta, l_data = singleBatch[:,0], singleBatch[:,1], combination[:,0]
    #print(f'alpha.shape{alpha.shape}, beta.shape={beta.shape}, l_data.shape{l_data.shape}')
    
    l_data = (l_data-l_data.mean())/l_data.std() + 1
    #print(f'intercept={intercept}, coefficients={coefficients}')
    #print(f'type intercept={type(intercept)}')
    #print(f'type coefficients={type(coefficients)}')
    #print(f'combined={[intercept] + coefficients[1:]}')
    #if True:
    #    return {}

    # Initial guess for elippse parameters A, B, C, h, k
    initial_guess_dict = extract_ellipse_parameters([intercept] + list(coefficients[1:]))
    initial_guess = [initial_guess_dict['A'], initial_guess_dict['B'], initial_guess_dict['C'], initial_guess_dict['h'], initial_guess_dict['k']]
    #initial_guess = initial_guess + [np.mean(l_data/ellipse_model(initial_guess+[1], alpha, beta))]
    
    # Perform least squares optimization
    result = least_squares(residuals, initial_guess, args=(alpha, beta, l_data))
    
    # Extract the optimized parameters
    optimized_params = result.x
    A_opt, B_opt, C_opt, h_opt, k_opt = optimized_params

    # Generate the ellipse equation string
    equation_str = ellipse_equation_string(optimized_params)

    # Predict l_data using the optimized parameters
    predicted_l_data = ellipse_model(optimized_params, alpha, beta)
    
    corr = pd.Series(predicted_l_data).corr(pd.Series(l_data))
    
    return {'ellipse_eq': equation_str, 'corr': corr, 'ellipse_params': {'A':A_opt, 'B':B_opt, 'C':C_opt, 'h':h_opt, 'k':  k_opt}, 'ellipse_rot_angle': clc_ellipse_theta(A_opt,B_opt,C_opt), 'axes_eccentricity': ellipse_axes_and_eccentricity(A_opt, B_opt, C_opt, h_opt, k_opt), 'focci': compute_foci(A_opt, B_opt, C_opt, h_opt, k_opt)}


# Define the model function
def ellipse_model(params, x, y):
    A, B, C, h, k = params
    return A * (x - h)**2 + B * (x - h) * (y - k) + C * (y - k)**2

# Define the residuals function to minimize
def residuals(params, x, y, l):
    #print(f'params={params}')
    #print(f'x.shape={x.shape}')
    #print(f'y.shape={y.shape}')
    #print(f'l.shape={l.shape}')
    #print(f'ellipse shape={ellipse_model(params, x, y).shape}')
    return ellipse_model(params, x, y) - l

# Function to generate the ellipse equation string
def ellipse_equation_string(params):
    A, B, C, h, k = params
    return f"{A:.4f}(x - {h:.4f})² + {B:.4f}(x - {h:.4f})(y - {k:.4f}) + {C:.4f}(y - {k:.4f})²"


############################################

def get_highScoreHyp(IRAS_runOnCoordinatesResults):
    s=0
    maxCorr = 0
    
    if len(IRAS_runOnCoordinatesResults['implicitPolyDictList'])==1:
        return 0
    
    implicitPolyDict = IRAS_runOnCoordinatesResults['implicitPolyDictList'][0]
    combination = implicitPolyDict['combination']
    hypotheses_regulations = implicitPolyDict['hypotheses_regulations']
    hyp0_pearsonCorr = pd.Series(combination[s].detach().cpu().flatten()).corr(pd.Series(hypotheses_regulations[s,:,0].flatten()))
    
    implicitPolyDict = IRAS_runOnCoordinatesResults['implicitPolyDictList'][1]
    combination = implicitPolyDict['combination']
    hypotheses_regulations = implicitPolyDict['hypotheses_regulations']
    hyp1_pearsonCorr = pd.Series(combination[s].detach().cpu().flatten()).corr(pd.Series(hypotheses_regulations[s,:,0].flatten()))
    
    if np.abs(hyp1_pearsonCorr) > np.abs(hyp0_pearsonCorr) + 0.1:
        highScoreHyp = 1
    else:
        highScoreHyp = 0
    
    return highScoreHyp
        

def solve_for_x0(coefficients, intercept, x1, x2, x3, comb):
    # Coefficients correspond to the polynomial terms in the order:
    # [1, x0, x1, x2, x3, x0^2, x0*x1, x0*x2, x0*x3, x1^2, x1*x2, x1*x3, x2^2, x2*x3, x3^2]
    
    a = coefficients[5]  # x0^2
    b = (coefficients[1] + coefficients[6]*x1 + coefficients[7]*x2 + coefficients[8]*x3)  # x0 terms
    c = (coefficients[0] + coefficients[2]*x1 + coefficients[3]*x2 + coefficients[4]*x3 +
         coefficients[9]*x1**2 + coefficients[10]*x1*x2 + coefficients[11]*x1*x3 +
         coefficients[12]*x2**2 + coefficients[13]*x2*x3 + coefficients[14]*x3**2 +
         intercept - comb)

    discriminant = b**2 - 4*a*c
    if discriminant < 0:
        return []  # No real solutions
    elif discriminant == 0:
        return [-b / (2*a)]
    else:
        sqrt_disc = np.sqrt(discriminant)
        return [(-b + sqrt_disc) / (2*a), (-b - sqrt_disc) / (2*a)]


def print_IRAS_res(IRAS_runOnCoordinatesResults, ih, kepler3=False):

    s=0
    fontize = 10
    
    implicitPolyDict = IRAS_runOnCoordinatesResults['implicitPolyDictList'][ih]
    CR_zeta1 = implicitPolyDict['CR_zeta1']

    planet = IRAS_runOnCoordinatesResults['planet']
    singleBatch = implicitPolyDict['singleBatch']
    
    combination = implicitPolyDict['combination']
    hypotheses_regulations = implicitPolyDict['hypotheses_regulations']
    degreeOfPolyFit = IRAS_runOnCoordinatesResults['degreeOfPolyFit'][ih]
    
    pearsonCorr = pd.Series(combination[s].detach().cpu().flatten()).corr(pd.Series(hypotheses_regulations[s,:,0].flatten()))
    
    tVec = np.arange(singleBatch.shape[1])[:,None]
    comb = combination[s].detach().cpu().numpy().flatten()        
    hyp = hypotheses_regulations[s,:,0].flatten()
    #hyp = (hyp-hyp.mean())/hyp.std()
    comb = (comb-comb.mean())/comb.std()
    if pearsonCorr < 0:
        comb = -comb
    comb = comb*hyp.std() + hyp.mean()
    #print(f'hyp mean = {hyp.mean()}')
    #print(f'comb mean = {comb.mean()}')
    #print(f'degreeOfPolyFit={degreeOfPolyFit}')
    #print(f'singleBatch[s,:5]={singleBatch[s,:5]}')
    poly = PolynomialFeatures(degreeOfPolyFit)
    X_poly = poly.fit_transform(singleBatch[s])
    feature_names = poly.get_feature_names_out()
    #print(f'X_poly={X_poly}')    
            
    # Evaluate the polynomial function
    poly_comb = X_poly @ np.concatenate(([implicitPolyDict['intercept']], implicitPolyDict['coefficients'][1:]))
    pCorrOfPolyFit = np.abs(pd.Series(poly_comb).corr(pd.Series(comb)))
    pCorrOfPolyWithHyp = np.abs(pd.Series(poly_comb).corr(pd.Series(hyp)))
    titleStr = planet + f"; Polynomial Regression Equation with corr {str(round(pCorrOfPolyFit,2))} with g() and {str(round(pCorrOfPolyWithHyp,2))} with hyp{ih}; CR = {str(round(CR_zeta1.item(),2))}" + f'\ng() and hyp corr = {str(round(pearsonCorr,2))}'

    polynomial_equation = poly_eq_str(implicitPolyDict['intercept'], implicitPolyDict['coefficients'], feature_names)
    
    nPoints2Plot = np.min([100, len(hyp)])
    
    plt.figure(figsize=(16,9/2))
    plt.title(titleStr)
    
    plt.plot(tVec[:nPoints2Plot], hyp[:nPoints2Plot], 'k', label=r'$hyp$')
    plt.plot(tVec[:nPoints2Plot], comb[:nPoints2Plot],'r--', label=r'$g_\theta()$')      
    
    plt.plot(tVec[:nPoints2Plot], poly_comb[:nPoints2Plot] ,'g--', label=f'Poly deg {degreeOfPolyFit}')      
    
    plt.xlabel(r'$samples; $' + f"mean(comb) = {str(round(poly_comb.mean(),2))}; \ncomb =" + polynomial_equation, fontsize=fontize)
    #plt.ylabel('Protein level', fontsize=fontize)
    plt.legend(loc='lower left', fontsize=fontize)
    plt.xticks([])
    #plt.yticks([])
    plt.grid()
    
    plt.show(block=False)

    
    if kepler3:
        plt.figure()#figsize=(16/4,9/2/4))
        plt.scatter(x=comb, y=hyp, s=5, label=f'corr={str(round(pearsonCorr,2))}')
        plt.xlabel(r'$g(a, e, T;\theta^{*}_{3})$', fontsize=18)
        plt.ylabel(r'$2\log(T)-3\log(a)$', fontsize=18)
        plt.legend(loc='upper right', fontsize=18)
        plt.tight_layout()
        plt.show(block=False)

        #plt.savefig('/Users/ron.teichner/Library/CloudStorage/OneDrive-Technion/Kepler/third.png', dpi=300)
    return pearsonCorr

def replace_variables(input_string):
    replacements = {
        'x0': 'x',
        'x1': 'y',
        'x2': 'z'
    }
    for old, new in replacements.items():
        input_string = input_string.replace(old, new)
    return input_string

def getCoeffVal(eqStr, coeffName):
    #eqStr = OrbitParams["plane eq"].values[0]
    tmpStr = eqStr[:eqStr.find('*' + coeffName + ' ')]
    #print(f'tmpStr={tmpStr}')
    #print(f'{tmpStr[3+tmpStr.rfind(" + "):-1]}')
    coeffOrig = float(tmpStr[3+tmpStr.rfind(' + '):-1])
    return coeffOrig


def angle_between_vectors(u, v):
    u = np.array(u)
    v = np.array(v)
    dot_product = np.dot(u, v)
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    cos_theta = dot_product / (norm_u * norm_v)
    # Clip to avoid numerical issues outside [-1, 1]
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    angle_rad = np.arccos(cos_theta)
    
    return angle_rad

    
def plot_manifold(IRAS_runOnCoordinatesResults, ih, true_anomaly_values_df, orbitalObs_df, orbitalParams_df, enable_2D_cuts=False, title3D=''):
    # Colors for each planet
    colors = {
        'Mercury': 'gray',
        'Venus': 'orange',
        'Earth': 'blue',
        'Mars': 'red'
    }
    
    implicitPolyDict = IRAS_runOnCoordinatesResults['implicitPolyDictList'][ih]
    degreeOfPolyFit = IRAS_runOnCoordinatesResults['degreeOfPolyFit'][ih]
    observations2IRAS = IRAS_runOnCoordinatesResults['observations2IRAS']
    planet = IRAS_runOnCoordinatesResults['planet']
    model = true_anomaly_values_df[true_anomaly_values_df['target'] == planet] 
    Obs = orbitalObs_df[orbitalObs_df['target'] == planet]
    OrbitParams = orbitalParams_df[orbitalParams_df['target'] == planet]
    hyp_mean = implicitPolyDict['hypotheses_regulations'].mean()
    
    x_Obs, y_Obs, z_Obs = observations2IRAS[:,:,0].flatten(), observations2IRAS[:,:,1].flatten(), observations2IRAS[:,:,2].flatten()
    
    if len(implicitPolyDict['coefficients']) == 6:
        # Create a grid of x0 and x1 values
        x0_range = np.linspace(-1, 1, 400)
        x1_range = np.linspace(-1, 1, 400)
        x0, x1 = np.meshgrid(x0_range, x1_range)
        
        
        # Flatten the grid and stack into a feature matrix
        X_grid = np.vstack([x0.ravel(), x1.ravel()]).T
        
        
        # Generate polynomial features
        poly = PolynomialFeatures(degree=degreeOfPolyFit)
        X_poly = poly.fit_transform(X_grid)
        
        
        # Evaluate the polynomial function
        P_values = X_poly @ np.concatenate(([implicitPolyDict['intercept']], implicitPolyDict['coefficients'][1:]))
        P_values = P_values.reshape(x0.shape)
        
        
        # Plot the zero-level contour
        plt.figure(figsize=(8, 6))
        contour = plt.contour(x0, x1, P_values, levels=[0], colors='blue')
        #contour.collections[0].set_label('ron')
        plt.clabel(contour, inline=True, fontsize=10)
        plt.scatter(observations2IRAS[:,:,0].flatten(), observations2IRAS[:,:,1].flatten(), color=colors[planet], marker='o', s=15, label=f"{planet} obs")
        plt.scatter(0, 0, color='yellow', label='Sun', s=100)
        plt.title(planet + r"; Implicit Curve of $P(x_t, y_t) = 0$")
        plt.xlabel(r"$x_t$")
        plt.ylabel(r"$y_t$")
        plt.grid(True)
        plt.legend()
        plt.axis('equal')
        plt.show(block=False)

        
    else:
            
        # Generate polynomial features
        poly = PolynomialFeatures(degree=degreeOfPolyFit)
        poly.fit_transform(np.zeros((10,3)))
        feature_names = poly.get_feature_names_out()
        feature_names = [replace_variables(feature_name) for feature_name in feature_names]
        
        
        origCoeffVec = np.asarray([0, OrbitParams['rotation_matrix'].to_numpy()[0][0,2], OrbitParams['rotation_matrix'].to_numpy()[0][1,2], OrbitParams['rotation_matrix'].to_numpy()[0][2,2]])
        polyFitCoeffVec = np.concatenate((implicitPolyDict['intercept'][None], implicitPolyDict['coefficients'][1:]))
        
        normOrigCoeffVec = np.linalg.norm(origCoeffVec)
        normPolyFitCoeffVec = np.linalg.norm(polyFitCoeffVec)
        
        angle_rad = angle_between_vectors(origCoeffVec[1:], polyFitCoeffVec[1:])
        
        
        polynomial_equation = poly_eq_str(normOrigCoeffVec/normPolyFitCoeffVec*(implicitPolyDict['intercept']-hyp_mean), [normOrigCoeffVec/normPolyFitCoeffVec*coeff for coeff in implicitPolyDict['coefficients']], feature_names)
        print(planet + f'; GT plane eq: 0={OrbitParams["plane eq"].values[0]}; IRAS-polyFit eq: 0={polynomial_equation}; angle between normals is {str(round(np.rad2deg(angle_rad), 4))} deg')
        print(f'sanity orbital plane eq: 0={origCoeffVec}')
        
        # Create a 3D grid of x0, x1, x2 values
        grid_size = 100
        x = np.linspace(-1, 1, grid_size)
        y = np.linspace(-1, 1, grid_size)
        z = np.linspace(-0.1, 0.1, grid_size)
        x0, x1 = np.meshgrid(x, y)
        
        
        if enable_2D_cuts:
            # Plot 2D contour slices at different x2 values
            fig, axes = plt.subplots(3, 3, figsize=(15, 15))
            axes = axes.ravel()
            z_slices = np.linspace(-0.1, 0.1, 8)
            
            z_slices = np.sort(np.concatenate((z_slices, [0])))
            
            for i, z_val in enumerate(z_slices):
                # Create input grid for fixed z
                x0_flat = x0.ravel()
                x1_flat = x1.ravel()
                x2_flat = np.full_like(x0_flat, z_val)
                X_grid = np.vstack([x0_flat, x1_flat, x2_flat]).T
            
                # Transform to polynomial features
                X_poly = poly.fit_transform(X_grid)
            
                # Evaluate the polynomial
                P_values = X_poly @ np.concatenate(([implicitPolyDict['intercept']-hyp_mean], implicitPolyDict['coefficients'][1:]))
                P_values = P_values.reshape(x0.shape)
            
                # Plot contour
                cs = axes[i].contour(x0, x1, P_values, levels=[0], colors='blue')
                
                indices = i==np.argmin(np.abs(np.repeat(z_Obs[:,None], z_slices.shape[0], 1) - np.repeat(z_slices[None,:], z_Obs.shape[0], 0)), axis=1)
                axes[i].scatter(x_Obs[indices], y_Obs[indices], color=colors[planet], marker='o', s=15, label=f"{planet} obs")
                axes[i].scatter(0, 0, color='yellow', label='Sun', s=100)
                axes[i].set_title(r"$z = $" + f"{z_val:.2f}")
                axes[i].set_xlabel(r"$x$")
                axes[i].set_ylabel(r"$y$")
                axes[i].grid(True)
            
            plt.suptitle(planet + r"; 2D Contour Slices of $P(x, y, z) = 0$ at Different $z_t$ Values")
            plt.tight_layout()
            plt.show(block=False)

    
    
    
        # Create a 3D grid of x0, x1, x2 values
        #grid_size = 50
        #x = np.linspace(-5, 5, grid_size)
        #y = np.linspace(-5, 5, grid_size)
        #z = np.linspace(-5, 5, grid_size)
        x0, x1, x2 = np.meshgrid(x, y, z, indexing='ij')
    
        # Flatten the grid and stack into a feature matrix
        X_grid = np.vstack([x0.ravel(), x1.ravel(), x2.ravel()]).T
    
        # Generate polynomial features
        X_poly = poly.fit_transform(X_grid)
    
        # Evaluate the polynomial function
        P_values = X_poly @ np.concatenate(([implicitPolyDict['intercept']-hyp_mean], implicitPolyDict['coefficients'][1:]))
        P_values = P_values.reshape((grid_size, grid_size, grid_size))
    
        # Extract the zero-level isosurface using marching cubes
        verts, faces, normals, values = measure.marching_cubes(P_values, level=0, spacing=(x[1]-x[0], y[1]-y[0], z[1]-z[0]))
    
        
        # Adjust vertices to reflect actual coordinate ranges
        verts[:, 0] += x[0]
        verts[:, 1] += y[0]
        verts[:, 2] += z[0]
    
    
        # Plot the surface
        fig = plt.figure(figsize=(10, 8))
        fig.suptitle(title3D, y=0.25, fontsize=16)  # Adjust 'y' to move the title lower (default is ~0.98)
        ax = fig.add_subplot(111, projection='3d')
        mesh = ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2],
                               cmap='Spectral', lw=1, alpha=0.8)
        
        indices = np.random.permutation(np.arange(len(x_Obs)))[:30]
        
        for specificPlanet in list(colors.keys()):
            planet_model = true_anomaly_values_df[true_anomaly_values_df['target'] == specificPlanet] 
            x_model = [planet_model['r'].to_numpy()[i][0,0] for i in range(planet_model.shape[0])]
            y_model = [planet_model['r'].to_numpy()[i][1,0] for i in range(planet_model.shape[0])]
            z_model = [planet_model['r'].to_numpy()[i][2,0] for i in range(planet_model.shape[0])]
            
            ax.plot(x_model, y_model, z_model, label=f"{specificPlanet}", color=colors[specificPlanet])
        #ax.scatter(x_Obs[indices], y_Obs[indices], color=colors[planet], marker='o', s=15, label=f"{planet} obs")
        # Plot the Sun at the origin
        ax.scatter(0, 0, 0, color='yellow', label='Sun', s=100)
        
        ax.view_init(elev=30, azim=60, roll=0)

        # Set labels and title
        ax.set_xlabel("X (AU)")
        ax.set_ylabel("Y (AU)")
        ax.set_zlabel("Z (AU)")
        ax.set_zlim(-0.1, 0.1)
        ax.legend(loc='upper right', bbox_to_anchor=(0.98, 0.65))
        #ax.set_title(title3D)
        plt.tight_layout()
        plt.show(block=False)

        #plt.savefig('/Users/ron.teichner/Library/CloudStorage/OneDrive-Technion/Kepler/'+planet+'Plane.png', dpi=300)
        
    
        

def getStatistics():
    
    
    # Define planet IDs
    planets = {
        'Mercury': 199,
        'Venus': 299,
        'Earth': 399,
        'Mars': 499
    }
    
    colors = {
            'Mercury': 'gray',
            'Venus': 'orange',
            'Earth': 'blue',
            'Mars': 'red'
        }
        
    # Define epoch for orbital elements
    epoch = {'start': '2008-01-06', 'stop': '2009-01-06', 'step': '30d'}
    # Define observation time range
    obs_epochs = {'start': '1993-01-01', 'stop': '2023-01-01', 'step': '30d'}
    
    if not os.path.exists('NASA_data.pkl'):
        orbitalObs_df, orbitalParams_df, true_anomaly_values_df, multi_orbitalParams_df, multi_orbitalObs_df = get_orbital_observations(planets, epoch, obs_epochs)
        with open('NASA_data.pkl', 'wb') as file:
            pickle.dump([orbitalObs_df, orbitalParams_df, true_anomaly_values_df, multi_orbitalParams_df, multi_orbitalObs_df], file)
    else:
        dataset = pickle.load(open('NASA_data.pkl', 'rb'))
        orbitalObs_df, orbitalParams_df, true_anomaly_values_df, multi_orbitalParams_df, multi_orbitalObs_df = dataset
    
    IRAS_runOnCoordinatesResultsDict = dict()
    
    if True:
        pearsonCorr_orbitalPlaneList = list()
        for i in range(100):
            plt.close('all')
            print(f'plane {i} out of 100')
            IRAS_runOnCoordinatesResultsDict['Mercury'] = runIRAS('Mercury', true_anomaly_values_df, orbitalObs_df, orbitalParams_df, externalReport=False)
            pearsonCorr = print_IRAS_res(IRAS_runOnCoordinatesResultsDict['Mercury'], 0)
            pearsonCorr_orbitalPlaneList.append(pearsonCorr)
            print(f'plane corr is {pearsonCorr}')
            if np.abs(pearsonCorr) > 0.92:
                break;
            
            #with open('pearsonCorr_orbitalPlaneList.pkl', 'wb') as file:
            #    pickle.dump(pearsonCorr_orbitalPlaneList, file)
    if True:
        alpha = 1e-2
        orbitalParams_df['est_orbitalPlaneNormal'] = orbitalParams_df.apply(lambda row: add_est_orbitalPlaneNormal(row, 'Mercury', IRAS_runOnCoordinatesResultsDict['Mercury']['implicitPolyDictList'][0]['coefficients'][1:][:,None], 'est_orbitalPlaneNormal'), axis=1)
        orbitalObs_df['r_proj2EstOrbitalPlane'] = orbitalObs_df.apply(lambda row: proj_r_2EstOrbitalPlane(row, orbitalParams_df), axis=1)
        orbitalObs_df['r_2D_est'] = orbitalObs_df.apply(lambda row: transform_2_2D(row, orbitalParams_df, np.array([[1], [0], [0]]), est=True), axis=1)
        orbitalObs_df['r_2D_est_noisy'] = orbitalObs_df.apply(lambda row: convert_to_r(row, orbitalObs_df, orbitalParams_df, alpha, workOn2D_est=True, est=True), axis=1)
        
        pearsonCorr_ellipseList = list()
        for i in range(100):
            plt.close('all')
            print(f'ellipse {i} out of 100')
            IRAS_runOnCoordinatesResultsDict['Mercury2D'] = runIRAS('Mercury', true_anomaly_values_df, orbitalObs_df, orbitalParams_df, runOn2D=True, externalReport=False)
            pearsonCorr = IRAS_runOnCoordinatesResultsDict['Mercury2D']['implicitPolyDictList'][0]['ellipse_fit']['corr']
            CR_zeta1 = IRAS_runOnCoordinatesResultsDict['Mercury2D']['implicitPolyDictList'][0]['CR_zeta1'].item()
            pearsonCorr_ellipseList.append({'pearsonCorr':pearsonCorr, 'CR_zeta1':CR_zeta1})
            with open('pearsonCorr_ellipseList.pkl', 'wb') as file:
                pickle.dump(pearsonCorr_ellipseList, file)
    if False:
        opdf = multi_orbitalParams_df[['a','e','T']]
        opdf = opdf.copy()
        
        print(f'a.max()/a.min() = {opdf["a"].max()/opdf["a"].min()}')
        print(f'e.max()/e.min() = {opdf["e"].max()/opdf["e"].min()}')
        print(f'T.max()/T.min() = {opdf["T"].max()/opdf["T"].min()}')
        
        #opdf['a']=np.log(opdf['a'])
        #opdf['e']=np.log(opdf['e'])
        #opdf['T']=np.log(opdf['T'])
        
        observations2IRAS = np.log(opdf.to_numpy()[None])
        #observations2IRAS = opdf.to_numpy()[None]
        observations_tVec = np.repeat(np.arange(observations2IRAS.shape[1])[None,:,None], observations2IRAS.shape[0], 0) 
        hypotheses_regulations = None
        
        observations_noisy = observations2IRAS
        for featureIdx in range(observations_noisy.shape[2]):
            std = observations2IRAS[:,:,featureIdx].flatten().std()
            observations_noisy[:,:,featureIdx] = observations2IRAS[:,:,featureIdx] + 1e-2*std*np.random.randn(observations2IRAS.shape[0], observations2IRAS.shape[1])
        
        hypotheses_regulations_pearson = (2*observations_noisy[:,:,-1] - 3*observations_noisy[:,:,0])[:,:,None]
        
        nIRAS_iter=100
        IRAS_results_list = list()
        for i in range(nIRAS_iter):
            plt.close('all')
            print(f'k3 {i} out of 100')
            implicitPolyDictList = IRAS_train_script(observations2IRAS, observations_tVec, hypotheses_regulations, seriesForPearson=observations_noisy, hypothesesForPearson=hypotheses_regulations_pearson, titleStr='', nativeIRAS=True, nEpochs=500, degreeOfPolyFit=[1], onlyPolyMixTerms=False, externalReport=False, features2ShuffleTogether=None, playerPerPatient=False)
    
            IRAS_results_list.append({'planet': '', 'observations2IRAS': observations2IRAS, 'degreeOfPolyFit': [1], 'implicitPolyDictList': implicitPolyDictList})
        
        IRAS_runOnCoordinatesResultsDict['k3'] = IRAS_results_list
        CR_zeta1 = [l['implicitPolyDictList'][0]['CR_zeta1'].item() for l in IRAS_runOnCoordinatesResultsDict['k3']]
        pearsonCorr_k3List = [np.abs(pd.Series(l['implicitPolyDictList'][0]['hypotheses_regulations'].flatten()).corr(pd.Series(l['implicitPolyDictList'][0]['combination'].detach().numpy().flatten()))) for l in IRAS_runOnCoordinatesResultsDict['k3']]
        with open('pearsonCorr_k3List.pkl', 'wb') as file:
            pickle.dump(pearsonCorr_k3List, file)
    
            
