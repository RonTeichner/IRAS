#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 13:18:19 2025

@author: ron.teichner
"""

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from astroquery.jplhorizons import Horizons
import pandas as pd
import numpy as np
import torch
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
    plt.savefig('/Users/ron.teichner/Library/CloudStorage/OneDrive-Technion/Kepler/orbits.png', dpi=300)
    
    
    
    
    
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
    
    alpha = 1e-1
    orbitalObs_df['rNoisy'] = orbitalObs_df.apply(lambda row: convert_to_r(row, orbitalObs_df, orbitalParams_df, alpha), axis=1)
    orbitalObs_df['vNoisy'] = orbitalObs_df.apply(lambda row: convert_to_r(row, orbitalObs_df, orbitalParams_df, alpha, to_v=True), axis=1)
    
    orbitalObs_df['r_tag'] = orbitalObs_df.apply(lambda row: rotate_r_to_r_tag(row, orbitalParams_df, False), axis=1)
    
    orbitalObs_df['r_tagNoisy'] = orbitalObs_df.apply(lambda row: rotate_r_to_r_tag(row, orbitalParams_df, True), axis=1)
    
    true_anomaly_values_df['r'] = true_anomaly_values_df.apply(lambda row: convert_to_r(row, true_anomaly_values_df, orbitalParams_df, 0), axis=1)
    true_anomaly_values_df['r_tag'] = true_anomaly_values_df.apply(lambda row: rotate_r_to_r_tag(row, orbitalParams_df, False), axis=1)
    
    true_anomaly_values_df['v'] = true_anomaly_values_df.apply(lambda row: convert_to_r(row, true_anomaly_values_df, orbitalParams_df, 0, to_v=True), axis=1)
    
    if False:#plot velocities
        planet = 'Mercury'
        Obs = orbitalObs_df[orbitalObs_df['target'] == planet]
        plt.figure()
        plt.scatter(x=Obs['vx'], y=Obs['vy'], s=1, label=planet)
        plt.xlabel('vx')
        plt.ylabel('vy')
        plt.show()
        
    
    

    return orbitalObs_df, orbitalParams_df, true_anomaly_values_df


    

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

def transform_2_2D(row, orbitalParams_df, a, est=False):
    planet = row['target']
    OrbitParams = orbitalParams_df[orbitalParams_df['target'] == planet]
    if est:
        normal = OrbitParams['est_orbitalPlaneNormal'].to_numpy()[0]
        r_proj = row['r_proj2EstOrbitalPlane']
    else:
        normal = OrbitParams['orbitalPlaneNormal'].to_numpy()[0]
        r_proj = row['r_proj2OrbitalPlane']
    
    u = (np.cross(normal[:,0],a[:,0]) / np.linalg.norm(np.cross(normal[:,0],a[:,0])))[:,None]
    v = (np.cross(normal[:,0],u[:,0]))[:,None]
    
    alpha = (u.T@r_proj)[0,0]
    beta = (v.T@r_proj)[0,0]
    
    return np.array([[alpha], [beta]])
    
    

def convert_to_r(row, orbitalObs_df, orbitalParams_df, alpha, workOn2D_est=False, to_v=False):
    assert not(to_v and workOn2D_est)
    #print(f'{row.shape}')
    planet = row['target']
    Obs = orbitalObs_df[orbitalObs_df['target'] == planet]
    OrbitParams = orbitalParams_df[orbitalParams_df['target'] == planet]
    if alpha > 0:
        if workOn2D_est:
            alpha_c = [Obs['r_2D_est'].to_numpy()[i][0,0] for i in range(Obs.shape[0])]
            beta_c = [Obs['r_2D_est'].to_numpy()[i][1,0] for i in range(Obs.shape[0])]  
            r_2D_est = row['r_2D_est']
            #print(f'{r_2D_est.shape}')
            #print(f'r_2D_est[0,0] = {r_2D_est[0,0]}')
            #print(f'r_2D_est[1,0] = {r_2D_est[1,0]}')
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
    
def proj_r_2OrbitalPlane(row, orbitalParams_df):
    planet = row['target']
    planetParams = orbitalParams_df[orbitalParams_df['target'] == planet]
    normal = planetParams['orbitalPlaneNormal'].to_numpy()[0]
    
    return project_point_2_plane(row['r'],normal)
    
def proj_r_2EstOrbitalPlane(row, orbitalParams_df):
    planet = row['target']
    planetParams = orbitalParams_df[orbitalParams_df['target'] == planet]
    hat_n = planetParams['est_orbitalPlaneNormal'].to_numpy()[0]
    
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
    plt.show()
    
def plot_est2D_ellipse(IRAS_runOnCoordinatesResultsDict, planet, orbitalParams_df, orbitalObs_df, title=''):
    colors = {
        'Mercury': 'gray',
        'Venus': 'orange',
        'Earth': 'blue',
        'Mars': 'red'
    }
    
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
    x_vals = np.linspace(h - 5, h + 5, 400)
    y_vals = np.linspace(k - 5, k + 5, 400)
    X, Y = np.meshgrid(x_vals, y_vals)
    
    # Compute the left-hand side of the ellipse equation
    Z = A * (X - h)**2 + B * (X - h) * (Y - k) + C * (Y - k)**2
    
    # Plot the contour where the equation equals 1
    plt.contour(X, Y, Z, levels=[1], colors='blue', label='fitted ellipse for IRAS g()')
    plt.scatter(sun_coordinates_on_plane[0,0], sun_coordinates_on_plane[1,0], color='yellow', label='Sun', s=100)
    plt.scatter(x=[focci[0][0], focci[1][0]], y=[focci[0][1], focci[1][1]], marker='+', s=100, color='blue', label='focci')
    #plt.title("Ellipse: A(x−h)^2 + B(x−h)(y−k) + C(y−k)^2 = 1")
    plt.text(-0.3, 0.2, r'$\hat{a} = $' + f'{str(round(axes_eccentricity["a"],3))}   ', fontsize=14)# + r'$a = $' + f'{str(round(OrbitParams["a"][0], 3))}', fontsize=12)
    plt.text(-0.3, 0.1, r'$\hat{e} = $' + f'{str(round(axes_eccentricity["e"],3))}   ', fontsize=14)# + r'$e = $' + f'{str(round(OrbitParams["e"][0], 3))}', fontsize=12)
    #plt.title("IRAS-estimated ellipse")
    plt.xlabel(r"$x_{\hat{n}}$")
    plt.ylabel(r"$y_{\hat{n}}$")
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True)
    
    Obs = orbitalObs_df[orbitalObs_df['target'] == 'Mercury']
    alpha = np.asarray([Obs['r_2D'].to_numpy()[i][0,0] for i in range(Obs.shape[0])])
    beta = np.asarray([Obs['r_2D'].to_numpy()[i][1,0] for i in range(Obs.shape[0])])
    alpha_hat = np.asarray([Obs['r_2D_est'].to_numpy()[i][0,0] for i in range(Obs.shape[0])])
    beta_hat = np.asarray([Obs['r_2D_est'].to_numpy()[i][1,0] for i in range(Obs.shape[0])])
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

    
    plt.xlim([-0.5, 0.5])
    plt.ylim([-0.5, 0.5])
    plt.savefig('/Users/ron.teichner/Library/CloudStorage/OneDrive-Technion/Kepler/'+planet+'2D_corr_'+f'{str(round(corr,3))}'+'_.png', dpi=300)
    
    
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
            plt.show()
    
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
        plt.show()
        


def runIRAS(planet, true_anomaly_values_df, orbitalObs_df, orbitalParams_df, runOn2D=False, runCoordinates_n_Velocities=False, externalReport=False):
    assert not(runOn2D and runCoordinates_n_Velocities)
    model = true_anomaly_values_df[true_anomaly_values_df['target'] == planet] 
    Obs = orbitalObs_df[orbitalObs_df['target'] == planet]
    OrbitParams = orbitalParams_df[orbitalParams_df['target'] == planet]
    
    x_model = [model['r'].to_numpy()[i][0,0] for i in range(model.shape[0])]
    y_model = [model['r'].to_numpy()[i][1,0] for i in range(model.shape[0])]
    z_model = [model['r'].to_numpy()[i][2,0] for i in range(model.shape[0])]
    observations_model = np.concatenate((np.asarray(x_model)[:,None], np.asarray(y_model)[:,None], np.asarray(z_model)[:,None]), axis=1)[None]
    
    vx_model = [model['v'].to_numpy()[i][0,0] for i in range(model.shape[0])]
    vy_model = [model['v'].to_numpy()[i][1,0] for i in range(model.shape[0])]
    vz_model = [model['v'].to_numpy()[i][2,0] for i in range(model.shape[0])]
    v_observations_model = np.concatenate((np.asarray(vx_model)[:,None], np.asarray(vy_model)[:,None], np.asarray(vz_model)[:,None]), axis=1)[None]
    
    x_model_tag = [model['r_tag'].to_numpy()[i][0,0] for i in range(model.shape[0])]
    y_model_tag = [model['r_tag'].to_numpy()[i][1,0] for i in range(model.shape[0])]
    z_model_tag = [model['r_tag'].to_numpy()[i][2,0] for i in range(model.shape[0])]
    
    
    x = [Obs['r'].to_numpy()[i][0,0] for i in range(Obs.shape[0])]
    y = [Obs['r'].to_numpy()[i][1,0] for i in range(Obs.shape[0])]
    z = [Obs['r'].to_numpy()[i][2,0] for i in range(Obs.shape[0])]
    observations = np.concatenate((np.asarray(x)[:,None], np.asarray(y)[:,None], np.asarray(z)[:,None]), axis=1)[None]
    
    vx = [Obs['v'].to_numpy()[i][0,0] for i in range(Obs.shape[0])]
    vy = [Obs['v'].to_numpy()[i][1,0] for i in range(Obs.shape[0])]
    vz = [Obs['v'].to_numpy()[i][2,0] for i in range(Obs.shape[0])]
    v_observations = np.concatenate((np.asarray(vx)[:,None], np.asarray(vy)[:,None], np.asarray(vz)[:,None]), axis=1)[None]
    
    if runOn2D:
        alpha = [Obs['r_2D_est'].to_numpy()[i][0,0] for i in range(Obs.shape[0])]
        beta = [Obs['r_2D_est'].to_numpy()[i][1,0] for i in range(Obs.shape[0])]
        observations2D_est = np.concatenate((np.asarray(alpha)[:,None], np.asarray(beta)[:,None]), axis=1)[None]
    
    observations_tVec = np.arange(len(x))[None,:,None]
    
    x_tag = [Obs['r_tag'].to_numpy()[i][0,0] for i in range(Obs.shape[0])]
    y_tag = [Obs['r_tag'].to_numpy()[i][1,0] for i in range(Obs.shape[0])]
    z_tag = [Obs['r_tag'].to_numpy()[i][2,0] for i in range(Obs.shape[0])]
    
    observations_tag = np.concatenate((np.asarray(x_tag)[:,None], np.asarray(y_tag)[:,None], np.asarray(z_tag)[:,None]), axis=1)[None]
    
    a = OrbitParams['a'].to_numpy()[0]
    e = OrbitParams['e'].to_numpy()[0]
    b = a*np.sqrt(1 - np.power(e,2))
    
    hypotheses_regulation_plane = np.asarray(z_tag)[None,:,None]
    hypotheses_regulation_ellipse = (np.power(x_tag + a*e, 2) / np.power(a, 2) + np.power(y_tag, 2) / np.power(b, 2))[None,:,None]

    hypotheses_regulations_model = (np.power(x_model_tag + a*e, 2) / np.power(a, 2) + np.power(y_model_tag, 2) / np.power(b, 2))[None,:,None]
    
    x_noisy = [Obs['rNoisy'].to_numpy()[i][0,0] for i in range(Obs.shape[0])]
    y_noisy = [Obs['rNoisy'].to_numpy()[i][1,0] for i in range(Obs.shape[0])]
    z_noisy = [Obs['rNoisy'].to_numpy()[i][2,0] for i in range(Obs.shape[0])]
    
    
    vx_noisy = [Obs['vNoisy'].to_numpy()[i][0,0] for i in range(Obs.shape[0])]
    vy_noisy = [Obs['vNoisy'].to_numpy()[i][1,0] for i in range(Obs.shape[0])]
    vz_noisy = [Obs['vNoisy'].to_numpy()[i][2,0] for i in range(Obs.shape[0])]
    v_observations_noisy = np.concatenate((np.asarray(vx_noisy)[:,None], np.asarray(vy_noisy)[:,None], np.asarray(vz_noisy)[:,None]), axis=1)[None]
    
    if runOn2D:
        alpha_noisy = [Obs['r_2D_est_noisy'].to_numpy()[i][0,0] for i in range(Obs.shape[0])]
        beta_noisy = [Obs['r_2D_est_noisy'].to_numpy()[i][1,0] for i in range(Obs.shape[0])]
        observations_noisy = np.concatenate((np.asarray(alpha_noisy)[:,None], np.asarray(beta_noisy)[:,None]), axis=1)[None]
    elif runCoordinates_n_Velocities:
        observations_noisy = np.concatenate((np.asarray(x_noisy)[:,None], np.asarray(y_noisy)[:,None], np.asarray(z_noisy)[:,None], np.asarray(vx_noisy)[:,None], np.asarray(vy_noisy)[:,None], np.asarray(vz_noisy)[:,None]), axis=1)[None]
    else:
        observations_noisy = np.concatenate((np.asarray(x_noisy)[:,None], np.asarray(y_noisy)[:,None], np.asarray(z_noisy)[:,None]), axis=1)[None]
    
    
    x_tag_noisy = [Obs['r_tagNoisy'].to_numpy()[i][0,0] for i in range(Obs.shape[0])]
    y_tag_noisy = [Obs['r_tagNoisy'].to_numpy()[i][1,0] for i in range(Obs.shape[0])]
    z_tag_noisy = [Obs['r_tagNoisy'].to_numpy()[i][2,0] for i in range(Obs.shape[0])]
    
    if runOn2D or runCoordinates_n_Velocities:
        hypotheses_regulations_plane_noisy, hypotheses_regulations_ellipse_noisy, hypotheses_regulations_pearson = None, None, None
    else:
        hypotheses_regulations_plane_noisy = np.asarray(z_tag_noisy)[None,None,:,None]
        hypotheses_regulations_ellipse_noisy = (np.power(x_tag_noisy + a*e, 2) / np.power(a, 2) + np.power(y_tag_noisy, 2) / np.power(b, 2))[None,None,:,None]
        hypotheses_regulations_pearson = np.concatenate((hypotheses_regulations_plane_noisy, hypotheses_regulations_ellipse_noisy), axis=0)
    
    observations_tag_noisy = np.concatenate((np.asarray(x_tag_noisy)[:,None], np.asarray(y_tag_noisy)[:,None], np.asarray(z_tag_noisy)[:,None]), axis=1)[None]
    
    #indices = np.random.permutation(np.arange(observations.shape[1]))
    #print(f'{indices}')
    #observations, observations_tVec, hypotheses_regulations = observations[:,indices], observations_tVec[:,indices], hypotheses_regulations[:,indices]
    
    
    #print(f'{observations[0,:3]}')
    
    #observations = observations[:,:120].reshape(3, 40, observations.shape[2])
    #print(f'{observations[0,:3]}')
    #observations_tVec = observations_tVec[:,:120].reshape(3, 40, 1)
    #hypotheses_regulations = hypotheses_regulations[:,:120].reshape(3, 40, 1)
    
    
    
    
    if runOn2D:
        observations2IRAS = observations2D_est
        degreeOfPolyFit_pearson = [2]
        features2ShuffleTogether = [[0], [1]]
    elif runCoordinates_n_Velocities:
        observations2IRAS = np.concatenate((observations, v_observations), axis=-1)
        degreeOfPolyFit_pearson = [1]
        features2ShuffleTogether = [[0,1,2], [3,4,5]]
    else:
        observations2IRAS = observations
        degreeOfPolyFit_pearson = [1,2]
        features2ShuffleTogether = [[0], [1], [2]]

    nIRAS_iter = 10
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
    
    for i in range(nIRAS_iter):
        implicitPolyDictList = IRAS_train_script(observations2IRAS, observations_tVec, hypotheses_regulations_plane_noisy, seriesForPearson=observations_noisy, hypothesesForPearson=hypotheses_regulations_pearson, titleStr=planet, nativeIRAS=True, nEpochs=500, degreeOfPolyFit=degreeOfPolyFit_pearson, externalReport=externalReport, features2ShuffleTogether=features2ShuffleTogether)
        if implicitPolyDictList[0]['CR_zeta1'] < min_CR_naive:
            minCR_implicitPolyDictList = implicitPolyDictList
            min_CR_naive = implicitPolyDictList[0]['CR_zeta1']
    
    if runOn2D:
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
                 
                 #minCR_implicitPolyDict['ellipse_fit']['rotated_ellipse'] = fit_ellipse_analytic(torch.tensor((minCR_implicitPolyDict['singleBatch'] @ minCR_implicitPolyDict['singleBatch'].numpy()[:,:,:,None])[:,:,:,0]), minCR_implicitPolyDict['combination'], minCR_implicitPolyDict['intercept'], minCR_implicitPolyDict['coefficients'])
             

    
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
        

    
def print_IRAS_res(IRAS_runOnCoordinatesResults, ih):

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
    titleStr = planet + f"; Polynomial Regression Equation with corr {str(round(pCorrOfPolyFit,2))} with g() and {str(round(pCorrOfPolyWithHyp,2))} with hyp{ih}; CR = {str(round(CR_zeta1.item(),2))}"

    polynomial_equation = poly_eq_str(implicitPolyDict['intercept'], implicitPolyDict['coefficients'], feature_names)
    
    nPoints2Plot = np.min([100, len(hyp)])
    
    plt.figure(figsize=(16,9/2))
    plt.title(titleStr)
    
    plt.plot(tVec[:nPoints2Plot], hyp[:nPoints2Plot], 'k', label=r'$hyp$')
    plt.plot(tVec[:nPoints2Plot], comb[:nPoints2Plot],'r--', label=r'$g_\theta()$')      
    
    plt.plot(tVec[:nPoints2Plot], poly_comb[:nPoints2Plot] ,'g--', label=f'Poly deg {degreeOfPolyFit}')      
    
    plt.xlabel(r'$samples; $' + f"mean(comb) = {str(round(poly_comb.mean(),2))}; comb =" + polynomial_equation, fontsize=fontize)
    #plt.ylabel('Protein level', fontsize=fontize)
    plt.legend(loc='lower left', fontsize=fontize)
    plt.xticks([])
    #plt.yticks([])
    plt.grid()
    
    plt.show()

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
        plt.show()
        
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
            plt.show()
    
    
    
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
        plt.savefig('/Users/ron.teichner/Library/CloudStorage/OneDrive-Technion/Kepler/'+planet+'Plane.png', dpi=300)
        
    
        

    


    
