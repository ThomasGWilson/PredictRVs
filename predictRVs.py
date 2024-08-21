# coding: utf-8

import juliet
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import numpy as np
from pyBGLS import bglsconst, bglsfreq, gls
import radvel
from tqdm import tqdm

################################################################################################################

def RV_model(nplanets, results, mod_times, kep_type='med'):
    '''
    A function to produce radvel summed Keplerian models for one or more planets from the output of a Juliet analysis.
    
    nplanets = number of planets as an integer
    result = RV analysis results as a Juliet results object
    mod_times = times over which to produce the Keplerian model as a 1D array
    kep_type = selection of the nominal, or upper or lower bounds of the RV analysis results as a string
    
    returns the Keplerian model 1D array
    '''
    synth_params = radvel.Parameters(nplanets, basis='per tc e w k')

    if nplanets > 0:
        if kep_type == 'med':
            indx = 0
        elif kep_type == 'upp':
            indx = 1
        elif kep_type == 'low':
            indx = 2
        else:
            indx = 0
        for h in np.arange(nplanets):
            synth_params['per'+str(h+1)] = radvel.Parameter(value = juliet.utils.get_quantiles(results.posteriors['posterior_samples']['P_p'+str(h+1)])[0], vary = False)
            synth_params['tc'+str(h+1)] = radvel.Parameter(value = juliet.utils.get_quantiles(results.posteriors['posterior_samples']['t0_p'+str(h+1)])[0], vary = False)
            synth_params['e'+str(h+1)] = radvel.Parameter(value = juliet.utils.get_quantiles(results.posteriors['posterior_samples']['ecc_p'+str(h+1)])[0], vary = False)
            synth_params['w'+str(h+1)] = radvel.Parameter(value = juliet.utils.get_quantiles(results.posteriors['posterior_samples']['omega_p'+str(h+1)])[0], vary = False)
            synth_params['k'+str(h+1)] = radvel.Parameter(value = juliet.utils.get_quantiles(results.posteriors['posterior_samples']['K_p'+str(h+1)])[indx], vary = False)

    synth_params['dvdt'] = radvel.Parameter(value=0, vary = False)
    synth_params['curv'] = radvel.Parameter(value=0, vary = False)
    synth_model = radvel.RVModel(params=synth_params)
    y_rv = synth_model(mod_times)

    return y_rv

###########################################################################################################################################

def ecdf(a):
    '''
    A function to conduct a cumulative distribution function analysis of an array.
    
    a = data array as a 1D array

    returns the data cumulative distribution function
    '''
    x, counts = np.unique(a, return_counts=True)
    cusum = np.cumsum(counts)
    y = cusum / cusum[-1]
    x = np.insert(x, 0, x[0])
    y = np.insert(y, 0, 0.)

    return y[1:]

###########################################################################################################################################

def sim_Data(times,errors,jitter,offset,nseasons,mod_times,rv_mod_med,rv_mod_upp,rv_mod_low,year_d):
    '''
    A function to produce simulted radial velocity data for a defined number of seasons using previous observations and results of a Juliet analysis of the data projected over the requested time.
    
    times = times of observed RVs as a 1D array
    errors = errors of observed RVs as a 1D array
    jitter = RV scatter result from Juliet analysis as a float
    offset = number of seasons between first observed RVs and beginning of simulated data as an integer
    nseasons = number of seasons of simulated data to be covered as an integer
    mod_times = times over which to produce the Keplerian model as a 1D array
    rv_mod_med = Keplerian model of the nominal of the RV analysis results as a string
    rv_mod_upp = Keplerian model of the upper bounds of the RV analysis results as a string
    rv_mod_low = Keplerian model of the lower bounds of the RV analysis results as a string
    year_d = constant year length as a float
    
    returns the simulated times, radial velocity, and radial velocity error arrays
    '''
    if nseasons > offset:
        sim_times, sim_rves = [], []
        for h in np.arange(nseasons):
            times_CDF = ecdf(times)
            sim_times_temp = np.array([sorted(times)[np.argmin(abs(times_CDF-i))] for i in np.random.rand(len(times))])+((offset+h)*year_d)
            sim_times = np.append(sim_times,sim_times_temp)

            rve_CDF = ecdf(errors)
            sim_rves_temp = np.array([sorted(errors)[np.argmin(abs(rve_CDF-i))] for i in np.random.rand(len(errors))])
            sim_rves = np.append(sim_rves,sim_rves_temp)
    else:
        times_CDF = ecdf(times)
        sim_times = np.array([sorted(times)[np.argmin(abs(times_CDF-i))] for i in np.random.rand(len(times))])+(offset*year_d)
    
        rve_CDF = ecdf(errors)
        sim_rves = np.array([sorted(errors)[np.argmin(abs(rve_CDF-i))] for i in np.random.rand(len(errors))])
        
    pred_rv_med = np.array([rv_mod_med[np.argmin(abs(i-mod_times))] for i in sim_times])
    pred_rv_upp = np.array([rv_mod_upp[np.argmin(abs(i-mod_times))] for i in sim_times])
    pred_rv_low = np.array([rv_mod_low[np.argmin(abs(i-mod_times))] for i in sim_times])
    
    rv_unc = np.hypot(np.nanmean(errors),jitter)
    pred_mod_unc = np.array([max(abs(pred_rv_upp[index]-pred_rv_med[index]),abs(pred_rv_med[index]-pred_rv_low[index])) for index, i in enumerate(pred_rv_med)])
    pred_comb_unc = np.array([np.hypot(i,rv_unc) for i in pred_mod_unc])
    sim_rvs = np.array([np.random.normal(pred_rv_med[index],pred_comb_unc[index]) for index, i in enumerate(pred_rv_med)])
    
    season_mask = (sim_times < min(sim_times)+nseasons*year_d)
    return sim_times[season_mask], sim_rvs[season_mask], sim_rves[season_mask]

###########################################################################################################################################

def periodogram(farray, t, d, e, n):
    '''
    A function to conduct a BGLS periodogram of data.

    farray = set of frequencies to conduct the periodogram over as a 1D array
    t = RV times as a 1D array
    d = RV data as a 1D array
    e = RV errors as a 1D array
    n = number of observations as an integer
 
    returns the RV semi-amplitude values and errors, and periodogram power and log likelihood 1D arrays
    '''
    W, Y, YYh = bglsconst(t, d, e)
    K = [np.float_(x) for x in range(0)]
    Kerr = [np.float_(x) for x in range(0)]
    power = [np.float_(x) for x in range(0)]
    logl = [np.float_(x) for x in range(0)]
    for f in farray:
        result = gls(t[:n], d[:n], e[:n], f, W, Y)
        K.append(result[0])
        power.append(result[1])
        Kerr.append(result[2])
        result2 = bglsfreq(t[:n], d[:n], e[:n], f, W, Y)
        logl.append(result2)

    return K, Kerr, power, logl

###########################################################################################################################################

def stackBGLS(pmin, pmax, t, d, e, oversamp=4.):
    '''
    A function to conduct a stacked BGLS periodogram of data.

    pmin = minimum period to conduct the periodogram over as an integer or float
    pmax = maximum period to conduct the periodogram over as an integer or float
    t = RV times as a 1D array
    d = RV data as a 1D array
    e = RV errors as a 1D array
    oversamp = value to over-sample the period and frequency arrays as an integer or float 
 
    returns the RV semi-amplitude values and errors, and periodogram power and log likelihood 2D arrays, and period 1D array
    '''    
    fs = 1/pmax
    fe = 1/pmin
        
    df = 1/oversamp/(np.max(t) - np.min(t))
    dP = df/fe/fe

    nf = np.int64(0.5+(pmax-pmin)/dP)
    Parray = np.linspace(pmin,pmax,nf)
    farray = 1/Parray
    
    K, Kerr, power, logl = periodogram(farray, t, d, e, 3)
    K2D = [K]
    Kerr2D = [Kerr]
    power2D = [power]
    logl2D = [logl]
    for i in tqdm(range(4,len(t)),desc="stack"):
        K, Kerr, power, logl = periodogram(farray, t, d, e, i)
        K2D.append(K)
        Kerr2D.append(Kerr)
        power2D.append(power)
        logl2D.append(logl)
        
    return K2D, Kerr2D, power2D, logl2D, Parray

###########################################################################################################################################

def predict_RVs(bjd,rv,rve,T0,P,K,ecc,omega,nseasons,pmin,pmax):
    '''
    A function to produce simulated radial velocities using observed data. A Keplerian model is produced from a Juliet analysis to the data that is propagated into the future and from which the simulated radial velocities are drawn. A stacked BGLS periodogram is conducted to retrieve the SNR of planet semi-amplitudes over the range of simulated data. Plots of the observed and simulated radial velocities, the stacked BGLS periodogram, and SNR estimates are shown. 

    bjd = RV times as a 1D array
    rv = RV data as a 1D array
    rve = RV errors as a 1D array
    T0 = transit centre time of a planet for use in Juliet analysis as a 1D array of ufloat objects
    P = orbital period of a planet for use in Juliet analysis as a 1D array of ufloat objects
    K = semi-amplitude of a planet for use in Juliet analysis as a 1D array of ufloat objects
    ecc = orbital eccentricity of a planet for use in Juliet analysis as a 1D array of ufloat objects
    omega = orbital argument of periastron of a planet for use in Juliet analysis as a 1D array of ufloat objects
    nseasons = number of seasons of simulated data to be covered as an integer    
    pmin = minimum period to conduct the periodogram over as an integer or float
    pmax = maximum period to conduct the periodogram over as an integer or float    
    ''' 
    instrument = 'HARPSN'
    priors = {}
    
    params = ['mu_'+instrument,'sigma_w_'+instrument]
    dists = ['uniform', 'loguniform']
    hyperps = [[np.median(rv)-(5*np.std(rv)),np.median(rv)+(5*np.std(rv))], [1e-3, 100]]
    
    if len(P) > 0:
        for h in np.arange(len(P)):
            params.append('P_p'+str(h+1))
            dists.append('normal') 
            hyperps.append([P[h].n,P[h].s])
            params.append('t0_p'+str(h+1))
            dists.append('normal') 
            hyperps.append([T0[h].n,T0[h].s])
            params.append('K_p'+str(h+1))
            dists.append('normal') 
            hyperps.append([K[h].n,K[h].s])
            params.append('ecc_p'+str(h+1))
            dists.append('normal') 
            hyperps.append([ecc[h].n,ecc[h].s])
            params.append('omega_p'+str(h+1))
            dists.append('normal') 
            hyperps.append([omega[h].n,omega[h].s])
    
    # Populate the priors dictionary:
    for param, dist, hyperp in zip(params, dists, hyperps):
        priors[param] = {}
        priors[param]['distribution'], priors[param]['hyperparameters'] = dist, hyperp
    
    times, rvs, rv_errors = {},{},{}
    times[instrument], rvs[instrument], rv_errors[instrument] = bjd,rv,rve
    
    out_folder = 'TEMP_rvs'
    dataset = juliet.load(priors = priors, t_rv = times, y_rv = rvs, yerr_rv = rv_errors, out_folder = out_folder)
    results = dataset.fit(n_live_points = 1000, sampler='dynesty')
    
    ###########################################################################################################################################

    year_d = 365.2422
    offset = math.ceil((np.max(dataset.times_rv['HARPSN'])-np.min(dataset.times_rv['HARPSN']))/year_d)
    min_time, max_time = np.min(dataset.times_rv['HARPSN'])-30, np.min(dataset.times_rv['HARPSN'])+((offset+nseasons)*year_d)
    x_rv  = np.linspace(min_time,max_time,int((max_time-min_time)*24*60))
    
    y_rv_med = RV_model(len(P), results, x_rv, kep_type='med')
    y_rv_upp = RV_model(len(P), results, x_rv, kep_type='upp')
    y_rv_low = RV_model(len(P), results, x_rv, kep_type='low')
    
    jitter = np.median(results.posteriors['posterior_samples']['sigma_w_'+instrument])
    mu = np.median(results.posteriors['posterior_samples']['mu_'+instrument])
    
    sim_times, sim_rvs, sim_rves = sim_Data(bjd,rve,jitter,offset,nseasons,x_rv,y_rv_med,y_rv_upp,y_rv_low,year_d)
    
    ###########################################################################################################################################
    
    fig1 = plt.figure(constrained_layout=True,figsize=(15,10))
    gs = fig1.add_gridspec(4, 4)
    ax1 = fig1.add_subplot(gs[:-1, :])
    plt.subplots_adjust(wspace=0, hspace=0)
    
    colors = ['cornflowerblue','orangered']
    
    ax1.errorbar(dataset.times_rv[instrument]-2457000,dataset.data_rv[instrument]-mu,\
                 yerr = dataset.errors_rv[instrument],fmt='o',\
                 mec=colors[0], ecolor=colors[0], mfc=colors[0], label='Observed data',\
                 alpha = 1, zorder=5, ms=10)
    
    ax1.errorbar(sim_times-2457000,sim_rvs,\
                 yerr = sim_rves,fmt='o',\
                 mec=colors[1], ecolor=colors[1], mfc=colors[1], label='Simulated data',\
                 alpha = 1, zorder=5, ms=10)
    
    ax1.plot(x_rv-2457000, y_rv_med, lw=1, c='black')
    ax1.fill_between(x_rv-2457000, y_rv_upp, y_rv_low, color='grey',alpha=0.5,zorder=1)
    
    #####################################################################################################
    
    ax2 = fig1.add_subplot(gs[-1:, :])
    plt.subplots_adjust(wspace=0, hspace=0)
    
    fitted_kep_real = []
    for jindex, j in enumerate(dataset.times_rv[instrument]):    
        fitted_kep_real.append(y_rv_med[min(range(len(x_rv)), key=lambda i: abs(x_rv[i]-j))])
    
    ax2.errorbar(dataset.times_rv[instrument]-2457000,dataset.data_rv[instrument]-mu-fitted_kep_real,\
                 yerr = dataset.errors_rv[instrument],fmt='o',\
                 mec=colors[0], ecolor=colors[0], mfc=colors[0],\
                 alpha = 1, zorder=5, ms=10)
    
    fitted_kep_sim = []
    for jindex, j in enumerate(sim_times):    
        fitted_kep_sim.append(y_rv_med[min(range(len(x_rv)), key=lambda i: abs(x_rv[i]-j))])
        
    ax2.errorbar(sim_times-2457000,sim_rvs-fitted_kep_sim,\
                 yerr = sim_rves,fmt='o',\
                 mec=colors[1], ecolor=colors[1], mfc=colors[1],\
                 alpha = 1, zorder=5, ms=10)
    
    ax2.plot([min_time-2457000, max_time-2457000], [0,0], color='black', ls="--", lw=3, zorder=1)
    
    #####################################################################################################
    
    ax1.set_ylabel('Radial Velocity (m/s)', fontsize=24)
    ax1.set_xlim([min_time-2457000, max_time-2457000]) 
    
    ax1.tick_params(axis='both', labelsize=24)
    ax1.tick_params(axis="x", direction="inout", length=16, width=2, which='major', bottom=True, top=True)
    ax1.tick_params(axis="y", direction="inout", length=10, width=2, which='major', left=True, right=True)
    ax1.tick_params(axis="x", direction="inout", length=8, width=1, which='minor', bottom=True, top=True)
    ax1.tick_params(axis="y", direction="inout", length=5, width=1, which='minor', left=True, right=True)
    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    ax1.yaxis.set_minor_locator(AutoMinorLocator());
    ax1.legend()
    
    ax2.set_ylabel('Residuals (m/s)', fontsize=24)
    ax2.set_xlabel('Time (BJD - 2457000)', fontsize=24)
    ax2.set_xlim([min_time-2457000, max_time-2457000]) 
    
    ax2.tick_params(axis='both', labelsize=24)
    ax2.tick_params(axis="x", direction="inout", length=16, width=2, which='major', bottom=True, top=True)
    ax2.tick_params(axis="y", direction="inout", length=10, width=2, which='major', left=True, right=True)
    ax2.tick_params(axis="x", direction="inout", length=8, width=1, which='minor', bottom=True, top=True)
    ax2.tick_params(axis="y", direction="inout", length=5, width=1, which='minor', left=True, right=True)
    ax2.xaxis.set_minor_locator(AutoMinorLocator())
    ax2.yaxis.set_minor_locator(AutoMinorLocator())
    
    ###########################################################################################################################################
    
    bjd_all = np.append(bjd,sim_times)
    rv_all = np.append(rv,sim_rvs)
    rve_all = np.append(rve,sim_rves)
    
    w = 1/rve_all/rve_all
    unit = np.ones_like(w)
    t = bjd_all - np.dot(w,bjd_all)/np.dot(w,unit)
    d = rv_all - np.dot(w,rv_all)/np.dot(w,unit)
    
    K2D, Kerr2D, power2D, logl2D, Parray = stackBGLS(pmin, pmax, t, d, rve_all, oversamp=4.)
    
    ###########################################################################################################################################
    
    nobs,nvel=np.shape(power2D)
    
    left = Parray[0]
    right = Parray[-1]
    bottom = 0
    top = len(t)
    
    plt.figure(constrained_layout=True,figsize=(16,9))
    
    plt.imshow(np.array(logl2D),origin='lower',aspect="auto",extent=(left,right,bottom,top))
    
    cbar = plt.colorbar(pad=0.025, format='%.2f')
    cbar.ax.tick_params(labelsize=24)
    cbar.set_label('log Likelihood', fontsize=24)
    plt.xlabel(r'Period [d]', fontsize=24)
    plt.ylabel(r'Observation number', fontsize=24)
    plt.ylim(3,top)
    
    plt.xscale('log')
    plt.tick_params(axis='both', labelsize=24)
    plt.tick_params(axis="x", direction="inout", length=16, width=2, which='major', bottom=True, top=True)
    plt.tick_params(axis="y", direction="inout", length=10, width=2, which='major', left=True, right=True)
    plt.tick_params(axis="x", direction="inout", length=8, width=1, which='minor', bottom=True, top=True)
    plt.tick_params(axis="y", direction="inout", length=5, width=1, which='minor', left=True, right=True);
    
    ###########################################################################################################################################
    
    for j in P:
    
        BGLS_period = Parray[(Parray>(j.n-0.5)) & (Parray<(j.n+0.5))][np.argmax(np.array(logl2D[-1])[(Parray>(j.n-0.5)) & (Parray<(j.n+0.5))])]
        BGLS_SNR = np.array([i[np.argmin(abs(Parray-BGLS_period))] for i in K2D])/np.array([i[np.argmin(abs(Parray-BGLS_period))] for i in Kerr2D])
    
        fig1 = plt.figure(constrained_layout=True,figsize=(10,6))
        ax1 = fig1.add_subplot()
        ax1.plot(np.linspace(0,len(BGLS_SNR),len(BGLS_SNR)),BGLS_SNR,"ko",label='P = ' + str(np.around(BGLS_period,decimals=2)) + ' d')
        ax1.plot([len(bjd),len(bjd)],[0,max(BGLS_SNR)],'r-')
        ax1.set_ylim(0,max(BGLS_SNR))
        ax1.set_xlabel("Number of Observations", fontsize=24)
        ax1.set_ylabel("SNR", fontsize=24)
        ax1.legend()
        
        ax1.tick_params(axis='both', labelsize=24)
        ax1.tick_params(axis="x", direction="inout", length=16, width=2, which='major', bottom=True, top=True)
        ax1.tick_params(axis="y", direction="inout", length=10, width=2, which='major', left=True, right=True)
        ax1.tick_params(axis="x", direction="inout", length=8, width=1, which='minor', bottom=True, top=True)
        ax1.tick_params(axis="y", direction="inout", length=5, width=1, which='minor', left=True, right=True)
        ax1.xaxis.set_minor_locator(AutoMinorLocator())
        ax1.yaxis.set_minor_locator(AutoMinorLocator());

################################################################################################################
