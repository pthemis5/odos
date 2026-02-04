import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from astropy.time import Time
from astropy import units as u
from astropy.coordinates import SkyCoord, EarthLocation
from astroplan import FixedTarget, Observer, is_always_observable
from astroplan import PhaseConstraint, AtNightConstraint, AltitudeConstraint, MoonSeparationConstraint
import warnings
from scipy.optimize import curve_fit
from .random_funcs import *
from datetime import datetime


class UnknownPeriodPlanet():
    """
    A class to help find period aliases for few - transits exoplanets

    ...

    Attributes
    ----------
    possible_period_values : array
        array of possible periods of the planet
    
        
    Methods
    ---------
    possible_periods()
        calculates the possible periods of the planet


    """
    def __init__(self,name, t_midtimes = [], P_min = 1, transit_duration = 0, transit_depth = 0):
        """
        Parameters
        ----------
        name : str
            name of the planet
        t_mid : array like
            midtimes of the transits
            >>> if len == 1, considered as period
            >>> if len == 2, subtrackts them to find the maximum possible period
            >>> if len == 3, performs linear square to find the maximum period
        """
        self.name = name

        self.t_midtimes = t_midtimes
        
        
        #we might have gotten half a transit at the edge of the period
        self.transit_edge_limit = 0
        self.observation_times = []
        self.observation_times_references = []
        self.P_max = None
        self.reference_midtime = None
        self.P_min = P_min
        self.observations = {}
        self.prior_prob_period = None
        self.total_prob_period = None
        self.transit_duration = transit_duration
        self.transit_depth = transit_depth
        self.N_transits = len(self.t_midtimes)
        # self.possible_periods = None

    # @property
    # def possible_periods(self):
    #     return self._possible_periods

    #so, we want to define a function, that takes 2 lists, one of transit midtimes, ther errors, and returns all the possible periods of the planet


            
    
    # @possible_periods.setter
    def get_continuous_obs_times(self,time, binning):
        # we assume that the array is shorted by time
        if binning is None:
            binning = np.median(np.diff(time))
        diffrences = np.diff(time)
        
        #we will assume that there is a new observation time when we miss more than 2 data_points
        split_points = np.where(diffrences>20*binning)[0]
        if len(split_points) == 0:
            self.observation_times.append([np.min(time), np.max(time)])
            self.observation_times_references.append(self.last_obs_added_key)
        
        else:
            split_points = np.append(np.array([-1]),split_points)
            split_points = np.append(split_points, np.array([len(time)-1]))
            for split_point in range(len(split_points)-1):
                self.observation_times.append([time[split_points[split_point]+1], time[split_points[split_point+1]]])
                self.observation_times_references.append(self.last_obs_added_key)


        
    ## this function should be depricated, shoud use add_lc_observations from flexoplanet
    def add_observation(self, time, flux, flux_err, binning = None, plot_label = None):
        
        if binning is None:
            binning = np.median(np.diff(time))
            print('Binning is not provided, assumed median binning of {:.2f} minutes'.format(binning*24*60))
        try:
            obs_id = np.max(list(self.observations.keys())) + 1
            if plot_label is None:
                plot_label = 'obs_id ' + str(obs_id)
            self.observations[obs_id] = {'time':time, 'flux':flux, 'flux_err':flux_err, 'binning':binning, 'plot_label':plot_label}
        except:
            plot_label = 'obs_id 0'
            self.observations[0] = {'time':time, 'flux':flux, 'flux_err':flux_err, 'binning':binning, 'plot_label':plot_label}
        
        self.last_obs_added_key = list(self.observations.keys())[-1]
        self.get_continuous_obs_times(time, binning)


    
    def propose_transits(self, plot_obs_id = None):

        if self.P_max is None or self.reference_midtime is None:
            raise ValueError('in order to propose transits provide a P_max and a reference midtime PLEASE!!!!!!!!!!')
        p = self.P_max[0]
        t0 = self.reference_midtime[0]

        possible_mids = []

        for i, obs_id in enumerate(self.observations.keys()):
            mint = np.min(self.observations[obs_id]['time'])
            maxt = np.max(self.observations[obs_id]['time'])
            N_min = int((mint-t0)/p)
            N_max = int((maxt-t0)/p)+1
            epochs = np.arange(N_min, N_max+1)
            possible_tmids_sector = []
            for ep in epochs:
                tmid = lin_eph(p, t0, ep)
                if tmid < maxt and tmid > mint:
                    print('Obs id: {} Transit at epoch {} at time {}'.format(obs_id, ep, tmid))
                    possible_tmids_sector.append(tmid)
                    possible_mids.append([tmid, lin_eph_err(self.P_max[1], self.reference_midtime[1], ep)])
            if plot_obs_id is not None:
                if obs_id == plot_obs_id:
                    fig = plt.figure(figsize = (13,5))
                    ax = fig.add_subplot(111)
                    ax.set(title = 'Proposed transits', xlabel = 'Time', ylabel = 'Flux')
                    ax.scatter(self.observations[obs_id]['time'], self.observations[obs_id]['flux'], c = 'black', s = 1)
                    for tmid in possible_tmids_sector:
                        ax.axvspan(tmid-self.transit_duration/2, tmid+self.transit_duration/2, alpha = 0.1, color = 'r')
                    plt.show()

        self.possible_midtimes = possible_mids


    def autoset_midtimes(self):
        if self.transit_duration == 0 or self.transit_depth == 0:
            raise ValueError('You have to provide the transit duration AND the transit depth to crosscorelate the boxcar function with the data')
        crossc_midtimes = []
        for obs_id in self.observations.keys():
            time_tot = self.observations[obs_id]['time']
            flux_tot = self.observations[obs_id]['flux']

            cadence_time = self.observations[obs_id]['binning']
            #consider one hour of baseline
            N_points_out = int(1/24/cadence_time)
            out_transit_flux = np.ones(N_points_out)

            N_points_in = int(self.transit_duration/cadence_time)
            in_transit_flux = np.ones(N_points_in) - self.transit_depth
            
            boxcar = np.concatenate((out_transit_flux, in_transit_flux, out_transit_flux))
            time_boxcar = np.arange(0, len(boxcar)*cadence_time, cadence_time)
            #in theory now we move the midtime as zero in the middle of the observation
            time_boxcar = time_boxcar - time_boxcar[int(len(boxcar)/2)]

            for pos_midtime in self.possible_midtimes:
                midtime = pos_midtime[0]

                cross_correlation = []
                if midtime > np.min(time_tot) and midtime < np.max(time_tot):
                    min_time = midtime - 3*self.transit_duration
                    max_time = midtime + 3*self.transit_duration
                    corr_indx_tot = (time_tot > min_time) * (time_tot < max_time)
                    time_corr = time_tot[corr_indx_tot]
                    flux_corr = flux_tot[corr_indx_tot]

                    for i_tmid in time_corr:
                        time_boxcar1 = time_boxcar + i_tmid
                        positions = np.array([np.where(time_corr.round(4) == element)[0][0] if len(np.where(time_corr.round(4) == element)[0])>0 else -1 for element in time_boxcar1.round(4)])
                        i = np.where(np.array(positions) != -1)[0]
                        positions = np.array(positions)[i]
                        time_mov = time_corr[positions]
                        flux_mov = flux_corr[positions]
                        positions_boxcar = np.array([np.where(time_boxcar1.round(4) == element)[0][0] if len(np.where(time_boxcar1.round(4) == element)[0])>0 else -1 for element in time_mov.round(4)])
                        ib = np.where(np.array(positions_boxcar) != -1)[0]
                        positions_boxcar = np.array(positions_boxcar)[ib]
                        boxcar_mov = boxcar[positions_boxcar]


                        if len(flux_mov) > 1 and len(np.unique(boxcar_mov)) > 1:
                            corr = np.corrcoef(boxcar_mov, flux_mov)[0,1]
                            cross_correlation.append(corr)
                    cross_correlation = np.array(cross_correlation)
                    indx = np.where(cross_correlation == np.max(cross_correlation))[0][0]
                    #why is the error like this?
                    crossc_midtimes.append([time_corr[indx], self.transit_duration/50])
                    print('Recomended ehpemeris midtime was {}'.format(midtime))
                    print('Crosscorrelation midtime is {}'.format(time_corr[indx]))
        
        self.t_midtimes = crossc_midtimes





    def setup_midtimes(self):
        """ A functiont o ascosiate midtimes with the corresponding observations
        """
        self.t_midtimes_references = []
        self.t_midtimes_continuous_obs_referenecs = []

        for midtime in self.t_midtimes:
            self.t_midtimes_references.append(None)
    
            for i, observation in enumerate(self.observation_times):
                if midtime[0] > observation[0] and midtime[0] < observation[1]:
                    self.t_midtimes_references[-1] = self.observation_times_references[i]
                    self.t_midtimes_continuous_obs_referenecs.append(i)
                    break


    def phase_plot(self):
        self.setup_midtimes()
        fig, ax = plt.subplots(1,1, figsize = (13,5))
        ax.set(title = self.name + ' ' + str(len(self.t_midtimes)) + ' Transits', xlabel = 'time', ylabel = 'flux')
        for i,tmid in enumerate(self.t_midtimes):
            obs = self.observations[self.t_midtimes_references[i]]
            indx = (obs['time'] > tmid[0] - self.transit_duration - 1/24) * (obs['time'] < tmid[0] + self.transit_duration + 1/24)
            time = obs['time'][indx] - tmid[0]
            flux = obs['flux'][indx]/np.mean(obs['flux'][indx])
            ax.scatter(time, flux, s = 1, alpha = 0.8, label = 'Midtime {:.5f}, {}'.format(tmid[0], obs['plot_label']))
        plt.legend()

                    

    def setup_periods(self):


        if len(self.t_midtimes) <= 1:
            if self.reference_midtime is None and self.P_max is None:
                raise ValueError('Provide at least 2 midtimes, or reference_midtime and P_max')
            self.N_transits = 1
            print('Provided maximum period of {:.2f} and midtime reference of {:.2f}'.format(self.P_max[0], self.reference_midtime[0]))

        
        if len(self.t_midtimes) == 2:
            t_mid = np.array(self.t_midtimes)[:,0]
            t_mid_err = np.array(self.t_midtimes)[:,1]
            max_period = np.max(t_mid) - np.min(t_mid)
            max_period_err = np.sqrt(t_mid_err[0]**2 + t_mid_err[1]**2)
            self.P_max = [max_period, max_period_err]
            self.reference_midtime = [t_mid[1], t_mid_err[1]]
            self.N_transits = 2
            print('Provided 2 midtimes, and calculated the maximum period: {:.2f} and midtime reference: {:.2f}'.format(self.P_max[0], self.reference_midtime[0]))

        if len(self.t_midtimes) > 2:
            #raise ValueError("It should be fairly easy, but because of Themis boredom, we still haven't written a part for the code that supports more than 2 midtimes")
            t_mid = np.array(self.t_midtimes)[:,0]
            t_mid_err = np.array(self.t_midtimes)[:,1]
            p = t_mid.argsort()
            t_mid = t_mid[p]
            t_mid_err = t_mid_err[p]
            if self.P_max is None:
                self.P_max = [np.min(np.diff(t_mid)),0.003]
                print('Assumed maximum period to be {}'.format(self.P_max))
            epochs = np.round((t_mid - self.reference_midtime[0])/self.P_max[0])
            # self.P_max[0] = self.P_max[0]*np.min(np.diff(epochs))
            # epochs = np.round((t_mid - self.reference_midtime[0])/self.P_max[0])
            popt, pcov = curve_fit(lin_eph, epochs, t_mid, sigma = t_mid_err)
            self.reference_midtime = [popt[0], np.sqrt(pcov[0,0])]
            self.P_max = [popt[1], np.sqrt(pcov[1,1])]
            print('Calculated maximum period: {:.2f} and midtime reference: {}'.format(self.P_max[0], self.reference_midtime[0]))
            self.N_transits = len(self.t_midtimes)



    def possible_periods(self):

        self.setup_periods()
        self.setup_midtimes()

        all_periods = np.array([])
        all_periods_err = np.array([])
        all_ns = np.array([])
        n = 1

        maximum_period = self.P_max[0]
        maximum_period_err = self.P_max[1]

        while maximum_period/n >= self.P_min:
            period1 = maximum_period/n
            all_periods = np.append(all_periods, period1)
            all_periods_err = np.append(all_periods_err, maximum_period_err/n)
            all_ns = np.append(all_ns, n)
            n += 1


        if self.reference_midtime is None:
            warnings.warn('Pay a lot of attention to your P_min, or you might end up with to much possible perods')
        #try to constrain a bit more our possible periods based on the observation times that we have
        self.missed_transits_total = np.zeros(len(all_periods))
        #we would like to know exactly where the transit was missed
        self.missed_transits_per_obs = {}
        if self.observation_times is not None:
            for i,period in enumerate(all_periods):
                self.missed_transits_per_obs[all_ns[i]] = {}
                for j,continuous_observation in enumerate(self.observation_times):
                    continuous_observation_minlim = continuous_observation[0]-self.transit_edge_limit
                    continuous_observation_maxlim = continuous_observation[1]+self.transit_edge_limit
                    phase = (np.array([continuous_observation_minlim,continuous_observation_maxlim]) - self.reference_midtime[0])/period
                    ## we will add a random big number, to avoif the fact that the int function of -0.2, and+0.1 aer both zero
                    add_random = 100000
                    ## i hope i do not need to make it bigger or smaller
                    n_missed_obs = int(phase[1]+add_random) - int(phase[0]+add_random)                    
                    n_actual_obs = int(np.sum(np.array(self.t_midtimes_continuous_obs_referenecs) == j))
                    self.missed_transits_total[i] += n_missed_obs - n_actual_obs
                    try:
                        self.missed_transits_per_obs[all_ns[i]][self.observation_times_references[j]] += (n_missed_obs - n_actual_obs)
                    except:
                        self.missed_transits_per_obs[all_ns[i]][self.observation_times_references[j]] = (n_missed_obs - n_actual_obs)

        ## lets see if this practice is good enough, by defining the results like this
        self.all_possible_periods = all_periods
        self.all_possible_periods_err = all_periods_err
        self.all_possible_periods_norder = all_ns
        self.all_possible_periods_df = pd.DataFrame({'n_order':all_ns,'period':all_periods, 'period_err':all_periods_err, 'missed_transits_total':self.missed_transits_total})

        
    def sanity_plot(self, plot_periods_orders, plot_observation):
        """
        A function to plot the possible periods of the planet, and the

        WRNING - does not work if you just add more midtimes - meaning that?
        """
        # try:
        indx = []

        for i in plot_periods_orders:
            indx.append(np.where(self.all_possible_periods_norder == i)[0][0])
        
        plot_periods = self.all_possible_periods[indx]
        maxlim = np.max(self.observations[plot_observation]['time']) + 20*self.transit_duration
        minlim = np.min(self.observations[plot_observation]['time']) - 20*self.transit_duration
        fig = plt.figure(figsize = (13,5))
        ax = fig.add_subplot(111)
        ax.set(title = 'Sanity PLot '+self.observations[plot_observation]['plot_label'], xlabel = 'Time]', ylabel = 'Flux')
        ax.scatter(self.observations[plot_observation]['time'], self.observations[plot_observation]['flux'], c = 'black', s = 1)
        for plot_period in plot_periods:
            max_epoch = int((maxlim - self.reference_midtime[0])/plot_period) + 5
            min_epoch = int((minlim - self.reference_midtime[0])/plot_period) - 5
            epochs = np.arange(min_epoch, max_epoch)
            col = (np.random.random(), np.random.random(), np.random.random())
            is_first = True
            if len(epochs) > 0:
                for epoch in epochs:
                    t0 = lin_eph(plot_period, self.reference_midtime[0], epoch)
                    
                    if t0 > minlim and t0 < maxlim:
                        if is_first == True:
                            ax.axvline(t0, c = col, alpha = 0.5,linewidth=1, label = 'Period {:.2f}'.format(plot_period))
                            is_first = False
                        else:
                            ax.axvline(t0, c = col, alpha = 0.5,linewidth=1)
                        ax.axvspan(t0-self.transit_duration/2, t0+self.transit_duration/2, alpha = 0.1, color = col)

        plt.legend()
        plt.show()

        # except:
        #     fig = plt.figure(figsize = (13,5))

        #     ax = fig.add_subplot(111)
        #     ax.set(title = 'Sanity PLot', xlabel = 'Time]', ylabel = 'Flux')
        #     ax.scatter(self.observations[plot_observation]['time'], self.observations[plot_observation]['flux'], c = 'black', s = 1)







    def setup_possible_periods(self, df = None):
        if df is None:
            df = self.all_possible_periods_df
            print('all the periods will be used for the calculation, if not provide a dataframe')
        
        df.reset_index(drop = True, inplace = True)
        self.possible_periods_values = np.array(df['period'].values)
        self.possible_periods_err_values = np.array(df['period_err'].values)
        self.possible_periods_norder_values = np.array(df['n_order'].values)
        self.possible_periods_df = df



    def obs_bias_1period_prob(self, period, N_transits):
        """A function to calculate the the probability we will have N observations of a transit, given our observation times, and the transit period

        Practically it phase folds the obsrvation times with the period, to get the result

        ...

        Parameters
        ----------
        period : float
            individual period

        N_transits: integer
            the number of transits you have observed
        prob_period : float, idealy smaller than 1
            can be changed, if you have some other evindence, for example from radial velocities, that can constrain the period
        """
        observation_times = np.array(self.observation_times)
        #here, we will take into account the transit duration
        observation_times[:, 0] = observation_times[:, 0] - self.transit_duration/2
        observation_times[:, 1] = observation_times[:, 1] + self.transit_duration/2
        ref_time = self.reference_midtime[0]
        n_phase_space = 100000
        phase_space = np.linspace(0,1001, n_phase_space)
        N_obervations_per_phase = np.zeros(len(phase_space))
        ##
        for obs_time in observation_times:
            start_phase = ((obs_time[0] - ref_time)/period) % 1
            tot_phase_cover = ((obs_time[1] - obs_time[0])/period)
            ## create array to add to the total probabilitities

            ## are you sure that this will always give a multiple of 
            add_phase_array = np.concatenate((np.zeros(round(start_phase*n_phase_space)), np.ones(round(tot_phase_cover*n_phase_space))))
            n_missing = n_phase_space - len(add_phase_array)%n_phase_space
            add_phase_array = np.concatenate((add_phase_array, np.zeros(n_missing)))

            add_phase_array = add_phase_array.reshape(int(len(add_phase_array)/n_phase_space), n_phase_space)
            N_obervations_per_phase += np.sum(add_phase_array, axis = 0)
        ## also, has to be added the probability that we see at least one, at least 2 transits etc
        ## are you sure that prior period prob can be treated like this?
        if N_transits == 0:
            return (1 - round(np.sum(N_obervations_per_phase>=1)/n_phase_space, 4))


        return (round(np.sum(N_obervations_per_phase == N_transits)/n_phase_space, 4))
    


    
    def obs_bias_period_prob(self, N_transits = None):

        if N_transits is None:
            if self.N_transits is None:
                raise ValueError('You have to specify the number of transits you have observed')
            else:
                N_transits = self.N_transits
                print('N transit is set automatically by the midtimes value that we had before')

        
        bias_per_prob = np.array([])
        for i in self.possible_periods_values:

            ppp = self.obs_bias_1period_prob(i, N_transits)
            bias_per_prob = np.append(bias_per_prob, ppp)

        self.obs_bias_period_prob_values = bias_per_prob
        self.possible_periods_df = self.possible_periods_df.assign(obs_bias_period_prob = bias_per_prob)



    def setup_probability_period(self):
        if self.prior_prob_period is None:
            self.prior_prob_period = np.ones(len(self.obs_bias_period_prob_values))

        if len(self.prior_prob_period) != len(self.obs_bias_period_prob_values):
            raise ValueError('You have to provide a prior probability for all possible periods')


        final_prob_period = self.prior_prob_period * self.obs_bias_period_prob_values
        final_prob_period = final_prob_period/np.sum(final_prob_period)

        self.final_prob_period = final_prob_period
        self.possible_periods_df = self.possible_periods_df.assign(final_prob_period = final_prob_period)

    def get_periods_dict(self):
        self.periods_dict = {}
        for i, period in enumerate(self.possible_periods_values):
            self.periods_dict[round(period,2)] = {'period':period,\
                            'period_error':self.possible_periods_err_values[i], 'n_order':self.possible_periods_norder_values[i],\
                            'final_prob':np.array(self.final_prob_period)[i]}
    


    def best_assumed_period(self):

        self.setup_probability_period()

        p = self.possible_periods_values
        perr = self.possible_periods_err_values


        N_init = len(p)
        print(N_init)
        N_is_transit = np.array([])
        N_no_transit = np.array([])
        prob_is_transit = np.array([])

        #so badly written
        for assumed_period, assumed_period_err in zip(p, perr):
            updated_planet = UnknownPeriodPlanet(self.name  + '_updated', P_min = np.min(self.possible_periods_values) - 1)
            updated_planet.P_max = [assumed_period, assumed_period_err]
            updated_planet.reference_midtime = self.reference_midtime
            updated_planet.possible_periods()
            updated_planet.setup_possible_periods()
            possible_assumed_period = updated_planet.possible_periods_values
            in_assumed = np.array([per.round(7) in p.round(7) for per in possible_assumed_period])
            possible_assumed_period = possible_assumed_period[in_assumed]
            #calculate the probability that we will observe a transit at this night
            #we round the values to 4 decimal places, isuppose it will not affect our periods
            prob_is_from_assumed = np.array([])
            for possible_per in possible_assumed_period:
                prob_possible_per = self.final_prob_period[np.where(possible_per.round(7) == p.round(7))]
                prob_is_from_assumed = np.append(prob_is_from_assumed, prob_possible_per)

            prob_is_transit = np.append(prob_is_transit, np.sum(prob_is_from_assumed))


            n_poss = len(possible_assumed_period)
            #calcualte the pnumber of possible periods we will have if this is a transit
            N_is_transit = np.append(N_is_transit, n_poss)
            #calcualte the pnumber of possible periods we will have if this is not a transit
            N_no_transit = np.append(N_no_transit, N_init - n_poss)
        


        self.N_is_transit = N_is_transit
        self.N_no_transit = N_no_transit
        self.prob_is_transit = prob_is_transit
        self.possible_periods_df['N_is_transit'] = N_is_transit
        self.possible_periods_df['N_no_transit'] = N_no_transit
        self.possible_periods_df['prob_is_transit'] = prob_is_transit


    def initialize_star(self, star_ra, star_dec):
        """
        A function to initialize the star that we are observing
        -----------------
        at the moment using astropy and obsplan package
        """
        #define observer and target
        star_coordinates = SkyCoord(ra = star_ra*u.deg, dec = star_dec*u.deg)
        target = FixedTarget(star_coordinates, name=self.name)
        print('RA = {} degrees, dec = {} degrees, I hope you didnt mess up'.format(star_coordinates.ra, star_coordinates.dec))
        self.star_coordinates = star_coordinates
        self.target = target


    def initalize_observatory_locations(self, observatory_locations):
        """
        ----------------
        A function to initialize the observatory locations
        ----------------

    
        Parameters
        ----------
        observatory_locations : list
            list of lists observatory locations, in the form of [lat, long, height, name]
        """
        locations = []
        observatories = []
        for observatory_location in observatory_locations:
            loc = EarthLocation.from_geodetic(observatory_location[1]*u.deg, observatory_location[0]*u.deg, height = observatory_location[2]*u.m)
            observer = Observer(loc, name = observatory_location[3], timezone = 'UTC')
            locations.append(loc)
            observatories.append(observer)
            print('Observatory {}: lat = {:.2f} degrees, long = {:.2f} degrees, height = {:.2f} m'.format(observatory_location[3],loc.lat.value, loc.lon.value, loc.height.value))
        self.observatories = observatories
        self.locations = locations



    def _all_transits_multiple_periods(self, obs_start_end, tw):
        """ A function to calculate observable transits
        Still in beta version

        
        work to be done
        ----------------
        - add the possibility to calculate transits without specification of observatory
        - more robust time converstions


        Parameters
        -----------
        obs_start_end : list
            2 - element list of the start and the end of the observation time

        tw : float
            the time we want the program to take into account when calculating the out of transit time
        """

        observ_start = Time(datetime.strptime(obs_start_end[0], "%Y-%m-%d")).jd
        observ_end = Time(datetime.strptime(obs_start_end[1], "%Y-%m-%d")).jd

        print('Observation start JD: ', observ_start)
        print('Observation end JD: ', observ_end)

       
        midtime_reference = self.reference_midtime[0]
        midtime_reference_err = self.reference_midtime[1]

        ## create dictionary with all the possible periods, and the corresponding midtimes in the calculated time interval
        midtimes_dict = {}
        for period, period_err in zip(self.possible_periods_values, self.possible_periods_err_values):
            #find minimum epoch
            N_min = int(np.ceil((observ_start - midtime_reference)/period))
            N_max = int((observ_end - midtime_reference)/period)
            epochs = np.arange(N_min, N_max+1)
            midtimes = lin_eph(period, midtime_reference, epochs)
            midtimes_err = lin_eph_err(period_err, midtime_reference_err, epochs)
            midtimes_dict[str(round(period,2))] = [midtimes, midtimes_err]

        f_midtimes = np.array([])
        f_midtimes_err = np.array([])
        affiliated_periods = []
        N_affiliated_periods = np.array([])
        prob_obs_transit = np.array([])

        for ii,k in enumerate(midtimes_dict.keys()):
            for i,x in enumerate(midtimes_dict[k][0]):
                if np.round(x,8) not in np.round(f_midtimes,8):
                    f_midtimes = np.append(f_midtimes, x)
                    f_midtimes_err = np.append(f_midtimes_err, midtimes_dict[k][1][i])
                    affiliated_periods.append(k + '-')
                    N_affiliated_periods = np.append(N_affiliated_periods, 1)
                    prob_obs_transit = np.append(prob_obs_transit, self.prob_is_transit[ii])

                else:
                    indx = np.where(np.round(x,8) == np.round(f_midtimes,8))[0][0]
                    N_affiliated_periods[indx] += 1
                    affiliated_periods[indx] = str(affiliated_periods[indx]) + k + '-'
                    if self.prob_is_transit[ii] > prob_obs_transit[indx]:
                        prob_obs_transit[indx] = self.prob_is_transit[ii]
        
        print(f_midtimes)

        ##################################
        ###### this time converstion i probably do not like......keep it in mind for improovemnt

        ######################THIS IS THE MISTAKE AAAAA
        midtimes_bjd = Time(f_midtimes*u.day, format = 'jd', scale = 'utc')

        #assuming one observatpry on earth, randomly!
        midtimes_jd = midtimes_bjd # - midtimes_bjd.light_travel_time(self.star_coordinates, location = self.locations[0])
        # location = loc)
        ##################################
        ####################
        ## and convert also to more human readablt formats
        ###############################
        conv_datetimes = midtimes_jd.isot
        conv_dates = []
        conv_times = []
        for x in np.char.split(conv_datetimes, 'T'):
            conv_dates.append(x[0])
            conv_times.append(x[1])


        start_transit = Time((midtimes_jd.value - self.transit_duration/2)*u.day, format = 'jd', scale = 'utc')
        end_transit = Time((midtimes_jd.value + self.transit_duration/2)*u.day, format = 'jd', scale = 'utc')
        start_obs = Time((midtimes_jd.value - self.transit_duration/2 - tw)*u.day, format = 'jd', scale = 'utc')
        end_obs = Time((midtimes_jd.value + self.transit_duration/2 + tw)*u.day, format = 'jd', scale = 'utc')



        transits_df = pd.DataFrame({'Date_obs' : conv_dates, 'midtime_obs (UT)' : conv_times, 'midtime_err_min':f_midtimes_err*24*60, \
                            'N_periods':N_affiliated_periods,'affiliated_periods' : affiliated_periods,'prob_is_transit' : prob_obs_transit, \
                            'start_transit' : start_transit.isot, 'end_trasnit':end_transit.isot, 'midtransit_isot':midtimes_jd.isot,\
                            'midtimes (BJD)':midtimes_jd.value, 'midtimes_err':f_midtimes_err,'start_obs': start_obs, 'end_obs':end_obs})
        
        transits_df = transits_df.sort_values(by = 'midtimes (BJD)')
        transits_df = transits_df.reset_index(drop = True)
        return transits_df

    def all_transits_multiple_periods(self, obs_start_end, tw):
        """
        function to handle multiple observable intervals
        """
        self.all_transits_df = None

        for obsstartend in obs_start_end:
            temporary_df = self._all_transits_multiple_periods(obsstartend, tw)
            try:
                self.all_transits_df = pd.concat([self.all_transits_df, temporary_df])

            except:
                self.all_transits_df = temporary_df.copy()
                


    def observable_transits_multiple_periods(self, min_h_horizon, min_moon_separation):
        self.all_transits_df.reset_index(drop = True, inplace = True)

        times_start =  Time(self.all_transits_df['midtimes (BJD)'].values + self.transit_duration*9/10, format = 'jd')
        times_end =  Time(self.all_transits_df['midtimes (BJD)'].values + self.transit_duration*11/10, format = 'jd')
        egress = self._observable_transits_multiple_periods(min_h_horizon, min_moon_separation, times_start, times_end)

        times_start =  Time(self.all_transits_df['midtimes (BJD)'].values - self.transit_duration*11/10, format = 'jd')
        times_end =  Time(self.all_transits_df['midtimes (BJD)'].values - self.transit_duration*9/10, format = 'jd')
        ingress = self._observable_transits_multiple_periods(min_h_horizon, min_moon_separation, times_start, times_end)

        times_start = self.all_transits_df['start_obs'].values
        times_end = self.all_transits_df['end_obs'].values
        
        full_transits = self._observable_transits_multiple_periods(min_h_horizon, min_moon_separation, times_start, times_end)

        for k,observatory in enumerate(self.observatories):
            tt = []
            print(len(full_transits[k]))
            for kk in range(len(full_transits[k])):
                print(kk)
                if full_transits[k][kk] == 1:
                    tt.append('full')
                    continue
                elif egress[k][kk] == 1 and ingress[k][kk] == 1:
                    tt.append('ingr + egr')
                    continue
                elif egress[k][kk] == 1:
                    tt.append('egress')
                    continue
                elif ingress[k][kk] == 1:
                    tt.append('ingress')
                    continue
                else:
                    tt.append(0)
            print(tt)
            print(len(tt))
            self.all_transits_df[observatory.name] = tt


        
    def _observable_transits_multiple_periods(self, min_h_horizon, min_moon_separation, times_start, times_end):
        constraints = [AltitudeConstraint(min=min_h_horizon * u.deg), MoonSeparationConstraint(min=min_moon_separation*u.deg),AtNightConstraint.twilight_astronomical()]
        #PhaseConstraint(binary_system, min=0.4, max=0.6)
        is_observable = []
        for k, obseratory in enumerate(self.observatories):
            is_observable.append([])
        for i in self.all_transits_df.index:
            for k,observatory in enumerate(self.observatories):
                aa = is_always_observable(constraints, observatory, self.target, time_range=[times_start[i], times_end[i]])
                is_observable[k].append(aa[0])

        return is_observable
        
        ## 'start_obs':start_obs, 'end_obs':end_obs
        ## 'midtimes (BJD)':midtimes_bjd,
                                
        #set the observability from the different obsevatories


        # check if it is observable from any of the observatories
        # is_observable_arr = np.array(is_observable)
        # is_observable_1d = np.any(is_observable_arr, axis = 0)


    def max_P_coverage(self, N_observations, df_transits = None, max_N_periods = 1000):
        """
        A function to calculate the best observing nights to see the transit, at least once
        cover the most possible orders


        Parameters
        ---------------------
        df_transits : DataFrame, 
        if not specified, the observable transits dataframe will be used from the class

        N_observations : integer
            how many observations do we have to find our planet?

        max_N_periods : integer
            i do not know if this is useful, we might need it, if for example we do not want to look again at the maximum possible period
        """
        if df_transits is None:
            try:
                df_transits = self.all_transits_df.copy()
            
            except:

                try:
                    df_transits = self.transits_df.copy()
                
                except:
                    raise ValueError('You have to calculate at least one dataframe with future observable transits')


        df_good = df_transits.query("0 < N_periods < @max_N_periods")

        if len(df_good) < N_observations:
            df_good = df_transits.copy()
            ## N_observations = len(df_good)??

        df_good = df_good.sort_values(by = ['N_periods', 'midtimes (BJD)'], ascending = [False, True])
        df_good.reset_index(inplace = True, drop = True)

        best_combination = []
        best_combination_index = []

        ### check, taking as granted, the first observation. This will be one of the highest afiliated periods observations - 
        ### and the number of times we check depends on the N observations we will make
        ###
        for i in range(N_observations):
            print(i)
            best_comb1 = []
            best_indx1 = [i]
            for j in range(N_observations - 1):
                diff = []
                for k in range(len(df_good)):
                    best_comb3 = add_periods(period_list(df_good['affiliated_periods'][i]),best_comb1,period_list(df_good['affiliated_periods'][k]))
                    diff.append(len(best_comb3) - len(best_comb1))
                
                #we find the index of the first maximum that we got to
                if max(diff) > 0:
                    max_indx = diff.index(max(diff))
                    best_comb1 = add_periods(period_list(df_good['affiliated_periods'][i]),best_comb1,period_list(df_good['affiliated_periods'][max_indx]))
                    best_indx1.append(max_indx)


            if len(best_comb1) > len(best_combination):
                best_combination = best_comb1.copy()
                best_combination_index = best_indx1.copy()
            
        
        
        df_good = df_good.iloc[best_combination_index]            
        df_good.reset_index(drop = True, inplace = True)

        ## calculate the total probability of seeing the transit at least once:
        best_combination_floats = np.array([float(x) for x in best_combination])
        best_combination_floats.sort()

        def get_prob(periods):
            prob = 0
            for period in periods:
                indx = np.where(np.round(self.possible_periods_values,2) == period)[0][0]
                prob += self.final_prob_period[indx]
            return prob
        

        n = 0
        all_periods1 = [[]]
        for i in df_good.index:
            diff0 = len(all_periods1)
            all_periods1 = add_periods(period_list(df_good['affiliated_periods'][i]))
            diff1 = len(all_periods1)

        total_prob = get_prob(best_combination_floats)
        print('If we observe these {} transits, we observe in total {}/{} period orders, with probability to see transit: {:.2f}'.format(len(df_good),len(best_combination), len(self.possible_periods_df),total_prob))

        return best_combination_floats, df_good
    




    def max_P_coverage_prob(self, N_observations, df_transits = None, max_N_periods = 1000):
        """
        A function to calculate the best observing nights to see the transit, at least once
        cover the most possible orders


        Parameters
        ---------------------
        df_transits : DataFrame, 
        if not specified, the observable transits dataframe will be used from the class

        N_observations : integer
            how many observations do we have to find our planet?

        max_N_periods : integer
            i do not know if this is useful, we might need it, if for example we do not want to look again at the maximum possible period
        """
        if df_transits is None:
            try:
                df_transits = self.all_transits_df.copy()
            
            except:

                try:
                    df_transits = self.transits_df.copy()
                
                except:
                    raise ValueError('You have to calculate at least one dataframe with future observable transits')


        df_good = df_transits.query("0 < N_periods < @max_N_periods")

        if len(df_good) < N_observations:
            df_good = df_transits.copy()
            ## N_observations = len(df_good)??
        ## this should play an important role to how we make the situation work hahaha,
        ## in theory we could actually remove the N periods, to see where this gets us
        ## setting ascending to False, in order not to Favor transits with only one possible period order very early
        ## i think this would depend a lot on the transit window that we choode to have hahaha
        df_good = df_good.sort_values(by = ['N_periods', 'prob_is_transit','midtimes (BJD)'], ascending = [False, False, False])
        df_good.reset_index(inplace = True, drop = True)

        best_combination = []
        best_combination_index = []

        ### check, taking as granted, the first observation. This will be one of the highest afiliated periods observations - 
        ### and the number of times we check depends on the N observations we will make
        ###
        for i in range(N_observations):
            best_comb1 = []
            best_indx1 = [i]
            for j in range(N_observations - 1):
                diff = []
                for k in range(len(df_good)):
                    best_comb3 = add_periods(period_list(df_good['affiliated_periods'][i]),best_comb1,period_list(df_good['affiliated_periods'][k]))
                    diff.append(len(best_comb3) - len(best_comb1))
                
                #we find the index of the first maximum that we got to
                if max(diff) > 0:
                    max_indx = diff.index(max(diff))
                    best_comb1 = add_periods(period_list(df_good['affiliated_periods'][i]),best_comb1,period_list(df_good['affiliated_periods'][max_indx]))
                    best_indx1.append(max_indx)


            if len(best_comb1) > len(best_combination):
                best_combination = best_comb1.copy()
                best_combination_index = best_indx1.copy()
            
        
        
        df_good = df_good.iloc[best_combination_index]            
        df_good.reset_index(drop = True, inplace = True)

        ## calculate the total probability of seeing the transit at least once:
        best_combination_floats = np.array([float(x) for x in best_combination])
        best_combination_floats.sort()

        def get_prob(periods):
            prob = 0
            for period in periods:
                indx = np.where(np.round(self.possible_periods_values,2) == period)[0][0]
                prob += self.final_prob_period[indx]
            return prob
        

        n = 0
        all_periods1 = [[]]
        for i in df_good.index:
            diff0 = len(all_periods1)
            all_periods1 = add_periods(period_list(df_good['affiliated_periods'][i]))
            diff1 = len(all_periods1)

        total_prob = get_prob(best_combination_floats)
        print('If we observe these {} transits, we observe in total {}/{} period orders, with probability to see transit: {:.2f}'.format(len(df_good),len(best_combination), len(self.possible_periods_df),total_prob))

        return best_combination_floats, df_good
    


    def create_plan(self, obs_df = None, observe_periods = 'all'):
        """
        A function to create a plan for the observations

        Parameters
        ----------------
        obd_df : DataFrame
            if not specified

            """
        if obs_df is None:
            obs_df = self.all_transits_df.copy()

        if observe_periods == 'all':
            a = add_periods_list(list(obs_df['affiliated_periods']))
            print('aaa')
        return a


    






