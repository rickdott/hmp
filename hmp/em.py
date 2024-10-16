from warnings import warn
import numpy as np
import multiprocessing as mp
import itertools

def EM_star(args):  # for tqdm usage
    return EM_class(*args).EM()


class EM_class:

    def __init__(
        self,
        magnitudes,
        parameters,
        maximization=True,
        magnitudes_to_fix=None,
        parameters_to_fix=None,
        max_iteration=1e3,
        tolerance=1e-4,
        min_iteration=1,
        mags_map=None,
        pars_map=None,
        conds=None,
        cpus=1,
        locations=None,
        shape=None,
        location=None,
        # location_corr_threshold=None, # Dont support location_corr_threshold as it requires keeping the entire dataset in memory on each core
        scale_to_mean=None,
        max_d=None,
        n_trials=None,
        durations=None,
        starts=None,
        ends=None,
        n_samples=None,
        n_dims=None,
        events=None,
        convolution=None,
        data_matrix=None,
        pdf=None,
        mean_to_scale=None,
    ):
        self.magnitudes = magnitudes
        self.parameters = parameters
        self.maximization = maximization
        self.magnitudes_to_fix = magnitudes_to_fix
        self.parameters_to_fix = parameters_to_fix
        self.max_iteration = max_iteration
        self.tolerance = tolerance
        self.min_iteration = min_iteration
        self.mags_map = mags_map
        self.pars_map = pars_map
        self.conds = conds
        self.cpus = cpus
        self.locations = locations
        self.shape = shape
        self.location = location
        # self.location_corr_threshold = location_corr_threshold
        self.scale_to_mean = scale_to_mean
        self.max_d = max_d
        self.n_trials = n_trials
        self.durations = durations
        self.starts = starts
        self.ends = ends
        self.n_samples = n_samples
        self.n_dims = n_dims
        self.events = events
        self.convolution = convolution
        self.data_matrix = data_matrix
        self.pdf = pdf
        self.mean_to_scale = mean_to_scale

    def EM(self):
        """
        Expectation maximization function underlying fit

        parameters
        ----------
        n_events : int
            how many events are estimated
        magnitudes : ndarray
            2D ndarray n_events * components (or 3D iteration * n_events * n_components), initial conditions for events magnitudes. If magnitudes are estimated, the list provided is used as starting point,
            if magnitudes are fixed, magnitudes estimated will be the same as the one provided. When providing a list, magnitudes need to be in the same order
            _n_th magnitudes parameter is  used for the _n_th event
        parameters : list
            list of initial conditions for Gamma distribution parameters parameter (2D stage * parameter or 3D iteration * n_events * n_components). If parameters are estimated, the list provided is used as starting point,
            if parameters are fixed, parameters estimated will be the same as the one provided. When providing a list, stage need to be in the same order
            _n_th gamma parameter is  used for the _n_th stage
        maximization: bool
            If True (Default) perform the maximization phase in EM() otherwhise skip
        magnitudes_to_fix: bool
            To fix (True) or to estimate (False, default) the magnitudes of the channel contribution to the events
        parameters_to_fix : bool
            To fix (True) or to estimate (False, default) the parameters of the gammas
        max_iteration: int
            Maximum number of iteration for the expectation maximization
        tolerance: float
            Tolerance applied to the expectation maximization
        min_iteration: int
            Minimum number of iteration for the expectation maximization in the EM() function
        locations : ndarray
            Initial locations (1D for normal model, 2D for condition based model (cond * n_events)

        Returns
        -------
        lkh : float
            Summed log probabilities
        magnitudes : ndarray
            Magnitudes of the channel contribution to each event
        parameters: ndarray
            parameters for the gammas of each stage
        eventprobs: ndarray
            Probabilities with shape max_samples*n_trials*n_events
        locations : ndarray
            locations for each event
        traces: ndarray
            Values of the log-likelihood for each EM iteration
        locations_dev : ndarray
            locations for each interation of EM
        param_dev : ndarray
            paramters for each iteration of EM
        """

        if (
            self.mags_map is not None
            or self.pars_map is not None
            or self.conds is not None
        ):  # condition version
            assert (
                self.mags_map is not None
                and self.pars_map is not None
                and self.conds is not None
            ), "Both magnitude and parameter maps need to be provided when doing EM based on conditions, as well as conditions."
            assert (
                self.mags_map.shape[0] == self.pars_map.shape[0]
            ), "Both maps need to indicate the same number of conditions."
            n_cond = self.mags_map.shape[0]
        else:
            n_cond = None

        if not isinstance(
            self.maximization, bool
        ):  # Backward compatibility with previous versions
            warn(
                "Deprecated use of the threshold function, use maximization and tolerance arguments. Setting tolerance at 1 for compatibility"
            )
            self.maximization = {1: True, 0: False}[self.maximization]
            if (
                self.maximization
            ):  # Backward compatibility, equivalent to previous threshold = 1
                tolerance = 1

        null_stages = np.where(self.parameters[..., -1].flatten() < 0)[0]
        wrong_shape = np.where(self.parameters[..., -2] != self.shape)[0]
        if n_cond is None and len(null_stages) > 0:
            raise ValueError(
                f"Wrong scale parameter input, provided scale parameter(s) {null_stages} should be positive but have value {self.parameters[...,-1].flatten()[null_stages]}"
            )
        if n_cond is None and len(wrong_shape) > 0:
            raise ValueError(
                f"Wrong shape parameter input, provided parameter(s) {wrong_shape} shape is {self.parameters[...,-2][wrong_shape]} but expected {self.shape}"
            )

        n_events = self.magnitudes.shape[self.magnitudes.ndim - 2]
        if n_events == 0:
            raise ValueError(f"At least one event has to be required")
        initial_magnitudes = self.magnitudes.copy()
        initial_parameters = self.parameters.copy()
        if self.locations is None:
            self.locations = np.zeros((n_events + 1,), dtype=int)  # location per stage
            self.locations[1:-1] = (
                self.location
            )  # default/starting point is self.location
            if n_cond is not None:
                self.locations = np.tile(self.locations, (n_cond, 1))
        else:
            self.locations = self.locations.astype(int)

        if n_cond is not None:
            lkh, eventprobs = self.estim_probs_conds()
        else:
            lkh, eventprobs = self.estim_probs(self.magnitudes, self.parameters, self.locations, n_events)

        traces = [lkh]
        locations_dev = [self.locations.copy()]  # store development of locations
        param_dev = [self.parameters.copy()]  # ... and parameters

        i = 0
        if not self.maximization:
            lkh_prev = lkh
        else:
            lkh_prev = lkh
            parameters_prev = self.parameters.copy()
            locations_prev = self.locations.copy()

            while i < self.max_iteration:  # Expectation-Maximization algorithm
                # if self.location_corr_threshold is None:  # standard threshold
                if i >= self.min_iteration and (
                    np.isneginf(lkh)
                    or self.tolerance > (lkh - lkh_prev) / np.abs(lkh_prev)
                ):
                    break

                # else:  # threshold adapted for location correlation threshold:
                #     # EM only stops if location was not change on last iteration
                #     # and events moved less than .1 sample. This ensures EM continues
                #     # when new locations are set or correlation is still too high.
                #     # (see also get_locations)
                #     stage_durations = np.array(
                #         [self.scale_to_mean(x[0], x[1]) for x in self.parameters]
                #     )
                #     stage_durations_prev = np.array(
                #         [self.scale_to_mean(x[0], x[1]) for x in parameters_prev]
                #     )

                #     if i >= self.min_iteration and (
                #         np.isneginf(lkh)
                #         or (self.locations == locations_prev).all()
                #         and self.tolerance > (lkh - lkh_prev) / np.abs(lkh_prev)
                #         and (np.abs(stage_durations - stage_durations_prev) < 0.1).all()
                #     ):
                #         if np.isneginf(
                #             lkh
                #         ):  # print a warning if log-likelihood is -inf,
                #             # as this typically happens when no good solution exits
                #             print(
                #                 "-!- Estimation stopped because of -inf log-likelihood: this typically indicates requesting too many and/or too closely spaced events -!-"
                #             )
                #         break

                # As long as new run gives better likelihood, go on
                lkh_prev = lkh.copy()
                locations_prev = self.locations.copy()
                parameters_prev = self.parameters.copy()

                if n_cond is not None:  # condition dependent
                    for c in range(n_cond):  # get params/mags

                        mags_map_cond = np.where(self.mags_map[c, :] >= 0)[0]
                        pars_map_cond = np.where(self.pars_map[c, :] >= 0)[0]
                        epochs_cond = np.where(self.conds == c)[0]

                        # get mags/pars/locs by condition
                        (
                            self.magnitudes[c, mags_map_cond, :],
                            self.parameters[c, pars_map_cond, :],
                        ) = self.get_magnitudes_parameters_expectation(
                            eventprobs[
                                np.ix_(range(self.max_d), epochs_cond, mags_map_cond)
                            ],
                            subset_epochs=epochs_cond,
                        )
                        # if (
                        #     self.location_corr_threshold is not None
                        # ):  # update location when location correlation threshold is used
                        #     self.locations[c, pars_map_cond] = self.get_locations(
                        #         self.locations[c, pars_map_cond],
                        #         self.magnitudes[c, mags_map_cond, :],
                        #         self.parameters[c, pars_map_cond, :],
                        #         parameters_prev[c, pars_map_cond, :],
                        #         eventprobs[
                        #             np.ix_(
                        #                 range(self.max_d), epochs_cond, mags_map_cond
                        #             )
                        #         ],
                        #         subset_epochs=epochs_cond,
                        #     )

                        self.magnitudes[c, self.magnitudes_to_fix, :] = (
                            initial_magnitudes[c, self.magnitudes_to_fix, :].copy()
                        )
                        self.parameters[c, self.parameters_to_fix, :] = (
                            initial_parameters[c, self.parameters_to_fix, :].copy()
                        )

                    # set mags to mean if requested in map
                    for m in range(n_events):
                        for m_set in np.unique(self.mags_map[:, m]):
                            if m_set >= 0:
                                self.magnitudes[self.mags_map[:, m] == m_set, m, :] = (
                                    np.mean(
                                        self.magnitudes[
                                            self.mags_map[:, m] == m_set, m, :
                                        ],
                                        axis=0,
                                    )
                                )

                    # set params and locations to mean/max if requested in map
                    for p in range(n_events + 1):
                        for p_set in np.unique(self.pars_map[:, p]):
                            if p_set >= 0:
                                self.parameters[self.pars_map[:, p] == p_set, p, :] = (
                                    np.mean(
                                        self.parameters[
                                            self.pars_map[:, p] == p_set, p, :
                                        ],
                                        axis=0,
                                    )
                                )
                                self.locations[self.pars_map[:, p] == p_set, p] = (
                                    np.max(
                                        self.locations[self.pars_map[:, p] == p_set, p],
                                        axis=0,
                                    )
                                )

                else:  # general
                    self.magnitudes, self.parameters = (
                        self.get_magnitudes_parameters_expectation(eventprobs)
                    )
                    self.magnitudes[self.magnitudes_to_fix, :] = initial_magnitudes[
                        self.magnitudes_to_fix, :
                    ].copy()
                    self.parameters[self.parameters_to_fix, :] = initial_parameters[
                        self.parameters_to_fix, :
                    ].copy()
                    # if (
                    #     self.location_corr_threshold is not None
                    # ):  # update location when location correlation threshold is used
                    #     locations = self.get_locations(
                    #         self.locations,
                    #         self.magnitudes,
                    #         self.parameters,
                    #         parameters_prev,
                    #         eventprobs,
                    #     )

                if n_cond is not None:
                    lkh, eventprobs = self.estim_probs_conds()
                else:
                    lkh, eventprobs = self.estim_probs(self.magnitudes, self.parameters, self.locations, n_events)

                traces.append(lkh)
                locations_dev.append(self.locations.copy())
                param_dev.append(self.parameters.copy())
                i += 1

        # Getting eventprobs without locations
        if n_cond is not None:
            self.locations = np.zeros(self.locations.shape).astype(int)
            _, eventprobs = self.estim_probs_conds()
        else:
            self.locations = np.zeros(self.locations.shape).astype(int)
            _, eventprobs = self.estim_probs(self.magnitudes, self.parameters, self.locations, n_events)
        if i == self.max_iteration:
            warn(
                f"Convergence failed, estimation hitted the maximum number of iteration ({int(self.max_iteration)})",
                RuntimeWarning,
            )
        return (
            lkh,
            self.magnitudes,
            self.parameters,
            eventprobs,
            self.locations,
            np.array(traces),
            np.array(locations_dev),
            np.array(param_dev),
        )

    def estim_probs(
        self,
        magnitudes,
        parameters,
        locations,
        n_events=None,
        subset_epochs=None,
        lkh_only=False,
        by_trial_lkh=False,
    ):
        """
        parameters
        ----------
        magnitudes : ndarray
            2D ndarray n_events * components (or 3D iteration * n_events * n_components), initial conditions for events magnitudes. If magnitudes are estimated, the list provided is used as starting point,
            if magnitudes are fixed, magnitudes estimated will be the same as the one provided. When providing a list, magnitudes need to be in the same order
            _n_th magnitudes parameter is  used for the _n_th event
        parameters : list
            list of initial conditions for Gamma distribution parameters parameter (2D stage * parameter or 3D iteration * n_events * n_components). If parameters are estimated, the list provided is used as starting point,
            if parameters are fixed, parameters estimated will be the same as the one provided. When providing a list, stage need to be in the same order
            _n_th gamma parameter is  used for the _n_th stage
        locations : ndarray
            1D ndarray of int with size n_events+1, locations for events
        n_events : int
            how many events are estimated
        subset_epochs : list
            boolean array indicating which epoch should be taken into account for condition-based calcs
        lkh_only: bool
            Returning eventprobs (True) or not (False)

        Returns
        -------
        likelihood : float
            Summed log probabilities
        eventprobs : ndarray
            Probabilities with shape max_samples*n_trials*n_events
        """
        if n_events is None:
            n_events = magnitudes.shape[0]
        n_stages = n_events + 1

        if subset_epochs is not None:
            if len(subset_epochs) == self.n_trials:  # boolean indices
                subset_epochs = np.where(subset_epochs)[0]
            n_trials = len(subset_epochs)
            durations = self.durations[subset_epochs]
            starts = self.starts[subset_epochs]
            ends = self.ends[subset_epochs]
        else:
            n_trials = self.n_trials
            durations = self.durations
            starts = self.starts
            ends = self.ends

        gains = np.zeros((self.n_samples, n_events), dtype=np.float64)
        for i in range(self.n_dims):
            # computes the gains, i.e. congruence between the pattern shape
            # and the data given the magnitudes of the sensors
            gains = (
                gains
                + self.events[:, i][np.newaxis].T * magnitudes[:, i]
                - magnitudes[:, i] ** 2 / 2
            )
        gains = np.exp(gains)
        probs = np.zeros(
            [self.max_d, n_trials, n_events], dtype=np.float64
        )  # prob per trial
        probs_b = np.zeros(
            [self.max_d, n_trials, n_events], dtype=np.float64
        )  # Sample and state reversed
        for trial in np.arange(n_trials):
            # Following assigns gain per trial to variable probs
            probs[: durations[trial], trial, :] = gains[
                starts[trial] : ends[trial] + 1, :
            ]
            # Same but samples and events are reversed, this allows to compute
            # fwd and bwd in the same way in the following steps
            probs_b[: durations[trial], trial, :] = gains[
                starts[trial] : ends[trial] + 1, :
            ][::-1, ::-1]

        pmf = np.zeros(
            [self.max_d, n_stages], dtype=np.float64
        )  # Gamma pmf for each stage scale
        for stage in range(n_stages):
            pmf[:, stage] = np.concatenate(
                (
                    np.repeat(0, locations[stage]),
                    self.distribution_pmf(parameters[stage, 0], parameters[stage, 1])[
                        locations[stage] :
                    ],
                )
            )
        pmf_b = pmf[:, ::-1]  # Stage reversed gamma pmf, same order as prob_b

        forward = np.zeros((self.max_d, n_trials, n_events), dtype=np.float64)
        backward = np.zeros((self.max_d, n_trials, n_events), dtype=np.float64)
        # Computing forward and backward helper variable
        #  when stage = 0:
        forward[:, :, 0] = (
            np.tile(pmf[:, 0][np.newaxis].T, (1, n_trials)) * probs[:, :, 0]
        )  # first stage transition is p(B) * p(d)
        backward[:, :, 0] = np.tile(
            pmf_b[:, 0][np.newaxis].T, (1, n_trials)
        )  # Reversed gamma (i.e. last stage) without probs as last event ends at time T

        for event in np.arange(
            1, n_events
        ):  # Following stage transitions integrate previous transitions
            add_b = (
                backward[:, :, event - 1] * probs_b[:, :, event - 1]
            )  # Next stage in back
            for trial in np.arange(n_trials):
                # convolution between gamma * gains at previous event and event
                forward[:, trial, event] = self.convolution(
                    forward[:, trial, event - 1], pmf[:, event]
                )[: self.max_d]
                # same but backwards
                backward[:, trial, event] = self.convolution(
                    add_b[:, trial], pmf_b[:, event]
                )[: self.max_d]
            forward[:, :, event] = forward[:, :, event] * probs[:, :, event]
        # re-arranging backward to the expected variable
        backward = backward[:, :, ::-1]  # undoes stage inversion
        for trial in np.arange(n_trials):  # Undoes sample inversion
            backward[: durations[trial], trial, :] = backward[
                : durations[trial], trial, :
            ][::-1]

        eventprobs = forward * backward
        eventprobs = np.clip(eventprobs, 0, None)  # floating point precision error

        # eventprobs can be so low as to be 0, avoid dividing by 0
        # this only happens when magnitudes are 0 and gammas are randomly determined
        if (eventprobs.sum(axis=0) == 0).any() or (
            eventprobs[:, :, 0].sum(axis=0) == 0
        ).any():

            # set likelihood
            eventsums = eventprobs[:, :, 0].sum(axis=0)
            eventsums[eventsums != 0] = np.log(eventsums[eventsums != 0])
            eventsums[eventsums == 0] = -np.inf
            likelihood = np.sum(eventsums)

            # set eventprobs, check if any are 0
            eventsums = eventprobs.sum(axis=0)
            if (eventsums == 0).any():
                for i in range(eventprobs.shape[0]):
                    eventprobs[i, :, :][eventsums == 0] = 0
                    eventprobs[i, :, :][eventsums != 0] = (
                        eventprobs[i, :, :][eventsums != 0] / eventsums[eventsums != 0]
                    )
            else:
                eventprobs = eventprobs / eventprobs.sum(axis=0)

        else:

            likelihood = np.sum(
                np.log(eventprobs[:, :, 0].sum(axis=0))
            )  # sum over max_samples to avoid 0s in log
            eventprobs = eventprobs / eventprobs.sum(axis=0)
        # conversion to probabilities, divide each trial and state by the sum of the likelihood of the n points in a trial

        if lkh_only:
            return likelihood
        elif by_trial_lkh:
            return forward * backward
        else:
            return [likelihood, eventprobs]

    def estim_probs_conds(
        self,
        lkh_only=False,
    ):
        """
        parameters
        ----------
        magnitudes : ndarray
            2D ndarray n_events * components (or 3D iteration * n_events * n_components), initial conditions for events magnitudes. If magnitudes are estimated, the list provided is used as starting point,
            if magnitudes are fixed, magnitudes estimated will be the same as the one provided. When providing a list, magnitudes need to be in the same order
            _n_th magnitudes parameter is  used for the _n_th event
        parameters : list
            list of initial conditions for Gamma distribution parameters parameter (2D stage * parameter or 3D iteration * n_events * n_components). If parameters are estimated, the list provided is used as starting point,
            if parameters are fixed, parameters estimated will be the same as the one provided. When providing a list, stage need to be in the same order
            _n_th gamma parameter is  used for the _n_th stage
        locations : ndarray
            2D n_cond * n_events array indication locations for all events
        n_events : int
            how many events are estimated
        lkh_only: bool
            Returning eventprobs (True) or not (False)

        Returns
        -------
        likelihood : float
            Summed log probabilities
        eventprobs : ndarray
            Probabilities with shape max_samples*n_trials*n_events
        """

        n_conds = self.mags_map.shape[0]
        likes_events_cond = []

        if self.cpus > 1:
            with mp.Pool(processes=self.cpus) as pool:
                likes_events_cond = pool.starmap(
                    self.estim_probs,
                    zip(
                        [self.magnitudes[c, self.mags_map[c, :] >= 0, :] for c in range(n_conds)],
                        [self.parameters[c, self.pars_map[c, :] >= 0, :] for c in range(n_conds)],
                        [self.locations[c, self.pars_map[c, :] >= 0] for c in range(n_conds)],
                        itertools.repeat(None),
                        [self.conds == c for c in range(n_conds)],
                        itertools.repeat(False),
                    ),
                )
        else:
            for c in range(n_conds):
                magnitudes_cond = self.magnitudes[
                    c, self.mags_map[c, :] >= 0, :
                ]  # select existing magnitudes
                parameters_cond = self.parameters[
                    c, self.pars_map[c, :] >= 0, :
                ]  # select existing params
                locations_cond = self.locations[c, self.pars_map[c, :] >= 0]
                likes_events_cond.append(
                    self.estim_probs(
                        magnitudes_cond,
                        parameters_cond,
                        locations_cond,
                        subset_epochs=(self.conds == c),
                    )
                )

        likelihood = np.sum([x[0] for x in likes_events_cond])
        eventprobs = np.zeros((self.max_d, len(self.conds), self.mags_map.shape[1]))
        for c in range(n_conds):
            eventprobs[np.ix_(range(self.max_d), self.conds == c, self.mags_map[c, :] >= 0)] = (
                likes_events_cond[c][1]
            )

        if lkh_only:
            return likelihood
        else:
            return [likelihood, eventprobs]

    def get_magnitudes_parameters_expectation(self, eventprobs, subset_epochs=None):
        n_events = eventprobs.shape[2]
        n_trials = eventprobs.shape[1]
        if subset_epochs is None:  # all trials
            subset_epochs = range(n_trials)

        magnitudes = np.zeros((n_events, self.n_dims))

        # Magnitudes from Expectation
        for event in range(n_events):
            for comp in range(self.n_dims):
                magnitudes[event, comp] = np.mean(
                    np.sum(
                        eventprobs[:, :, event]
                        * self.data_matrix[:, subset_epochs, comp],
                        axis=0,
                    )
                )
            # scale cross-correlation with likelihood of the transition
            # sum by-trial these scaled activation for each transition events
            # average across trials

        # Gamma parameters from Expectation
        # calc averagepos here as mean_d can be condition dependent, whereas scale_parameters() assumes it's general
        event_times_mean = np.concatenate(
            [
                np.arange(self.max_d) @ eventprobs.mean(axis=1),
                [np.mean(self.durations[subset_epochs]) - 1],
            ]
        )
        parameters = self.scale_parameters(
            eventprobs=None, n_events=n_events, averagepos=event_times_mean
        )

        return [magnitudes, parameters]

    # def get_locations(
    #     self,
    #     locations,
    #     magnitudes,
    #     parameters,
    #     parameters_prev,
    #     eventprobs,
    #     subset_epochs=None,
    # ):
    #     # if no correlation threshold is set, locations are set for all stages except first and last stage
    #     # to self.location (this function should not even be called in that case)
    #     # else, locations are only set for stages that exceed location_corr_threshold

    #     n_events = magnitudes.shape[magnitudes.ndim - 2]

    #     if self.location_corr_threshold is None:
    #         locations[1:-1] = self.location
    #     else:
    #         if (
    #             n_events > 1 and not (magnitudes == 0).all()
    #         ):  # if not on first iteration

    #             topos = self.compute_topos_locations(
    #                 eventprobs, subset_epochs
    #             )  # compute the topologies of the events

    #             corr = np.corrcoef(topos.T)[
    #                 :-1, 1:
    #             ].diagonal()  # only interested in sequential corrs
    #             stage_durations = np.array(
    #                 [self.scale_to_mean(x[0], x[1]) for x in parameters[1:-1, :]]
    #             )
    #             stage_durations_prev = np.array(
    #                 [self.scale_to_mean(x[0], x[1]) for x in parameters_prev[1:-1, :]]
    #             )

    #             for ev in range(n_events - 1):
    #                 # high correlation and moving away from each other,
    #                 if (
    #                     corr[ev] > self.location_corr_threshold
    #                     and stage_durations[ev] - stage_durations_prev[ev] < 0.1
    #                 ):
    #                     # and either close to each other or location_corr_duration is None
    #                     if (
    #                         self.location_corr_duration is None
    #                         or stage_durations[ev]
    #                         < self.location_corr_duration / self.steps
    #                     ):
    #                         locations[ev + 1] += 1  # indexes stages
    #     return locations

    def distribution_pmf(self, shape, scale):
        '''
        Returns PMF for a provided scipy disttribution with shape and scale, on a range from 0 to max_length 
        
        parameters
        ----------
        shape : float
            shape parameter
        scale : float
            scale parameter     
        Returns
        -------
        p : ndarray
            probabilty mass function for the distribution with given scale
        '''
        if scale == 0:
            warn('Convergence failed: one stage has been found to be null')
            p = np.repeat(0,self.max_d)
        else:
            p = self.pdf(np.arange(self.max_d), shape, scale=scale)
            p = p/np.sum(p)
            p[np.isnan(p)] = 0 #remove potential nans
        return p
    
    def scale_parameters(self, eventprobs=None, n_events=None, averagepos=None):
        '''
        Used for the re-estimation in the EM procdure. The likeliest location of 
        the event is computed from eventprobs. The scale parameter are then taken as the average 
        distance between the events
        
        parameters
        ----------
        eventprobs : ndarray
            [samples(max_d)*n_trials*n_events] = [max_d*trials*nTransition events]
        durations : ndarray
            1D array of trial length
        mags : ndarray
            2D ndarray components * nTransition events, initial conditions for events magnitudes
        shape : float
            shape parameter for the gamma, defaults to 2  
        
        Returns
        -------
        params : ndarray
            shape and scale for the gamma distributions
        '''
        params = np.zeros((n_events+1,2), dtype=np.float64)
        params[:,0] = self.shape
        params[:,1] = np.diff(averagepos, prepend=0)
        params[:,1] = [self.mean_to_scale(x[1],x[0]) for x in params]
        return params