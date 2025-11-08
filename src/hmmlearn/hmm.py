import warnings
from scipy import linalg
from sklearn.utils.validation import check_array, check_is_fitted, check_random_state
from tqdm import tqdm
import numpy as np
from sklearn.base import BaseEstimator
from . import _hmmc, _utils
from .utils import (
    normalize,
    log_normalize,
    normalize_by_indexes,
)

from hmmlearn.converge_monitor import ConvergenceMonitor, _log


DECODER_ALGORITHMS = frozenset(("viterbi", "map"))


class CategoricalHMM(BaseEstimator):
    """
    Base class for Hidden Markov Models learned from Expectation-Maximization.

    This class allows for easy evaluation of, sampling from, and maximum a
    posteriori estimation of the parameters of a HMM.

    Attributes
    ----------
    monitor_ : ConvergenceMonitor
        Monitor object used to check the convergence of EM.
    startprob_ : array, shape (n_components, )
        Initial state occupation distribution.
    transmat_ : array, shape (n_components, n_components)
        Matrix of transition probabilities between states.

    Notes
    -----
    Normally, one should use a subclass of `.BaseHMM`, with its specialization
    towards a given emission model.  In rare cases, the base class can also be
    useful in itself, if one simply wants to generate a sequence of states
    using `.BaseHMM.sample`.  In that case, the feature matrix will have zero
    features.
    """

    def __init__(
        self,
        n_states=1,
        startprob_prior=1e-6,
        transmat_prior=1e-6,
        *,
        emissionprob_prior=1e-6,
        algorithm="viterbi",
        random_state=None,
        n_iter=10,
        tol=1e-2,
        verbose=False,
        params="ste",
        init_params="ste",
        implementation="log",
        n_clusters=1,
        dropout_rate=0.0,
    ):
        """
        Parameters
        ----------
        n_states : int
            Number of states.

        startprob_prior : array, shape (n_components, ), optional
            Parameters of the Dirichlet prior distribution for
            :attr:`startprob_`.

        transmat_prior : array, shape (n_components, n_components), optional
            Parameters of the Dirichlet prior distribution for each row
            of the transition probabilities :attr:`transmat_`.

        emissionprob_prior : array, shape (n_components, n_features), optional
            Parameters of the Dirichlet prior distribution for
            :attr:`emissionprob_`.

        algorithm : {"viterbi", "map"}, optional
            Decoder algorithm.

            - "viterbi": finds the most likely sequence of states, given all
              emissions.
            - "map" (also known as smoothing or forward-backward): finds the
              sequence of the individual most-likely states, given all
              emissions.

        random_state: RandomState or an int seed, optional
            A random number generator instance.

        n_iter : int, optional
            Maximum number of iterations to perform.

        tol : float, optional
            Convergence threshold. EM will stop if the gain in log-likelihood
            is below this value.

        verbose : bool, optional
            Whether per-iteration convergence reports are printed to
            :data:`sys.stderr`.  Convergence can also be diagnosed using the
            :attr:`monitor_` attribute.

        params, init_params : string, optional
            The parameters that get updated during (``params``) or initialized
            before (``init_params``) the training.  Can contain any
            combination of 's' for startprob, 't' for transmat, and 'e' for
            emissionprob.  Defaults to all parameters.

        implementation : string, optional
            Determines if the forward-backward algorithm is implemented with
            logarithms ("log"), or using scaling ("scaling").  The default is
            to use logarithms for backwards compatability.

        n_clusters : int, optional
        """

        self.n_states = n_states
        self.params = params
        self.init_params = init_params
        self.algorithm = algorithm
        self.n_iter = n_iter
        self.tol = tol
        self.verbose = verbose
        self.implementation = implementation
        self.random_state = random_state
        self.startprob_prior = startprob_prior
        self.transmat_prior = transmat_prior
        self.monitor_ = ConvergenceMonitor(self.tol, self.n_iter, self.verbose)
        self.emissionprob_prior = emissionprob_prior
        self.n_tokens = None
        self.n_emit_components = None
        self.n_clusters = n_clusters
        assert self.n_states % self.n_clusters == 0, (
            "n_components must be divisible by n_clusters"
        )
        self.clusters_offset = None
        self.dropout_rate = dropout_rate

    def score_samples(self, X, lengths=None):
        """
        Compute the log probability under the model and compute posteriors.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix of individual samples.
        lengths : array-like of integers, shape (n_sequences, ), optional
            Lengths of the individual sequences in ``X``. The sum of
            these should be ``n_samples``.

        Returns
        -------
        log_prob : float
            Log likelihood of ``X``.
        posteriors : array, shape (n_samples, n_components)
            State-membership probabilities for each sample in ``X``.

        See Also
        --------
        score : Compute the log probability under the model.
        decode : Find most likely state sequence corresponding to ``X``.
        """
        return self._score(X, lengths, compute_posteriors=True)

    def perplexity(self, X, lengths=None):
        return np.exp(self.average_loss(X, lengths))

    def average_loss(self, X, lengths=None):
        return self.loss(X, lengths) / len(X)

    def loss(self, X, lengths=None):
        return -self.score(X, lengths)

    def score(self, X, lengths=None):
        """
        Compute the log probability under the model.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix of individual samples.
        lengths : array-like of integers, shape (n_sequences, ), optional
            Lengths of the individual sequences in ``X``. The sum of
            these should be ``n_samples``.

        Returns
        -------
        log_prob : float
            Log likelihood of ``X``.

        See Also
        --------
        score_samples : Compute the log probability under the model and
            posteriors.
        decode : Find most likely state sequence corresponding to ``X``.
        """
        return self._score(X, lengths, compute_posteriors=False)[0]

    def _score(self, X, lengths=None, *, compute_posteriors):
        """
        Helper for `score` and `score_samples`.

        Compute the log probability under the model, as well as posteriors if
        *compute_posteriors* is True (otherwise, an empty array is returned
        for the latter).
        """
        check_is_fitted(self, "startprob_")
        self._check()

        X = check_array(X)
        impl = {
            "scaling": self._score_scaling,
            "log": self._score_log,
        }[self.implementation]
        return impl(X=X, lengths=lengths, compute_posteriors=compute_posteriors)

    def _score_log(self, X, lengths=None, *, compute_posteriors):
        """
        Compute the log probability under the model, as well as posteriors if
        *compute_posteriors* is True (otherwise, an empty array is returned
        for the latter).
        """
        log_prob = 0
        sub_posteriors = [np.empty((0, self.n_states))]
        for sub_X in _utils.split_X_lengths(X, lengths):
            log_frameprob = self._compute_log_likelihood(sub_X)
            clusters_offset = self._compute_offsets(sub_X)
            log_probij, fwdlattice = _hmmc.forward_log(
                self.startprob_, self.transmat_, log_frameprob, clusters_offset
            )
            log_prob += log_probij
            if compute_posteriors:
                bwdlattice = _hmmc.backward_log(
                    self.startprob_, self.transmat_, log_frameprob, clusters_offset
                )
                sub_posteriors.append(
                    self._compute_posteriors_log(fwdlattice, bwdlattice)
                )
        return log_prob, np.concatenate(sub_posteriors)

    def _score_scaling(self, X, lengths=None, *, compute_posteriors):
        log_prob = 0
        sub_posteriors = [np.empty((0, self.n_states))]
        for sub_X in _utils.split_X_lengths(X, lengths):
            frameprob = self._compute_likelihood(sub_X)
            clusters_offset = self._compute_offsets(sub_X)
            log_probij, fwdlattice, scaling_factors = _hmmc.forward_scaling(
                self.startprob_, self.transmat_, frameprob, clusters_offset
            )
            log_prob += log_probij
            if compute_posteriors:
                bwdlattice = _hmmc.backward_scaling(
                    self.startprob_,
                    self.transmat_,
                    frameprob,
                    clusters_offset,
                    scaling_factors,
                )
                sub_posteriors.append(
                    self._compute_posteriors_scaling(fwdlattice, bwdlattice)
                )

        return log_prob, np.concatenate(sub_posteriors)

    def _decode_viterbi(self, X):
        log_frameprob = self._compute_log_likelihood(X)
        return _hmmc.viterbi(self.startprob_, self.transmat_, log_frameprob)

    def _decode_map(self, X):
        _, posteriors = self.score_samples(X)
        log_prob = np.max(posteriors, axis=1).sum()
        state_sequence = np.argmax(posteriors, axis=1)
        return log_prob, state_sequence

    def decode(self, X, lengths=None, algorithm=None):
        """
        Find most likely state sequence corresponding to ``X``.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix of individual samples.
        lengths : array-like of integers, shape (n_sequences, ), optional
            Lengths of the individual sequences in ``X``. The sum of
            these should be ``n_samples``.
        algorithm : {"viterbi", "map"}, optional
            Decoder algorithm.

            - "viterbi": finds the most likely sequence of states, given all
              emissions.
            - "map" (also known as smoothing or forward-backward): finds the
              sequence of the individual most-likely states, given all
              emissions.
            If not given, :attr:`decoder` is used.

        Returns
        -------
        log_prob : float
            Log probability of the produced state sequence.
        state_sequence : array, shape (n_samples, )
            Labels for each sample from ``X`` obtained via a given
            decoder ``algorithm``.

        See Also
        --------
        score_samples : Compute the log probability under the model and
            posteriors.
        score : Compute the log probability under the model.
        """
        check_is_fitted(self, "startprob_")
        self._check()

        algorithm = algorithm or self.algorithm
        if algorithm not in DECODER_ALGORITHMS:
            raise ValueError(f"Unknown decoder {algorithm!r}")

        decoder = {"viterbi": self._decode_viterbi, "map": self._decode_map}[algorithm]

        X = check_array(X)
        log_prob = 0
        sub_state_sequences = []
        for sub_X in _utils.split_X_lengths(X, lengths):
            # XXX decoder works on a single sample at a time!
            sub_log_prob, sub_state_sequence = decoder(sub_X)
            log_prob += sub_log_prob
            sub_state_sequences.append(sub_state_sequence)

        return log_prob, np.concatenate(sub_state_sequences)

    def predict(self, X, lengths=None):
        """
        Find most likely state sequence corresponding to ``X``.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix of individual samples.
        lengths : array-like of integers, shape (n_sequences, ), optional
            Lengths of the individual sequences in ``X``. The sum of
            these should be ``n_samples``.

        Returns
        -------
        state_sequence : array, shape (n_samples, )
            Labels for each sample from ``X``.
        """
        _, state_sequence = self.decode(X, lengths)
        return state_sequence

    def predict_proba(self, X, lengths=None):
        """
        Compute the posterior probability for each state in the model.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix of individual samples.
        lengths : array-like of integers, shape (n_sequences, ), optional
            Lengths of the individual sequences in ``X``. The sum of
            these should be ``n_samples``.

        Returns
        -------
        posteriors : array, shape (n_samples, n_components)
            State-membership probabilities for each sample from ``X``.
        """
        _, posteriors = self.score_samples(X, lengths)
        return posteriors

    def sample(self, n_samples=1, random_state=None, currstate=None):
        """
        Generate random samples from the model.

        Parameters
        ----------
        n_samples : int
            Number of samples to generate.
        random_state : RandomState or an int seed
            A random number generator instance. If ``None``, the object's
            ``random_state`` is used.
        currstate : int
            Current state, as the initial state of the samples.

        Returns
        -------
        X : array, shape (n_samples, n_features)
            Feature matrix.
        state_sequence : array, shape (n_samples, )
            State sequence produced by the model.

        Examples
        --------
        ::

            # generate samples continuously
            _, Z = model.sample(n_samples=10)
            X, Z = model.sample(n_samples=10, currstate=Z[-1])
        """
        check_is_fitted(self, "startprob_")
        self._check()

        if random_state is None:
            random_state = self.random_state
        random_state = check_random_state(random_state)

        transmat_cdf = np.cumsum(self.transmat_, axis=1)

        if currstate is None:
            startprob_cdf = np.cumsum(self.startprob_)
            currstate = (startprob_cdf > random_state.rand()).argmax()

        state_sequence = [currstate]
        X = [self._generate_sample_from_state(currstate, random_state=random_state)]

        for t in range(n_samples - 1):
            currstate = (transmat_cdf[currstate] > random_state.rand()).argmax()
            state_sequence.append(currstate)
            X.append(
                self._generate_sample_from_state(currstate, random_state=random_state)
            )

        return np.atleast_2d(X), np.array(state_sequence, dtype=int)

    def fit(self, X, lengths=None, valid=None, valid_lengths=None):
        """
        Estimate model parameters.

        An initialization step is performed before entering the
        EM algorithm. If you want to avoid this step for a subset of
        the parameters, pass proper ``init_params`` keyword argument
        to estimator's constructor.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix of individual samples.
        lengths : array-like of integers, shape (n_sequences, )
            Lengths of the individual sequences in ``X``. The sum of
            these should be ``n_samples``.

        Returns
        -------
        self : object
            Returns self.
        """
        X = check_array(X)
        if lengths is None:
            lengths = np.asarray([X.shape[0]])

        self._init(X, lengths)
        self._check()
        self.monitor_._reset()

        for _ in range(self.n_iter):
            stats, curr_logprob = self._do_estep(X, lengths)

            # XXX must be before convergence check, because otherwise
            #     there won't be any updates for the case ``n_iter=1``.
            self._do_mstep(stats)

            perplexity = self.perplexity(X, lengths)
            print(f"Train Perplexity: {perplexity}")
            if valid is not None:
                perplexity = self.perplexity(valid, valid_lengths)
                print(f"Valid Perplexity: {perplexity}")

            self.monitor_.report(curr_logprob)
            # if self.monitor_.converged:
            #     break

            if (self.transmat_.sum(axis=1) == 0).any():
                _log.warning(
                    "Some rows of transmat_ have zero sum because no "
                    "transition from the state was ever observed."
                )
        return self

    def _choose_active(self, frameprob, X, dropout_rate):
        unique_vals, inverse_indices = np.unique(X, return_inverse=True)
        unique_masks = np.random.rand(len(unique_vals), frameprob.shape[1]) <= dropout_rate
        mask = unique_masks[inverse_indices.ravel(), :]
        frameprob[mask] = 0

    def _fit_scaling(self, X, clusters_offset):
        frameprob = self._compute_likelihood(X)
        self._choose_active(frameprob, X, self.dropout_rate)

        log_prob, fwdlattice, scaling_factors = _hmmc.forward_scaling(
            self.startprob_, self.transmat_, frameprob, clusters_offset
        )
        bwdlattice = _hmmc.backward_scaling(
            self.startprob_,
            self.transmat_,
            frameprob,
            clusters_offset,
            scaling_factors,
        )
        posteriors = self._compute_posteriors_scaling(fwdlattice, bwdlattice)
        return frameprob, log_prob, posteriors, fwdlattice, bwdlattice

    def _fit_log(self, X, clusters_offset):
        log_frameprob = self._compute_log_likelihood(X)
        log_prob, fwdlattice = _hmmc.forward_log(
            self.startprob_, self.transmat_, log_frameprob, clusters_offset
        )
        bwdlattice = _hmmc.backward_log(
            self.startprob_, self.transmat_, log_frameprob, clusters_offset
        )
        posteriors = self._compute_posteriors_log(fwdlattice, bwdlattice)
        return log_frameprob, log_prob, posteriors, fwdlattice, bwdlattice

    def _compute_posteriors_scaling(self, fwdlattice, bwdlattice):
        posteriors = fwdlattice * bwdlattice
        normalize(posteriors, axis=1)
        return posteriors

    def _compute_posteriors_log(self, fwdlattice, bwdlattice):
        # gamma is guaranteed to be correctly normalized by log_prob at
        # all frames, unless we do approximate inference using pruning.
        # So, we will normalize each frame explicitly in case we
        # pruned too aggressively.
        log_gamma = fwdlattice + bwdlattice
        log_normalize(log_gamma, axis=1)
        with np.errstate(under="ignore"):
            return np.exp(log_gamma)

    def _needs_init(self, code, name):
        if code in self.init_params:
            if hasattr(self, name):
                _log.warning(
                    "Even though the %r attribute is set, it will be "
                    "overwritten during initialization because 'init_params' "
                    "contains %r",
                    name,
                    code,
                )
            return True
        if not hasattr(self, name):
            return True
        return False

    def _check_and_set_n_features(self, X):
        """
        Check if ``X`` is a sample from a categorical distribution, i.e. an
        array of non-negative integers.
        """
        if not np.issubdtype(X.dtype, np.integer):
            raise ValueError("Symbols should be integers")
        if X.min() < 0:
            raise ValueError("Symbols should be nonnegative")
        if self.n_tokens is not None:
            if self.n_tokens - 1 < X.max():
                raise ValueError(
                    f"Largest symbol is {X.max()} but the model only emits "
                    f"symbols up to {self.n_tokens - 1}"
                )
        else:
            self.n_tokens = X.max() + 1

    def _check_sum_1(self, name):
        """Check that an array describes one or more distributions."""
        s = getattr(self, name).sum(axis=-1)
        if not np.allclose(s, self.n_clusters):
            raise ValueError(
                f"{name} must sum to 1 (got {s:.4f})"
                if s.ndim == 0
                else f"{name} rows must sum to 1 (got row sums of {s})"
                if s.ndim == 1
                else "Expected 1D or 2D array"
            )

    def _check(self):
        """
        Validate model parameters prior to fitting.

        Raises
        ------
        ValueError
            If any of the parameters are invalid, e.g. if :attr:`startprob_`
            don't sum to 1.
        """
        self.startprob_ = np.asarray(self.startprob_)
        if len(self.startprob_) != self.n_states:
            raise ValueError("startprob_ must have length n_components")
        # self._check_sum_1("startprob_")

        self.transmat_ = np.asarray(self.transmat_)
        if self.transmat_.shape != (self.n_states, self.n_states):
            raise ValueError("transmat_ must have shape (n_components, n_components)")
        # self._check_sum_1("transmat_")

        self.emissionprob_ = np.atleast_2d(self.emissionprob_)
        if self.n_tokens is None:
            self.n_tokens = self.emissionprob_.shape[1]
        if self.emissionprob_.shape != (self.n_emit_components, self.n_tokens):
            raise ValueError(
                f"emissionprob_ must have shape({self.n_emit_components}, {self.n_tokens})"
            )
        # self._check_sum_1("emissionprob_")

    def _compute_likelihood(self, X):
        return self.emissionprob_[:, X.squeeze(1)].T

    def _compute_offsets(self, X):
        return self.clusters_offset[X.squeeze(1)].T

    def _compute_log_likelihood(self, X):
        """
        Compute per-component emission log probability under the model.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix of individual samples.

        Returns
        -------
        log_prob : array, shape (n_samples, n_components)
            Emission log probability of each sample in ``X`` for each of the
            model states, i.e., ``log(p(X|state))``.
        """
        likelihood = self._compute_likelihood(X)
        with np.errstate(divide="ignore"):
            return np.log(likelihood)

    def _generate_sample_from_state(self, state, random_state=None):
        cdf = np.cumsum(self.emissionprob_[state, :])
        random_state = check_random_state(random_state)
        return [(cdf > random_state.rand()).argmax()]

    def _initialize_sufficient_statistics(self):
        """
        Initialize sufficient statistics required for M-step.

        The method is *pure*, meaning that it doesn't change the state of
        the instance.  For extensibility computed statistics are stored
        in a dictionary.

        Returns
        -------
        nobs : int
            Number of samples in the data.
        start : array, shape (n_components, )
            An array where the i-th element corresponds to the posterior
            probability of the first sample being generated by the i-th state.
        trans : array, shape (n_components, n_components)
            An array where the (i, j)-th element corresponds to the posterior
            probability of transitioning between the i-th to j-th states.
        """
        stats = {
            "nobs": 0,
            "start": np.zeros(self.n_states),
            "trans": np.zeros((self.n_states, self.n_states)),
            "obs": np.zeros((self.n_emit_components, self.n_tokens)),
        }
        return stats

    def _accumulate_sufficient_statistics(
        self, stats, X, lattice, posteriors, fwdlattice, bwdlattice, clusters_offset
    ):
        """
        Update sufficient statistics from a given sample.

        Parameters
        ----------
        stats : dict
            Sufficient statistics as returned by
            :meth:`~.BaseHMM._initialize_sufficient_statistics`.

        X : array, shape (n_samples, n_features)
            Sample sequence.

        lattice : array, shape (n_samples, n_components)
            Probabilities OR Log Probabilities of each sample
            under each of the model states.  Depends on the choice
            of implementation of the Forward-Backward algorithm

        posteriors : array, shape (n_samples, n_components)
            Posterior probabilities of each sample being generated by each
            of the model states.

        fwdlattice, bwdlattice : array, shape (n_samples, n_components)
            forward and backward probabilities.
        """

        impl = {
            "scaling": self._accumulate_sufficient_statistics_scaling,
            "log": self._accumulate_sufficient_statistics_log,
        }[self.implementation]

        impl(
            stats=stats,
            X=X,
            lattice=lattice,
            posteriors=posteriors,
            fwdlattice=fwdlattice,
            bwdlattice=bwdlattice,
            clusters_offset=clusters_offset,
        )

        if "e" in self.params:
            if X.shape[1] != 1:
                warnings.warn(
                    "Inputs of shape other than (n_samples, 1) are deprecated.",
                    DeprecationWarning,
                )
                X = np.concatenate(X)[:, None]
            np.add.at(stats["obs"].T, X.squeeze(1), posteriors)

    def _accumulate_sufficient_statistics_scaling(
        self, stats, X, lattice, posteriors, fwdlattice, bwdlattice, clusters_offset
    ):
        """
        Implementation of `_accumulate_sufficient_statistics`
        for ``implementation = "log"``.
        """
        stats["nobs"] += 1
        if "s" in self.params:
            offset = clusters_offset[0]
            stats["start"][offset : offset + self.n_emit_components] += posteriors[0]
        if "t" in self.params:
            n_samples, n_components = lattice.shape
            # when the sample is of length 1, it contains no transitions
            # so there is no reason to update our trans. matrix estimate
            if n_samples <= 1:
                return

            _hmmc.compute_scaling_xi_sum(
                fwdlattice,
                self.transmat_,
                bwdlattice,
                lattice,
                stats["trans"],
                clusters_offset,
            )

    def _accumulate_sufficient_statistics_log(
        self, stats, X, lattice, posteriors, fwdlattice, bwdlattice, clusters_offset
    ):
        """
        Implementation of `_accumulate_sufficient_statistics`
        for ``implementation = "log"``.
        """
        stats["nobs"] += 1
        if "s" in self.params:
            offset = clusters_offset[0]
            stats["start"][offset : offset + self.n_emit_components] += posteriors[0]
        if "t" in self.params:
            n_samples, n_components = lattice.shape
            # when the sample is of length 1, it contains no transitions
            # so there is no reason to update our trans. matrix estimate
            if n_samples <= 1:
                return

            log_xi_sum = _hmmc.compute_log_xi_sum(
                fwdlattice, self.transmat_, bwdlattice, lattice, clusters_offset
            )
            with np.errstate(under="ignore"):
                stats["trans"] += np.exp(log_xi_sum)

    def _do_mstep(self, stats):
        """
        Perform the M-step of EM algorithm.

        Parameters
        ----------
        stats : dict
            Sufficient statistics updated from all available samples.
        """
        # If a prior is < 1, `prior - 1 + starts['start']` can be negative.  In
        # that case maximization of (n1+e1) log p1 + ... + (ns+es) log ps under
        # the conditions sum(p) = 1 and all(p >= 0) show that the negative
        # terms can just be set to zero.
        # The ``np.where`` calls guard against updating forbidden states
        # or transitions in e.g. a left-right HMM.
        if "s" in self.params:
            self.startprob_ = stats["start"] + self.startprob_prior
            normalize_by_indexes(self.startprob_, self.cluster2states)

        if "t" in self.params:
            self.transmat_ = stats["trans"] + self.transmat_prior
            normalize(self.transmat_, axis=1)

        if "e" in self.params:
            self.emissionprob_ = stats["obs"] + self.emissionprob_prior
            normalize_by_indexes(self.emissionprob_, self.cluster2tokens, axis=1)

    def _do_estep(self, X, lengths):
        impl = {
            "scaling": self._fit_scaling,
            "log": self._fit_log,
        }[self.implementation]

        stats = self._initialize_sufficient_statistics()
        curr_logprob = 0
        for sub_X in tqdm(_utils.split_X_lengths(X, lengths)):
            clusters_offset = self._compute_offsets(sub_X)
            lattice, logprob, posteriors, fwdlattice, bwdlattice = impl(
                sub_X, clusters_offset
            )
            # Derived HMM classes will implement the following method to
            # update their probability distributions, so keep
            # a single call to this method for simplicity.
            self._accumulate_sufficient_statistics(
                stats,
                sub_X,
                lattice,
                posteriors,
                fwdlattice,
                bwdlattice,
                clusters_offset,
            )
            curr_logprob += logprob
        return stats, curr_logprob

    def get_stationary_distribution(self):
        """Compute the stationary distribution of states."""
        # The stationary distribution is proportional to the left-eigenvector
        # associated with the largest eigenvalue (i.e., 1) of the transition
        # matrix.
        check_is_fitted(self, "transmat_")
        eigvals, eigvecs = linalg.eig(self.transmat_.T)
        eigvec = np.real_if_close(eigvecs[:, np.argmax(eigvals)])
        return eigvec / eigvec.sum()

    def _init(self, X, lengths=None):
        """
        Initialize model parameters prior to fitting.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix of individual samples.
        """
        self._check_and_set_n_features(X)
        self.n_emit_components = self.n_states // self.n_clusters
        random_state = check_random_state(self.random_state)

        if self._needs_init("e", "emissionprob_"):
            self.clusters_offset = np.sort(
                np.random.choice(self.n_clusters, self.n_tokens)
                * self.n_emit_components
            )
            self.cluster2tokens = [
                np.where(self.clusters_offset == i * self.n_emit_components)[0]
                for i in range(self.n_clusters)
            ]
            self.emissionprob_ = random_state.rand(
                self.n_emit_components, self.n_tokens
            )
            normalize_by_indexes(self.emissionprob_, self.cluster2tokens, axis=1)

        if self._needs_init("s", "startprob_"):
            # init = 1.0 / self.n_emit_components
            init = 1.0
            self.startprob_ = np.array(
                list(
                    random_state.dirichlet(np.full(self.n_emit_components, init))
                    for _ in range(self.n_clusters)
                )
            ).flatten()
            cluster2states_2d = np.arange(
                self.n_clusters * self.n_emit_components
            ).reshape(self.n_clusters, self.n_emit_components)
            self.cluster2states = [
                cluster2states for cluster2states in cluster2states_2d
            ]
            normalize_by_indexes(self.startprob_, self.cluster2states)

        if self._needs_init("t", "transmat_"):
            alpha = 1e-3
            init = 1.0 / self.n_states + alpha
            self.transmat_ = random_state.dirichlet(
                np.full(self.n_states, init), size=self.n_states
            )
