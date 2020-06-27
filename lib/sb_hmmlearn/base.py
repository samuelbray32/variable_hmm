import logging
import string
import sys
from collections import deque

import numpy as np
from scipy.special import logsumexp
from sklearn.base import BaseEstimator
from sklearn.utils import check_array, check_random_state
import scipy.optimize
from . import _hmmc, _utils
from .utils import normalize, log_normalize, iter_from_X_lengths, log_mask_zero


_log = logging.getLogger(__name__)
#: Supported decoder algorithms.
DECODER_ALGORITHMS = frozenset(("viterbi", "map"))


class ConvergenceMonitor:
    """Monitors and reports convergence to :data:`sys.stderr`.

    Parameters
    ----------
    tol : double
        Convergence threshold. EM has converged either if the maximum
        number of iterations is reached or the log probability
        improvement between the two consecutive iterations is less
        than threshold.

    n_iter : int
        Maximum number of iterations to perform.

    verbose : bool
        If ``True`` then per-iteration convergence reports are printed,
        otherwise the monitor is mute.

    Attributes
    ----------
    history : deque
        The log probability of the data for the last two training
        iterations. If the values are not strictly increasing, the
        model did not converge.

    iter : int
        Number of iterations performed while training the model.

    Examples
    --------
    Use custom convergence criteria by subclassing ``ConvergenceMonitor``
    and redefining the ``converged`` method. The resulting subclass can
    be used by creating an instance and pointing a model's ``monitor_``
    attribute to it prior to fitting.

    >>> from hmmlearn.base import ConvergenceMonitor
    >>> from hmmlearn import hmm
    >>>
    >>> class ThresholdMonitor(ConvergenceMonitor):
    ...     @property
    ...     def converged(self):
    ...         return (self.iter == self.n_iter or
    ...                 self.history[-1] >= self.tol)
    >>>
    >>> model = hmm.GaussianHMM(n_components=2, tol=5, verbose=True)
    >>> model.monitor_ = ThresholdMonitor(model.monitor_.tol,
    ...                                   model.monitor_.n_iter,
    ...                                   model.monitor_.verbose)
    """
    _template = "{iter:>10d} {logprob:>16.4f} {delta:>+16.4f}"

    def __init__(self, tol, n_iter, verbose):
        self.tol = tol
        self.n_iter = n_iter
        self.verbose = verbose
        self.history = deque(maxlen=2)
        self.iter = 0

    def __repr__(self):
        class_name = self.__class__.__name__
        params = sorted(dict(vars(self), history=list(self.history)).items())
        return ("{}(\n".format(class_name)
                + "".join(map("    {}={},\n".format, *zip(*params)))
                + ")")

    def _reset(self):
        """Reset the monitor's state."""
        self.iter = 0
        self.history.clear()

    def report(self, logprob):
        """Reports convergence to :data:`sys.stderr`.

        The output consists of three columns: iteration number, log
        probability of the data at the current iteration and convergence
        rate.  At the first iteration convergence rate is unknown and
        is thus denoted by NaN.

        Parameters
        ----------
        logprob : float
            The log probability of the data as computed by EM algorithm
            in the current iteration.
        """
        if self.verbose:
            delta = logprob - self.history[-1] if self.history else np.nan
            message = self._template.format(
                iter=self.iter + 1, logprob=logprob, delta=delta)
            print(message, file=sys.stderr)

        self.history.append(logprob)
        self.iter += 1

    @property
    def converged(self):
        """``True`` if the EM algorithm converged and ``False`` otherwise."""
        # XXX we might want to check that ``logprob`` is non-decreasing.
        return (self.iter == self.n_iter or
                (len(self.history) == 2 and
                 self.history[1] - self.history[0] < self.tol))


class _BaseHMM(BaseEstimator):
    r"""Base class for Hidden Markov Models.

    This class allows for easy evaluation of, sampling from, and
    maximum a posteriori estimation of the parameters of a HMM.

    See the instance documentation for details specific to a
    particular object.

    Parameters
    ----------
    n_components : int
        Number of states in the model.

    startprob_prior : array, shape (n_components, ), optional
        Parameters of the Dirichlet prior distribution for
        :attr:`startprob_`.

    transmat_prior : array, shape (n_components, n_components), optional
        Parameters of the Dirichlet prior distribution for each row
        of the transition probabilities :attr:`transmat_`.

    algorithm : string, optional
        Decoder algorithm. Must be one of "viterbi" or "map".
        Defaults to "viterbi".

    random_state: RandomState or an int seed, optional
        A random number generator instance.

    n_iter : int, optional
        Maximum number of iterations to perform.

    tol : float, optional
        Convergence threshold. EM will stop if the gain in log-likelihood
        is below this value.

    verbose : bool, optional
        When ``True`` per-iteration convergence reports are printed
        to :data:`sys.stderr`. You can diagnose convergence via the
        :attr:`monitor_` attribute.

    params : string, optional
        Controls which parameters are updated in the training
        process.  Can contain any combination of 's' for startprob,
        't' for transmat, and other characters for subclass-specific
        emission parameters. Defaults to all parameters.

    init_params : string, optional
        Controls which parameters are initialized prior to
        training.  Can contain any combination of 's' for
        startprob, 't' for transmat, and other characters for
        subclass-specific emission parameters. Defaults to all
        parameters.

    Attributes
    ----------
    monitor\_ : ConvergenceMonitor
        Monitor object used to check the convergence of EM.

    startprob\_ : array, shape (n_components, )
        Initial state occupation distribution.

    transmat\_ : array, shape (n_components, n_components)
        Matrix of transition probabilities between states.
    """
    def __init__(self, n_components=1,
                 startprob_prior=1.0, transmat_prior=1.0,
                 algorithm="viterbi", random_state=None,
                 n_iter=10, tol=1e-2, verbose=False,
                 params=string.ascii_letters,
                 init_params=string.ascii_letters,
                 A1 = None,
                 a = 0,
                 b = 1,
                 grad_iter = 100,
                 grad_conv = 10**-2,
                 grad_lr = 10**-2,
                 t0 = 0,
                 transmat_ = None,
                 startprob_ = None):
        self.n_components = n_components
        self.params = params
        self.init_params = init_params
        self.startprob_prior = startprob_prior
        self.transmat_prior = transmat_prior
        self.algorithm = algorithm
        self.random_state = random_state
        self.n_iter = n_iter
        self.tol = tol
        self.verbose = verbose
        self.monitor_ = ConvergenceMonitor(self.tol, self.n_iter, self.verbose)
        # added samuelbray32
        self.A1 = A1 #time dependent transition matrix
        self.a = a #parameters of sigmoid function
        self.b = b
        self.grad_iter = grad_iter #stop criteria for _grad_descent
        self.grad_conv = grad_conv
        self.grad_lr = grad_lr
        self.t0 = t0
        self.transmat_ = transmat_
        self.startprob_ = startprob_
        self.track_params = []
        self.grad_method = 'newtons_linesearch'

    def get_stationary_distribution(self):
        """Compute the stationary distribution of states.
        """
        # The stationary distribution is proportional to the left-eigenvector
        # associated with the largest eigenvalue (i.e., 1) of the transition
        # matrix.
        _utils.check_is_fitted(self, "transmat_")
        eigvals, eigvecs = np.linalg.eig(self.transmat_.T)
        eigvec = np.real_if_close(eigvecs[:, np.argmax(eigvals)])
        return eigvec / eigvec.sum()

    def score_samples(self, X, lengths=None):
        """Compute the log probability under the model and compute posteriors.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix of individual samples.

        lengths : array-like of integers, shape (n_sequences, ), optional
            Lengths of the individual sequences in ``X``. The sum of
            these should be ``n_samples``.

        Returns
        -------
        logprob : float
            Log likelihood of ``X``.

        posteriors : array, shape (n_samples, n_components)
            State-membership probabilities for each sample in ``X``.

        See Also
        --------
        score : Compute the log probability under the model.
        decode : Find most likely state sequence corresponding to ``X``.
        """
        _utils.check_is_fitted(self, "startprob_")
        self._check()

        X = check_array(X)
        n_samples = X.shape[0]
        logprob = 0
        posteriors = np.zeros((n_samples, self.n_components))
        for i, j in iter_from_X_lengths(X, lengths):
            framelogprob = self._compute_log_likelihood(X[i:j])
            logprobij, fwdlattice = self._do_forward_pass(framelogprob)
            logprob += logprobij

            bwdlattice = self._do_backward_pass(framelogprob)
            posteriors[i:j] = self._compute_posteriors(fwdlattice, bwdlattice)
        return logprob, posteriors

    def score(self, X, lengths=None):
        """Compute the log probability under the model.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix of individual samples.

        lengths : array-like of integers, shape (n_sequences, ), optional
            Lengths of the individual sequences in ``X``. The sum of
            these should be ``n_samples``.

        Returns
        -------
        logprob : float
            Log likelihood of ``X``.

        See Also
        --------
        score_samples : Compute the log probability under the model and
            posteriors.
        decode : Find most likely state sequence corresponding to ``X``.
        """
        _utils.check_is_fitted(self, "startprob_")
        self._check()

        X = check_array(X)
        # XXX we can unroll forward pass for speed and memory efficiency.
        logprob = 0
        for i, j in iter_from_X_lengths(X, lengths):
            framelogprob = self._compute_log_likelihood(X[i:j])
            logprobij, _fwdlattice = self._do_forward_pass(framelogprob)
            logprob += logprobij
        return logprob

    def _decode_viterbi(self, X):
        framelogprob = self._compute_log_likelihood(X)
        return self._do_viterbi_pass(framelogprob)

    def _decode_map(self, X):
        _, posteriors = self.score_samples(X)
        logprob = np.max(posteriors, axis=1).sum()
        state_sequence = np.argmax(posteriors, axis=1)
        return logprob, state_sequence

    def decode(self, X, lengths=None, algorithm=None):
        """Find most likely state sequence corresponding to ``X``.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix of individual samples.

        lengths : array-like of integers, shape (n_sequences, ), optional
            Lengths of the individual sequences in ``X``. The sum of
            these should be ``n_samples``.

        algorithm : string
            Decoder algorithm. Must be one of "viterbi" or "map".
            If not given, :attr:`decoder` is used.

        Returns
        -------
        logprob : float
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
        _utils.check_is_fitted(self, "startprob_")
        self._check()

        algorithm = algorithm or self.algorithm
        if algorithm not in DECODER_ALGORITHMS:
            raise ValueError("Unknown decoder {!r}".format(algorithm))

        decoder = {
            "viterbi": self._decode_viterbi,
            "map": self._decode_map
        }[algorithm]

        X = check_array(X)
        n_samples = X.shape[0]
        logprob = 0
        state_sequence = np.empty(n_samples, dtype=int)
        for i, j in iter_from_X_lengths(X, lengths):
            # XXX decoder works on a single sample at a time!
            logprobij, state_sequenceij = decoder(X[i:j])
            logprob += logprobij
            state_sequence[i:j] = state_sequenceij

        return logprob, state_sequence

    def predict(self, X, lengths=None):
        """Find most likely state sequence corresponding to ``X``.

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
        """Compute the posterior probability for each state in the model.

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

    def sample(self, n_samples=1, random_state=None):
        """Generate random samples from the model.

        Parameters
        ----------
        n_samples : int
            Number of samples to generate.

        random_state : RandomState or an int seed
            A random number generator instance. If ``None``, the object's
            ``random_state`` is used.

        Returns
        -------
        X : array, shape (n_samples, n_features)
            Feature matrix.

        state_sequence : array, shape (n_samples, )
            State sequence produced by the model.
        """
        _utils.check_is_fitted(self, "startprob_")
        self._check()

        if random_state is None:
            random_state = self.random_state
        random_state = check_random_state(random_state)

        startprob_cdf = np.cumsum(self.startprob_)
        transmat_cdf = np.cumsum(self.transmat_, axis=1)

        currstate = (startprob_cdf > random_state.rand()).argmax()
        state_sequence = [currstate]
        X = [self._generate_sample_from_state(
            currstate, random_state=random_state)]

        for t in range(n_samples - 1):
            currstate = (transmat_cdf[currstate] > random_state.rand()) \
                .argmax()
            state_sequence.append(currstate)
            X.append(self._generate_sample_from_state(
                currstate, random_state=random_state))

        return np.atleast_2d(X), np.array(state_sequence, dtype=int)

    def fit(self, X, lengths=None):
        """Estimate model parameters.

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
        self._init(X, lengths=lengths)
        self._check()

        self.monitor_._reset()
        for iter in range(self.n_iter):
            print('E iteration: ', iter)
            stats = self._initialize_sufficient_statistics()
            curr_logprob = 0
            for i, j in iter_from_X_lengths(X, lengths):
                framelogprob = self._compute_log_likelihood(X[i:j])
                logprob, fwdlattice = self._do_forward_pass(framelogprob)
                curr_logprob += logprob
                bwdlattice = self._do_backward_pass(framelogprob)
                posteriors = self._compute_posteriors(fwdlattice, bwdlattice)
                self._accumulate_sufficient_statistics(
                    stats, X[i:j], framelogprob, posteriors, fwdlattice,
                    bwdlattice)
                print('fwd: ', fwdlattice[10])
                print('bwd: ', bwdlattice[10])
            # XXX must be before convergence check, because otherwise
            #     there won't be any updates for the case ``n_iter=1``.
            del fwdlattice
            del bwdlattice
            del framelogprob
            self._do_mstep(stats, posteriors)

            self.monitor_.report(curr_logprob)
            if self.monitor_.converged:
                break

        if (self.transmat_.sum(axis=1) == 0).any():
            _log.warning("Some rows of transmat_ have zero sum because no "
                         "transition from the state was ever observed.")

        return self

    def _do_viterbi_pass(self, framelogprob):
        n_samples, n_components = framelogprob.shape
        state_sequence, logprob = _hmmc._viterbi(
            n_samples, n_components, log_mask_zero(self.startprob_),
            log_mask_zero(self.transmat_), framelogprob)
        return logprob, state_sequence

    def _do_forward_pass(self, framelogprob):
        n_samples, n_components = framelogprob.shape
        fwdlattice = np.zeros((n_samples, n_components))
        # _hmmc._forward(n_samples, n_components,
        #                log_mask_zero(self.startprob_),
        #                self.transmat_,
        #                framelogprob, fwdlattice,
        #                self.A1,
        #                self.U(np.arange(n_samples)+self.t0))
        t = np.arange(n_samples)+self.t0
        log_trans = self.transmat_ + np.transpose(self.A1[:,:,None]*self.U(t),[2,0,1])
        log_trans[log_trans<0] = 10**-20
        print('trans: ', np.min(log_trans))
        log_trans = log_mask_zero(log_trans)
        print('log_trans: ', np.max(log_trans))
        _hmmc._forward(n_samples, n_components,
                       log_mask_zero(self.startprob_),
                       log_trans,
                       framelogprob, fwdlattice,)

        with np.errstate(under="ignore"):
            return logsumexp(fwdlattice[-1]), fwdlattice

    def _do_backward_pass(self, framelogprob):
        n_samples, n_components = framelogprob.shape
        bwdlattice = np.zeros((n_samples, n_components))
        # _hmmc._backward(n_samples, n_components,
        #                 log_mask_zero(self.startprob_),
        #                 self.transmat_,
        #                 framelogprob, bwdlattice,
        #                 self.A1,
        #                 self.U(np.arange(n_samples)+self.t0))
        t = np.arange(n_samples)+self.t0
        log_trans = self.transmat_ + np.transpose(self.A1[:,:,None]*self.U(t),[2,0,1])
        log_trans[log_trans<0] = 10**-20
        print('trans: ', np.min(log_trans))
        log_trans = log_mask_zero(log_trans)
        print('log_trans: ', np.max(log_trans))
        _hmmc._backward(n_samples, n_components,
                       log_mask_zero(self.startprob_),
                       log_trans,
                       framelogprob, bwdlattice,)
        return bwdlattice

    def _compute_posteriors(self, fwdlattice, bwdlattice):
        # gamma is guaranteed to be correctly normalized by logprob at
        # all frames, unless we do approximate inference using pruning.
        # So, we will normalize each frame explicitly in case we
        # pruned too aggressively.
        log_gamma = fwdlattice + bwdlattice
        log_normalize(log_gamma, axis=1)
        with np.errstate(under="ignore"):
            return np.exp(log_gamma)

    def _init(self, X, lengths):
        """Initializes model parameters prior to fitting.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix of individual samples.

        lengths : array-like of integers, shape (n_sequences, )
            Lengths of the individual sequences in ``X``. The sum of
            these should be ``n_samples``.
        """
        init = 1. / self.n_components
        if 's' in self.init_params or not hasattr(self, "startprob_"):
            self.startprob_ = np.full(self.n_components, init)
        if 't' in self.init_params or not hasattr(self, "transmat_"):
            self.transmat_ = np.full((self.n_components, self.n_components),
                                     init)
        n_fit_scalars_per_param = self._get_n_fit_scalars_per_param()
        n_fit_scalars = sum(n_fit_scalars_per_param[p] for p in self.params)
        if X.size < n_fit_scalars:
            _log.warning("Fitting a model with {} free scalar parameters with "
                         "only {} data points will result in a degenerate "
                         "solution.".format(n_fit_scalars, X.size))

    def _check(self):
        """Validates model parameters prior to fitting.

        Raises
        ------

        ValueError
            If any of the parameters are invalid, e.g. if :attr:`startprob_`
            don't sum to 1.
        """
        self.startprob_ = np.asarray(self.startprob_)
        if len(self.startprob_) != self.n_components:
            raise ValueError("startprob_ must have length n_components")
        if not np.allclose(self.startprob_.sum(), 1.0):
            raise ValueError("startprob_ must sum to 1.0 (got {:.4f})"
                             .format(self.startprob_.sum()))

        self.transmat_ = np.asarray(self.transmat_)
        if self.transmat_.shape != (self.n_components, self.n_components):
            raise ValueError(
                "transmat_ must have shape (n_components, n_components)")
        if not np.allclose(self.transmat_.sum(axis=1), 1.0):
            raise ValueError("rows of transmat_ must sum to 1.0 (got {})"
                             .format(self.transmat_.sum(axis=1)))

    def _compute_log_likelihood(self, X):
        """Computes per-component log probability under the model.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix of individual samples.

        Returns
        -------
        logprob : array, shape (n_samples, n_components)
            Log probability of each sample in ``X`` for each of the
            model states.
        """

    def _generate_sample_from_state(self, state, random_state=None):
        """Generates a random sample from a given component.

        Parameters
        ----------
        state : int
            Index of the component to condition on.

        random_state: RandomState or an int seed
            A random number generator instance. If ``None``, the object's
            ``random_state`` is used.

        Returns
        -------
        X : array, shape (n_features, )
            A random sample from the emission distribution corresponding
            to a given component.
        """

    # Methods used by self.fit()

    def _initialize_sufficient_statistics(self):
        """Initializes sufficient statistics required for M-step.

        The method is *pure*, meaning that it doesn't change the state of
        the instance.  For extensibility computed statistics are stored
        in a dictionary.

        Returns
        -------
        nobs : int
            Number of samples in the data.

        start : array, shape (n_components, )
            An array where the i-th element corresponds to the posterior
            probability of the first sample being generated by the i-th
            state.

        trans : array, shape (n_components, n_components)
            An array where the (i, j)-th element corresponds to the
            posterior probability of transitioning between the i-th to j-th
            states.
        """
        stats = {'nobs': 0,
                 'start': np.zeros(self.n_components),
                 'trans': np.zeros((self.n_components, self.n_components))}
        return stats

    def _accumulate_sufficient_statistics(self, stats, X, framelogprob,
                                          posteriors, fwdlattice, bwdlattice):
        """Updates sufficient statistics from a given sample.

        Parameters
        ----------
        stats : dict
            Sufficient statistics as returned by
            :meth:`~base._BaseHMM._initialize_sufficient_statistics`.

        X : array, shape (n_samples, n_features)
            Sample sequence.

        framelogprob : array, shape (n_samples, n_components)
            Log-probabilities of each sample under each of the model states.

        posteriors : array, shape (n_samples, n_components)
            Posterior probabilities of each sample being generated by each
            of the model states.

        fwdlattice, bwdlattice : array, shape (n_samples, n_components)
            Log-forward and log-backward probabilities.
        """
        stats['nobs'] += 1
        if 's' in self.params:
            stats['start'] += posteriors[0]
        if 't' in self.params:
            n_samples, n_components = framelogprob.shape
            # when the sample is of length 1, it contains no transitions
            # so there is no reason to update our trans. matrix estimate
            if n_samples <= 1:
                return

            log_xi_sum = np.full((n_components, n_components), -np.inf)
            _hmmc._compute_log_xi_sum(n_samples, n_components, fwdlattice,
                                      log_mask_zero(self.transmat_),
                                      bwdlattice, framelogprob,
                                      log_xi_sum)
            with np.errstate(under="ignore"):
                stats['trans'] += np.exp(log_xi_sum)

    def _do_mstep(self, stats, posteriors):
        """Performs the M-step of EM algorithm.

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
        if 's' in self.params:
            startprob_ = np.maximum(self.startprob_prior - 1 + stats['start'],
                                    0)
            self.startprob_ = np.where(self.startprob_ == 0, 0, startprob_)
            normalize(self.startprob_)
        if 't' in self.params:
            transmat_ = np.maximum(self.transmat_prior - 1 + stats['trans'], 0)
            self.transmat_ = np.where(self.transmat_ == 0, 0, transmat_)
            normalize(self.transmat_, axis=1)
        if not self.a == None:
            # self._grad_ascent(self._transition_posterior(posteriors))
            log_P_tij = log_mask_zero(np.zeros((posteriors.shape[0],self.n_components,self.n_components)))
            _hmmc._transition_posterior(log_P_tij.shape[0], self.n_components, log_P_tij, log_mask_zero(posteriors))
            if self.grad_method == 'newtons_linesearch':
                self._newtons_linesearch(np.exp(log_P_tij[1:]))
            elif self.grad_method == 'newtons':
                self._newtons(np.exp(log_P_tij[1:]))
            elif self.grad_method == 'grad_ascent':
                self._grad_ascent(np.exp(log_P_tij[1:]))
            elif self.grad_method == 'grad_ascent_linesearch':
                self._grad_ascent_linesearch(np.exp(log_P_tij[1:]))
            elif self.grad_method == 'scipy':
                self._scipy(np.exp(log_P_tij[1:]))
            else:
                print (self.grad_method, ' is not a defined method')

    def _transition_posterior_external(self, log_P_tij, posteriors):
        _hmmc._transition_posterior(log_P_tij.shape[0], self.n_components, log_P_tij, log_mask_zero(posteriors))

    def U(self, t):
        return 1-self.sigma(t)

    def dU(self, t):
        return -self.sigma(t)*(1-self.sigma(t))

    # NEW VERSION: x = t/a - b
    def sigma(self, t):
        # print('new sigma')
        t = t/24/60/120
        return (1+np.exp(-(t-self.b)/self.a))**-1

    def _grad_ascent(self, P_tij):
        print('transition posteriors:', P_tij.shape)
        print('max: ', np.max(P_tij))
        print('sumcheck: ', np.sum(P_tij[1,:,:]), np.sum(P_tij[-1,:,:]) )
        if np.array(self.grad_lr).size == 1:
            self.grad_lr = [self.grad_lr, self.grad_lr]
        t = np.arange(P_tij.shape[0]) + self.t0
        for iter in range(self.grad_iter):
            mu = self._mu(t) * P_tij
            db = -mu.sum()/self.a *self.grad_lr[1]
            da = -(((t/24/60/120-self.b)[:,None,None]/self.a**2)*mu).sum()*self.grad_lr[0]

            print('mu: ', self._mu(t)[10].sum())
            print('gradients: ', da, db)
            if (np.abs(da) <= self.grad_conv * np.abs(self.a)) and (np.abs(db) <= self.grad_conv * np.abs(self.b)):
                print('converged: ', iter)
                print('a: ', self.a)
                print('b: ', self.b)
                return;
            self.a = self.a + da
            self.b = self.b + db
            print('a: ', self.a)
            print('b: ', self.b)
        print(self.grad_iter)

    def _grad_ascent_linesearch(self, P_tij):
        t = np.arange(P_tij.shape[0])
        L = lambda : (P_tij*np.log(self.transmat_ + np.transpose(self.A1[:,:,None]*self.U(t),[2,0,1]))).sum()
        # grad_ascent update w/ linesearch
        L_prev = L()
        for iter in range(self.grad_iter):
            # define direction
            mu = self._mu(t) * P_tij
            db = -mu.sum()/self.a
            da = -(((t/24/60/120-self.b)[:,None,None]/self.a**2)*mu).sum()
            delta = np.array([da,db])
            delta =delta/np.linalg.norm(delta, ord=2)
            print('grad: ', da, db)
            # determine step size
            prev_a = self.a
            prev_b = self.b
            updated = False
            for s in self.grad_lr:
                self.a = self.a + s*delta[0]
                self.b = self.b + s*delta[1]
                L_new = L()
                # check if sufficient improvement, if so quit
                #note: log likelihood (L) will always be <0
                #therefore WANT L_new/L_prev < 1
                if L_new/L_prev < self.grad_conv:
                    L_prev = L_new
                    updated = True
                    break;
                else:
                    self.a = prev_a
                    self.b = prev_b

            # check convergence
            # if (np.abs(delta[0]/self.a) < self.grad_conv) and (np.abs(delta[1]/self.b) < self.grad_conv): #TODO: put in convergence criteria
            #     break
            if not updated:
                self.a = prev_a
                self.b = prev_b
                print('not updated')
                break
            self.track_params.append([self.a,self.b, delta[0], delta[1], L_new])
            print('a: ',self.a)
            print('b: ', self.b)


    def _newtons(self,P_tij):
        for i in range(self.grad_iter):
            print(i)
            J, g = self._jacobian_grad(P_tij)
            delta =  - np.matmul(np.linalg.inv(J), g)
            print('Jac: ', J)
            print('grad: ', g)
            print('Diff: ', delta)
            if (np.abs(delta[0]/self.a) < self.grad_conv) and (np.abs(delta[1]/self.b) < self.grad_conv): #TODO: put in convergence criteria
                break
            self.a = self.a + self.grad_lr[0] * delta[0]
            self.b = self.b + self.grad_lr[1] * delta[1]
            self.track_params.append([self.a,self.b, delta[0], delta[1]])
            print('a: ',self.a)
            print('b: ', self.b)
        return

    def _newtons_linesearch(self,P_tij):
        t = np.arange(P_tij.shape[0])
        L = lambda : (P_tij*np.log(self.transmat_ + np.transpose(self.A1[:,:,None]*self.U(t),[2,0,1]))).sum()
        #newton update w/ linesearch
        L_prev = L()
        for i in range(self.grad_iter):
            # get step direction
            print(i)
            J, g = self._jacobian_grad(P_tij)
            delta =  - np.matmul(np.linalg.inv(J), g)
            if np.linalg.norm(delta, ord=2)>1: #sets a maximum to the step size
                delta =delta/np.linalg.norm(delta, ord=2)
            print('Jac: ', J)
            print('grad: ', g)
            print('Diff: ', delta)
            # determine step size
            prev_a = self.a
            prev_b = self.b
            updated = False
            for s in self.grad_lr:
                self.a = self.a + s*delta[0]
                self.b = self.b + s*delta[1]
                L_new = L()
                # check if sufficient improvement, if so quit
                #note: log likelihood (L) will always be <0
                #therefore WANT L_new/L_prev < 1
                print('L',L_new,L_prev)
                if L_new/L_prev < self.grad_conv:
                    L_prev = L_new
                    updated = True
                    break;
                else:
                    self.a = prev_a
                    self.b = prev_b

            # check convergence
            # if (np.abs(delta[0]/self.a) < self.grad_conv) and (np.abs(delta[1]/self.b) < self.grad_conv): #TODO: put in convergence criteria
            #     break
            if not updated:
                self.a = prev_a
                self.b = prev_b
                print('not updated')
                break
            self.track_params.append([self.a,self.b, delta[0], delta[1], L_new])
            print('a: ',self.a)
            print('b: ', self.b)

        return

    def _scipy(self, P_tij):
        t = np.arange(P_tij.shape[0])
        def nL(x):
            a = x[0]
            b = x[1]
            u_loc = lambda t: 1-(1+np.exp(-(t/120/60/24-b)/a))**-1
            t = np.arange(P_tij.shape[0])
            return -(P_tij*np.log(self.transmat_ + np.transpose(self.A1[:,:,None]*u_loc(t),[2,0,1]))).sum()
        def jac(x):
            a = x[0]
            b = x[1]
            sigma_loc = lambda t: 1-(1+np.exp(-(t/120/60/24-b)/a))**-1
            u_loc = lambda t: 1-sigma_loc(t)
            du_loc = lambda t: -sigma_loc(t)*(1-sigma_loc(t))
            mu_loc =lambda t: np.transpose(self.A1[:,:,None]*u_loc(t),[2,0,1])/self.transmat_ + np.transpose(self.A1[:,:,None]*u_loc(t),[2,0,1])
            t = np.arange(P_tij.shape[0])
            mu = mu_loc(t) * P_tij
            dxda = -((t/24/60/120-b)[:,None,None]/a**2)
            dxdb = -1/a
            db = mu*dxdb
            da = mu*dxda
            return -np.array([np.sum(da), np.sum(db)])
        def hess(x):
            a = x[0]
            b = x[1]
            sigma_loc = lambda t: 1-(1+np.exp(-(t/120/60/24-b)/a))**-1
            u_loc = lambda t: 1-sigma_loc(t)
            du_loc = lambda t: -sigma_loc(t)*(1-sigma_loc(t))
            mu_loc =lambda t: np.transpose(self.A1[:,:,None]*u_loc(t),[2,0,1])/self.transmat_ + np.transpose(self.A1[:,:,None]*u_loc(t),[2,0,1])
            t = np.arange(P_tij.shape[0])
            mu = mu_loc(t) * P_tij
            dxda = -((t/24/60/120-b)[:,None,None]/a**2)
            dxdb = -1/a
            db = -mu*dxdb
            da = -mu*dxda
            dL2_dada = (da*(-mu_loc(t)*dxda+(2*u_loc(t)-1)[:,None,None]*dxda-2/a)).sum()
            dL2_dbdb = (db*(-mu_loc(t)*dxdb+(2*u_loc(t)-1)[:,None,None]*dxdb)).sum()
            dL2_dadb = (db*(-mu_loc(t)*dxda+(2*u_loc(t)-1)[:,None,None]*dxda+dxdb)).sum()
            dL2_dbda = dL2_dadb#(db*(-self._mu(t)*dxda+(2*self.U(t)-1)[:,None,None]*dxda+dxdb)
            return np.array([[dL2_dada, dL2_dadb],[dL2_dbda, dL2_dbdb]])

        #result = scipy.optimize.minimize(nL,[self.a,self.b],method='Newton-CG', jac=jac, hess=hess, tol=self.grad_conv,)
        result = scipy.optimize.minimize(nL,[self.a,self.b])
        self.a = result.x[0]
        self.b = result.x[1]
        self.track_params.append([self.a,self.b,result.fun])
        self._grad_ascent_linesearch(P_tij)



    def _jacobian_grad(self,P_tij):
        t = np.arange(P_tij.shape[0])
        mu = self._mu(t) * P_tij
        dxda = -((t/24/60/120-self.b)[:,None,None]/self.a**2)
        dxdb = -1/self.a
        db = mu*dxdb
        da = mu*dxda

        G = np.array([np.sum(da), np.sum(db)])

        dL2_dada = (da*(-self._mu(t)*dxda+(2*self.U(t)-1)[:,None,None]*dxda-2/self.a)).sum()
        dL2_dbdb = (db*(-self._mu(t)*dxdb+(2*self.U(t)-1)[:,None,None]*dxdb)).sum()
        dL2_dadb = (db*(-self._mu(t)*dxda+(2*self.U(t)-1)[:,None,None]*dxda+dxdb)).sum()
        dL2_dbda = dL2_dadb#(db*(-self._mu(t)*dxda+(2*self.U(t)-1)[:,None,None]*dxda+dxdb)
        J = np.array([[dL2_dada, dL2_dadb],[dL2_dbda, dL2_dbdb]])
        return J, G

    # # OLD VERSION: x = a*(t-b)
    # def sigma(self, t):
    #     return (1+np.exp(-self.a*(t-self.b)))**-1
    #
    # def _grad_ascent(self, P_tij):
    #     print('transition posteriors:', P_tij.shape)
    #     print('max: ', np.max(P_tij))
    #     print('sumcheck: ', np.sum(P_tij[1,:,:]), np.sum(P_tij[-1,:,:]) )
    #     t = np.arange(P_tij.shape[0]) + self.t0
    #     for iter in range(self.grad_iter):
    #         mu = self._mu(t) * P_tij
    #         db = -mu.sum()#*self.a*-1
    #         da = ((t[:,None,None]-self.b)*mu).sum()
    #
    #         print('mu: ', self._mu(t)[10].sum())
    #         print('gradients: ', da, db)
    #         if (self.grad_lr * np.abs(da) <= self.grad_conv * np.abs(self.a)) and (self.grad_lr * np.abs(db) <= self.grad_conv * np.abs(self.b)):
    #             print('converged: ', iter)
    #             return;
    #         #self.a = self.a + self.grad_lr * da
    #         self.b = self.b + self.grad_lr * db
    #     print(self.grad_iter)


    def _transition_posterior(self, posteriors):
        print('posterior shape: ',posteriors.shape)
        print('max: ', np.max(posteriors))
        print('sumcheck: ', np.sum(posteriors[1,:]), np.sum(posteriors[-1,:]) )
        P_tij = np.zeros((posteriors.shape[0],self.n_components,self.n_components))
        for t in range(1, P_tij.shape[0]):
            for i in range(self.n_components):
                for j in range(self.n_components):
                    P_tij[t,i,j] = posteriors[t-1,i]*posteriors[t,j]
        return P_tij

    def _mu(self,t):
        num = np.transpose(self.A1[:,:,None]*self.dU(t),[2,0,1])
        denom = self.transmat_ + np.transpose(self.A1[:,:,None]*self.U(t),[2,0,1])
        # print('mu num: ', num[10,:,:].sum())
        # print('mu denom: ', denom[10].sum())
        return num/denom

    def _set_transmat(self, transmat_):
        self.transmat_ = (transmat_)

    def _set_startprob(self,startprob):
        self.startprob_ = startprob
