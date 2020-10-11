# %% imports
import scipy
import scipy.io
import scipy.stats

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import dynamicmodels
import measurementmodels
import ekf
import imm
import pda

# %% custom imports
from typing import List

from gaussparams import GaussParams
from mixturedata import MixtureParameters
import estimationstatistics as estats

# %% plot config check and style setup


# to see your plot config
print(f"matplotlib backend: {matplotlib.get_backend()}")
print(f"matplotlib config file: {matplotlib.matplotlib_fname()}")
print(f"matplotlib config dir: {matplotlib.get_configdir()}")
plt.close("all")

# try to set separate window ploting
if "inline" in matplotlib.get_backend():
    print("Plotting is set to inline at the moment:", end=" ")

    if "ipykernel" in matplotlib.get_backend():
        print("backend is ipykernel (IPython?)")
        print("Trying to set backend to separate window:", end=" ")
        import IPython

        IPython.get_ipython().run_line_magic("matplotlib", "")
    else:
        print("unknown inline backend")

print("continuing with this plotting backend", end="\n\n\n")


# set styles
try:
    # installed with "pip install SciencePLots" (https://github.com/garrettj403/SciencePlots.git)
    # gives quite nice plots
    plt_styles = ["science", "grid", "bright", "no-latex"]
    plt.style.use(plt_styles)
    print(f"pyplot using style set {plt_styles}")
except Exception as e:
    print(e)
    print("setting grid and only grid and legend manually")
    plt.rcParams.update(
        {
            # setgrid
            "axes.grid": True,
            "grid.linestyle": ":",
            "grid.color": "k",
            "grid.alpha": 0.5,
            "grid.linewidth": 0.5,
            # Legend
            "legend.frameon": True,
            "legend.framealpha": 1.0,
            "legend.fancybox": True,
            "legend.numpoints": 1,
        }
    )


# %% load data and plot
filename_to_load = "data_joyride.mat"
loaded_data = scipy.io.loadmat(filename_to_load)
K = loaded_data["K"].item()
Ts = loaded_data["Ts"].squeeze() # different from run imm PDA
Ts = np.append(Ts, 2.5)
Xgt = loaded_data["Xgt"].T
Z = [zk.T for zk in loaded_data["Z"].ravel()]

# plot measurements close to the trajectory
fig1, ax1 = plt.subplots(num=1, clear=True)

Z_plot_data = np.empty((0, 2), dtype=float)
plot_measurement_distance = 45
for Zk, xgtk in zip(Z, Xgt):
    to_plot = np.linalg.norm(Zk - xgtk[None:2], axis=1) <= plot_measurement_distance
    Z_plot_data = np.append(Z_plot_data, Zk[to_plot], axis=0)

ax1.scatter(*Z_plot_data.T, color="C1")
ax1.plot(*Xgt.T[:2], color="C0", linewidth=1.5)
ax1.set_title("True trajectory and the nearby measurements")
plt.show(block=False)

# =============================================================================
# # %% play measurement movie. Remember that you can cross out the window
# play_movie = True
# play_slice = slice(0, K)
# if play_movie:
#     if "inline" in matplotlib.get_backend():
#         print("the movie might not play with inline plots")
#     fig2, ax2 = plt.subplots(num=2, clear=True)
#     sh = ax2.scatter(np.nan, np.nan)
#     th = ax2.set_title(f"measurements at step 0")
#     mins = np.vstack(Z).min(axis=0)
#     maxes = np.vstack(Z).max(axis=0)
#     ax2.axis([mins[0], maxes[0], mins[1], maxes[1]])
#     plotpause = 0.1
#     # sets a pause in between time steps if it goes to fast
#     for k, Zk in enumerate(Z[play_slice]):
#         sh.set_offsets(Zk)
#         th.set_text(f"measurements at step {k}")
#         fig2.canvas.draw_idle()
#         plt.show(block=False)
#         plt.pause(plotpause)
# =============================================================================

# %% setup and track

run_three_models = False  # Set to false to run IMM with two models (CV and CT)

# sensor
sigma_z = 12 # 10
clutter_intensity = 2 / (4000*4000) # 1e-2
PD = 0.65 # 0.8
gate_size = 5

# dynamic models
sigma_a_CV = 4
sigma_a_CT = 4
sigma_omega = 0.3
sigma_a_CV_high = 10


# markov chain
PI11 = 0.6
PI22 = 0.6
PI33 = 0.6

p10 = 0.3  # initvalue for mode probabilities
p20 = 0.35
p30 = 0.35 # Change when using three models, p10, p20 and p30 must sum to 1

PI = np.array([[PI11, (1-PI11)], [(1-PI22), PI22]])
if run_three_models:
    PI = np.array([[PI11, (1-PI11)/2, (1-PI11)/2], [(1-PI22)/2, PI22, (1-PI22)/2], [(1-PI33)/2, (1-PI33)/2, PI33]])
assert np.allclose(np.sum(PI, axis=1), 1), "rows of PI must sum to 1"

mean_init = np.array([7100, 3620, 0, 0, 0])
cov_init = np.diag([40, 40, 10, 10, 0.1]) ** 2 # np.diag([1000, 1000, 30, 30, 0.1]) ** 2  # THIS WILL NOT BE GOOD
mode_probabilities_init = np.array([p10, p20])
mode_states_init = GaussParams(mean_init, cov_init)
init_imm_state = MixtureParameters(mode_probabilities_init, [mode_states_init] * 2)
if run_three_models:
    mode_probabilities_init = np.array([p10, p20, p30])
    init_imm_state = MixtureParameters(mode_probabilities_init, [mode_states_init] * 3)

assert np.allclose(
    np.sum(mode_probabilities_init), 1
), "initial mode probabilities must sum to 1"

# make model
measurement_model = measurementmodels.CartesianPosition(sigma_z, state_dim=5)
dynamic_models: List[dynamicmodels.DynamicModel] = []
dynamic_models.append(dynamicmodels.WhitenoiseAccelleration(sigma_a_CV, n=5))
dynamic_models.append(dynamicmodels.ConstantTurnrate(sigma_a_CT, sigma_omega))
ekf_filters = []
ekf_filters.append(ekf.EKF(dynamic_models[0], measurement_model))
ekf_filters.append(ekf.EKF(dynamic_models[1], measurement_model))
if run_three_models:
    dynamic_models.append(dynamicmodels.WhitenoiseAccelleration(sigma_a_CV_high, n=5))
    ekf_filters.append(ekf.EKF(dynamic_models[2], measurement_model))
imm_filter = imm.IMM(ekf_filters, PI)

tracker = pda.PDA(imm_filter, clutter_intensity, PD, gate_size)

# init_imm_pda_state = tracker.init_filter_state(init__immstate)


NEES = np.zeros(K)
NEESpos = np.zeros(K)
NEESvel = np.zeros(K)

tracker_update = init_imm_state
tracker_update_list = []
tracker_predict_list = []
tracker_estimate_list = []
# estimate
for k, (Zk, x_true_k, Tsk) in enumerate(zip(Z, Xgt, Ts)):
    tracker_predict = tracker.predict(tracker_update, Tsk)
    tracker_update = tracker.update(Zk, tracker_predict)

    # You can look at the prediction estimate as well
    tracker_estimate = tracker.estimate(tracker_update)

    NEES[k] = estats.NEES(*tracker_estimate, x_true_k, idxs=np.arange(4))
    NEESpos[k] = estats.NEES(*tracker_estimate, x_true_k, idxs=np.arange(2))
    NEESvel[k] = estats.NEES(*tracker_estimate, x_true_k, idxs=np.arange(2, 4))

    tracker_predict_list.append(tracker_predict)
    tracker_update_list.append(tracker_update)
    tracker_estimate_list.append(tracker_estimate)


x_hat = np.array([est.mean for est in tracker_estimate_list])
prob_hat = np.array([upd.weights for upd in tracker_update_list])

# calculate a performance metrics
poserr = np.linalg.norm(x_hat[:, :2] - Xgt[:, :2], axis=0)
velerr = np.linalg.norm(x_hat[:, 2:4] - Xgt[:, 2:4], axis=0)
posRMSE = np.sqrt(
    np.mean(poserr ** 2)
)  # not true RMSE (which is over monte carlo simulations)
velRMSE = np.sqrt(np.mean(velerr ** 2))
# not true RMSE (which is over monte carlo simulations)
peak_pos_deviation = poserr.max()
peak_vel_deviation = velerr.max()


# consistency
confprob = 0.9
CI2 = np.array(scipy.stats.chi2.interval(confprob, 2))
CI4 = np.array(scipy.stats.chi2.interval(confprob, 4))

confprob = confprob
CI2K = np.array(scipy.stats.chi2.interval(confprob, 2 * K)) / K
CI4K = np.array(scipy.stats.chi2.interval(confprob, 4 * K)) / K
ANEESpos = np.mean(NEESpos)
ANEESvel = np.mean(NEESvel)
ANEES = np.mean(NEES)

# %% plots

 # trajectory
fig3, axs3 = plt.subplots(1, 2, num=3, clear=True)
axs3[0].plot(*x_hat.T[:2], label=r"$\hat x$")
axs3[0].plot(*Xgt.T[:2], label="$x$")
axs3[0].set_title(
    f"RMSE(pos, vel) = ({posRMSE:.3f}, {velRMSE:.3f})\npeak_dev(pos, vel) = ({peak_pos_deviation:.3f}, {peak_vel_deviation:.3f})"
)
axs3[0].axis("equal")
# probabilities
axs3[1].plot(np.arange(K) * Ts[0], prob_hat.T[0], label="CV")
axs3[1].plot(np.arange(K) * Ts[0], prob_hat.T[1], label="CT")
if run_three_models:
    axs3[1].plot(np.arange(K) * Ts[0], prob_hat.T[2], label="CV-high")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
axs3[1].set_ylim([0, 1])
axs3[1].set_ylabel("mode probability")
axs3[1].set_xlabel("time")

plt.show()
 












