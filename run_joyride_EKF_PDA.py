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

sigma_z = 20
sigma_omega = 0.1
PD = 0.65
clutter_intensity = 4 / (4000*4000)
gate_size = 5

useTurnRateModel = True
    
if useTurnRateModel:
    sigma_a = 4 # works really well with sigma_omega = 0.1
    dynamic_model = dynamicmodels.ConstantTurnrate(sigma_a, sigma_omega)
else: # constant velocity model
    sigma_a = 2
    dynamic_model = dynamicmodels.WhitenoiseAccelleration(sigma_a, n = 5)


measurement_model = measurementmodels.CartesianPosition(sigma_z, state_dim = 5)
ekf_filter = ekf.EKF(dynamic_model, measurement_model)

tracker = pda.PDA(ekf_filter, clutter_intensity, PD, gate_size)

# allocate
NEES = np.zeros(K)
NEESpos = np.zeros(K)
NEESvel = np.zeros(K)

# initialize
x_bar_init = np.array([7100, 3620, 0, 0, 0])

P_bar_init = np.diag([40, 40, 10, 10, 0.1]) ** 2

#init_state = tracker.init_filter_state({"mean": x_bar_init, "cov": P_bar_init})
init_state = GaussParams(x_bar_init, P_bar_init)

tracker_update = init_state
tracker_update_list = []
tracker_predict_list = []
tracker_estimate_list = []
# estimate
for k, (Zk, x_true_k, Tsk) in enumerate(zip(Z, Xgt, Ts)):
    tracker_predict = tracker.predict(tracker_update, Tsk)
    tracker_update = tracker.update(Zk, tracker_predict)
    tracker_estimate = tracker.estimate(tracker_update)
    
    NEES[k] = estats.NEES(*tracker_estimate, x_true_k, idxs=np.arange(4))
    NEESpos[k] = estats.NEES(*tracker_estimate, x_true_k, idxs=np.arange(2))
    NEESvel[k] = estats.NEES(*tracker_estimate, x_true_k, idxs=np.arange(2, 4))

    tracker_predict_list.append(tracker_predict)
    tracker_update_list.append(tracker_update)
    tracker_estimate_list.append(tracker_estimate)


x_hat = np.array([est.mean for est in tracker_estimate_list])
prob_hat = np.array([upd.cov for upd in tracker_update_list])

# calculate a performance metrics
poserr = np.linalg.norm(x_hat[:, :2] - Xgt[:, :2], axis=0)
velerr = np.linalg.norm(x_hat[:, 2:4] - Xgt[:, 2:4], axis=0)
posRMSE = np.sqrt(np.mean(poserr ** 2))  # not true RMSE (which is over monte carlo simulations)
velRMSE = np.sqrt(np.mean(velerr ** 2)) # not true RMSE (which is over monte carlo simulations)
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
fig3, ax3 = plt.subplots(num=3, clear=True)
ax3.plot(*x_hat.T[:2], label=r"$\hat x$")
ax3.plot(*Xgt.T[:2], label="$x$")
ax3.set_title(
    rf"$\sigma_a = {sigma_a:.3f}$, $\sigma_z = {sigma_z:.3f}$, posRMSE = {posRMSE:.2f}, velRMSE = {velRMSE:.2f}"
)

adjTs = np.arange(K) * Ts[0]
avgTs = np.average(Ts)

# NEES
fig4, axs4 = plt.subplots(3, sharex=True, num=4, clear=True)
axs4[0].plot(adjTs, NEESpos)
axs4[0].plot([0, (K - 1) * avgTs], np.repeat(CI2[None], 2, 0), "--r")
axs4[0].set_ylabel("NEES pos")
inCIpos = np.mean((CI2[0] <= NEESpos) * (NEESpos <= CI2[1]))
axs4[0].set_title(f"{inCIpos*100:.1f}% inside {confprob*100:.1f}% CI")

axs4[1].plot(adjTs, NEESvel)
axs4[1].plot([0, (K - 1) * avgTs], np.repeat(CI2[None], 2, 0), "--r")
axs4[1].set_ylabel("NEES vel")
inCIvel = np.mean((CI2[0] <= NEESvel) * (NEESvel <= CI2[1]))
axs4[1].set_title(f"{inCIvel*100:.1f}% inside {confprob*100:.1f}% CI")

axs4[2].plot(adjTs, NEES)
axs4[2].plot([0, (K - 1) * avgTs], np.repeat(CI4[None], 2, 0), "--r")
axs4[2].set_ylabel("NEES")
inCI = np.mean((CI4[0] <= NEES) * (NEES <= CI4[1]))
axs4[2].set_title(f"{inCI*100:.1f}% inside {confprob*100:.1f}% CI")

print(f"ANEESpos = {ANEESpos:.2f} with CI = [{CI2K[0]:.2f}, {CI2K[1]:.2f}]")
print(f"ANEESvel = {ANEESvel:.2f} with CI = [{CI2K[0]:.2f}, {CI2K[1]:.2f}]")
print(f"ANEES = {ANEES:.2f} with CI = [{CI4K[0]:.2f}, {CI4K[1]:.2f}]")

# errors
fig5, axs5 = plt.subplots(2, num=5, clear=True)
axs5[0].plot(adjTs, np.linalg.norm(x_hat[:, :2] - Xgt[:, :2], axis=1))
axs5[0].set_ylabel("position error")

axs5[1].plot(adjTs, np.linalg.norm(x_hat[:, 2:4] - Xgt[:, 2:4], axis=1))
axs5[1].set_ylabel("velocity error")

plt.show()


