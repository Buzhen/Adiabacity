import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import pandas as pd

from matplotlib.animation import FFMpegWriter
plt.rcParams["animation.ffmpeg_path"] = 'c:\\ffmpeg\\ffmpeg-2023-11-27-git-0ea9e26636-full_build\\bin\\ffmpeg.exe'
plt.rcParams.update({'text.usetex' : True})

def RK4step(f, fdot, dt, c, invdx2, bDebug=False):

    k1 = np.zeros((f.shape[0]))
    k2 = np.zeros((f.shape[0]))
    k3 = np.zeros((f.shape[0]))
    k4 = np.zeros((f.shape[0]))
    fdot1 = np.zeros((f.shape[0]))
    fdot2 = np.zeros((f.shape[0]))
    fdot3 = np.zeros((f.shape[0]))
    fdot_next = np.zeros((f.shape[0]))
    f1 = np.zeros((f.shape[0]))
    f2 = np.zeros((f.shape[0]))
    f3 = np.zeros((f.shape[0]))
    f_next = np.zeros((f.shape[0]))

    k1[1:-1] = (c**2) * (f[2:] - 2*f[1:-1] + f[:-2])*invdx2
    fdot1[1:-1] = fdot[1:-1] + 0.5*k1[1:-1]*dt
    f1[1:-1] = f[1:-1] + 0.25*(fdot[1:-1] + fdot1[1:-1])*dt

    k2[1:-1] = (c**2) * (f1[2:] - 2*f1[1:-1] + f1[:-2])*invdx2
    fdot2[1:-1] = fdot[1:-1] + 0.5 * k2[1:-1]*dt
    f2[1:-1] = f[1:-1] + 0.25 * (fdot[1:-1] + fdot2[1:-1])*dt

    k3[1:-1] = (c**2) * (f2[2:] - 2*f2[1:-1] + f2[:-2])*invdx2
    fdot3[1:-1] = fdot[1:-1] + k3[1:-1]*dt
    f3[1:-1] = f[1:-1] + 0.5*(fdot[1:-1] + fdot3[1:-1])*dt

    k4[1:-1] = (c**2) * (f3[2:] - 2*f3[1:-1] + f3[:-2])*invdx2
    fdot_next[1:-1] = fdot[1:-1] + dt*(k1[1:-1] + 2*k2[1:-1] + 2*k3[1:-1] + k4[1:-1])/6
    f_next[1:-1] = f[1:-1] + dt*(fdot1[1:-1] + 2*fdot2[1:-1] + 2*fdot3[1:-1] + fdot_next[1:-1])/6

    if bDebug:
        fdot_dbg = np.zeros((f.shape[1]))
        f_dbg = np.zeros((f.shape[1]))
        for xI in range(1, len(f)-1):
            k1[xI] = (c**2) * (f[xI+1] - 2*f[xI] + f[xI-1])*invdx2
            fdot1[xI] = fdot[xI] + 0.5*k1[xI]*dt
            f1[xI] = f[xI] + 0.25*(fdot[xI] + fdot1[xI])*dt
        for xI in range(1, len(f)-1):
            k2[xI] = (c**2) * (f1[xI+1] - 2*f1[xI] + f1[xI-1])*invdx2
            fdot2[xI] = fdot[xI] + 0.5*k2[xI]*dt
            f2[xI] = f[xI] + 0.25*(fdot[xI] + fdot2[xI])*dt
        for xI in range(1, len(f)):
            k3[xI] = (c**2) * (f2[xI+1] - 2*f2[xI] + f2[xI-1])*invdx2
            fdot3[xI] = fdot[xI] + k3[xI]*dt
            f3[xI] = f[xI] + 0.5*(fdot[xI] + fdot3[xI])*dt
        for xI in range(1, len(f)):
            k4[xI] = (c**2) * (f3[xI+1] - 2 * f3[xI] + f3[xI-1])*invdx2
            fdot_dbg[xI] = fdot[xI] + dt*(k1[xI] + 2*k2[xI] + 2*k3[xI] + k4[xI])/6
            f_dbg[xI] = f[xI] + dt * (fdot1[xI] + 2 * fdot2[xI] + 2 * fdot3[xI] + fdot_dbg[xI]) / 6
        assert np.all(f_dbg == f)

    return f_next, fdot_next


def sim_string(dx=0.025, dt=0.001, method='RK4', bSaveFrames=True, bMakeVid=True, bDebug=False, rate=0.01):

    assert method in ['RK4', 'FinDiff']
    Li = 1
    DL = 1
    Lf = Li + DL

    c = 1
    invdx2 = 1/(dx**2)

    # rate = dL/Ldt, omega_n = c*pi*n. omega>>dL/Ldt adiabatic, omega<<dL/Ldt non-adiabatic
    dL = Li*dt*rate
    mu = np.power((c*dt/dx), 2.0)

    xs = np.arange(0, Lf, dx)
    Nx, Nt = len(xs), int(DL/dL)

    psi = np.zeros((2, Nx), dtype=np.float64)

    # Boundary conditions on the Left and Right side boundaries. dada - to do, add to code explicitly
    BCL = 0
    BCR = 0

    # Initial conditions:
    # 1. psi(0,0<x<Li) = sin(pi*x/Li), psi(0,x>Li) = 0
    psi[0, :] = np.sin(np.pi*xs/Li)
    xBoundaryInd = np.argmin(np.abs(Li - xs))
    psi[0, xBoundaryInd:] = 0
    # 2. dpsi/dt(0,x) = 0
    psidotIC = np.zeros(Nx, dtype=np.float64)

    # By Taylor series of order 2: a) psi(dt,x) = psi(0,x) + psi_t(0,x)dt + 0.5psi_tt(0,x)dt^2
    # By the wave equation: b) psi_tt(t,x) = (c/dx)^2*(psi(t,x+dx) - 2psi(t,x) + psi(t,x-dx))
    # Substituting (b) into (a): psi(dt,x) = (1-mu)psi(0,x) + 0.5mu(psi(0,x+dx)+psi(0,x-dx)) + psi_t(0,x)dt
    psidot = np.zeros((2, Nx), dtype=np.float64)
    psidot[0, :] = psidotIC
    psi[1, 1:xBoundaryInd] = (1 - mu)*psi[0, 1:xBoundaryInd] + 0.5*mu*(psi[0, 2:xBoundaryInd+1] + psi[0, :xBoundaryInd-1])\
                             + psidot[0, 1:xBoundaryInd]*dt

    if bSaveFrames:
        # Display resolution like dx
        res = dx
        xIndsFrames = np.arange(0, Nx, res/dx, dtype=np.uint)
        xsFrames = np.arange(0, Lf, res)
        assert ((res/dx) % 1) == 0 #np.abs(((res/dx) % 1)) < 1e-10

        DT = Nt*dt
        # Run at 20fps
        fps = 20
        frame_dt = 1/fps
        assert np.abs(((frame_dt/dt) % 1)) < 1e-10
        tIndsFrames = np.arange(0, Nt, frame_dt/dt, dtype=np.uint)
        tIndsFrames = np.append(tIndsFrames, [Nt-1])
        Nframes = len(tIndsFrames)
        ts = np.arange(0, DT, dt)
        tsFrames = ts[tIndsFrames]
        tsFrames_dbg = np.arange(0, DT, frame_dt); tsFrames_dbg = np.round(tsFrames*fps, 0); tsFrames_dbg *= frame_dt
        assert np.all(np.abs(tsFrames[:-1]-tsFrames_dbg[:-1]) < 1e-10)

        psiFrames = np.zeros((Nframes, xsFrames.shape[0]), dtype=np.float64)
        psidotFrames = np.zeros((Nframes, xsFrames.shape[0]), dtype=np.float64)
        LsFrames = np.zeros((Nframes), dtype=np.float64)

    if bDebug:
        psi1_dbg = np.zeros((xBoundaryInd))
        for xI in range(1, xBoundaryInd):
            psi1_dbg[xI] = (1 - mu) * psi[0, xI] + 0.5 * mu * (psi[0, xI + 1] + psi[0, xI - 1]) + psidot[0, xI] * dt
        assert np.all(psi1_dbg==psi[1, :xBoundaryInd])

    for tI in range(0, Nt):
        if tI == 0 and method == 'FinDiff':
            psi_next = np.zeros(Nx, dtype=np.float64)
            continue
        t = tI * dt
        L = Li + tI*dL

        if bSaveFrames and tI in tIndsFrames:
            frameInd = np.where(tIndsFrames == tI)
            assert np.abs(tsFrames[frameInd]-t) < 1e-10
            psiFrames[frameInd, :] = psi[0, xIndsFrames]
            psidotFrames[frameInd, :] = psidot[0, xIndsFrames]
            LsFrames[frameInd] = L

        xBoundaryInd = np.argmin(np.abs(L-xs))

        if (method == 'RK4'):
            psi[0, :xBoundaryInd+1], psidot[0, :xBoundaryInd+1] = RK4step(psi[0, :xBoundaryInd+1],
                                                                          psidot[0, :xBoundaryInd+1], dt, c, invdx2)

        elif (method == 'FinDiff'):
            psi_next[0] = 2.0 * psi[1, 0] - psi[0, 0]
            psi_next[1:xBoundaryInd] = 2*(1-mu)*psi[1, 1:xBoundaryInd] + \
                                       mu*(psi[1, 2:xBoundaryInd+1]+psi[1, :xBoundaryInd-1]) - psi[0, 1:xBoundaryInd]
            psidot[0, :] = (psi_next - psi[1, :])/dt
            psi[0, :], psi[1, :] = psi[1, :], psi_next
        else:
            assert False

    # dE = 0.5(T*(dpsi/dx)^2 + rhp*(dpsi/dt)^2) = 0.5rho((c*dpsi/dx)^2 + (dpsi/dt)^2)
    psitag = (psiFrames[:, 1:]-psiFrames[:, :-1])/res
    Ep = 0.5*(np.sum((c*psitag)**2*dx, axis=1))
    Ek = 0.5*(np.sum(psidotFrames[:, 1:-1]**2*res, axis=1))
    Energy = Ep + Ek
    deltaE = Energy[0]-Energy[-1]

    N = 8
    As = np.zeros((tsFrames.shape[0], N))
    As_bar_labels = []
    for k in range(1, N+1):
        As_sint = 2*np.sum(psiFrames*np.sin(np.pi*k*xsFrames/LsFrames[:,np.newaxis])*res, axis=1)/LsFrames
        As_cost = 2*np.sum(psidotFrames*np.sin(np.pi*k*xsFrames/LsFrames[:, np.newaxis])*res, axis=1)/(np.pi*k*c)
        As[:, k-1] = np.sqrt(np.power(As_sint, 2) + np.power(As_cost, 2))
        As_bar_labels.append(f'$A_{k}$')

    fig, (ax1, ax2) = plt.subplots(1, 2)

    def animate(frame):
        ax1.clear()
        ax2.clear()
        time = np.round(frame*frame_dt, 5)
        L = LsFrames[frame]
        fig.suptitle(f't={time}sec, L={L}m')
        xBoundaryInd = np.argmin(np.abs(L-xsFrames))
        ax1.set_xlim(0, Lf)
        ax1.set_ylim(-1.2, 1.2)
        ax1.set_xlabel('x[m]')
        ax1.set_ylabel('$\psi(x)[m]$')
        ax1.plot(xsFrames[xBoundaryInd], 0, 'or')
        ax1.plot(xsFrames, psiFrames[frame, :])
        ax2.bar(As_bar_labels, As[frame, :])
        ax2.set_ylim(-0.2, 1.2)

    if bMakeVid and (Nframes*frame_dt>0.5):
        ani = animation.FuncAnimation(fig, animate, frames=Nframes, interval=frame_dt*1000)
        writevideo = animation.FFMpegWriter(fps=fps)
        ani.save(f'c:/Users/eitan/Desktop/String Simulation/rate={rate}.mp4', writer=writevideo)

    return As[-1, :]


rates = [100, 50, 25, 10, 5, 2.5, 1, 0.5, 0.25, 0.1, 0.05, 0.025, 0.01]
As_f = np.zeros((len(rates), 8), dtype=np.float64)

for rateI in range(len(rates)):
    print(rates[rateI])
    As_f[rateI, :] = sim_string(bDebug=False, dx=0.0001, dt=0.05/5000, method='RK4', rate=rates[rateI])
"""
dxs = 0.1/np.array([1, 2, 5, 10, 20, 50])#, 100, 200, 500, 1000])
dts = 0.05/np.array([5, 10, 20, 50, 100, 200])#, 500, 1000, 2000, 5000, 10000])
deltaEs = np.zeros((len(dxs), len(dts)))

for dxI in range(3):#len(dxs)):
    print(f'dx={dxs[dxI]}')
    if 1:
        aaa = np.argwhere(dts<dxs[dxI])
        deltaEs[dxI, aaa[0, 0]] = sim_string(bDebug=False, dx=0.01, dt=0.05/50, method='RK4', rate=1)
        break
    for dtI in range(len(dts)):
        if dts[dtI]>=dxs[dxI]:
            continue
        print(f'dt={dts[dtI]}')
        deltaEs[dxI, dtI] = sim_string(bDebug=False, dx=dxs[dxI], dt=dts[dtI], method='RK4')
    inds = dts<dxs[dxI]
    plt.loglog(dts[inds], deltaEs[dxI, inds], '-o', label=f'dx={dxs[dxI]}')
"""
plt.semilogx(rates, As_f[:, 0], '-o', label=f'dx={dxs[dxI]}')
plt.xlabel('$rate=\\tau/\omega$')
plt.ylabel('$A_1$')
plt.savefig(f'c:/Users/eitan/Desktop/String Simulation/A1.jpg')

plt.semilogx(rates, As_f[:, 1], '-o', label=f'dx={dxs[dxI]}')
plt.xlabel('$rate=\\tau/\omega$')
plt.ylabel('$A_2$')
plt.savefig(f'c:/Users/eitan/Desktop/String Simulation/A2.jpg')

plt.semilogx(rates, As_f[:, 2], '-o', label=f'dx={dxs[dxI]}')
plt.xlabel('$rate=\\tau/\omega$')
plt.ylabel('$A_3$')
plt.savefig(f'c:/Users/eitan/Desktop/String Simulation/A3.jpg')

plt.semilogx(rates, As_f[:, 3], '-o', label=f'dx={dxs[dxI]}')
plt.xlabel('$rate=\\tau/\omega$')
plt.ylabel('$A_4$')
plt.savefig(f'c:/Users/eitan/Desktop/String Simulation/A4.jpg')

plt.semilogx(rates, As_f[:, 4], '-o', label=f'dx={dxs[dxI]}')
plt.xlabel('$rate=\\tau/\omega$')
plt.ylabel('$A_5$')
plt.savefig(f'c:/Users/eitan/Desktop/String Simulation/A5.jpg')

plt.semilogx(rates, As_f[:, 5], '-o', label=f'dx={dxs[dxI]}')
plt.xlabel('$rate=\\tau/\omega$')
plt.ylabel('$A_6$')
plt.savefig(f'c:/Users/eitan/Desktop/String Simulation/A6.jpg')

plt.semilogx(rates, As_f[:, 6], '-o', label=f'dx={dxs[dxI]}')
plt.xlabel('$rate=\\tau/\omega$')
plt.ylabel('$A_7$')
plt.savefig(f'c:/Users/eitan/Desktop/String Simulation/A7.jpg')

plt.semilogx(rates, As_f[:, 7], '-o', label=f'dx={dxs[dxI]}')
plt.xlabel('$rate=\\tau/\omega$')
plt.ylabel('$A_8$')
plt.savefig(f'c:/Users/eitan/Desktop/String Simulation/A8.jpg')

print(1)