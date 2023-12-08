import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation

from matplotlib.animation import FFMpegWriter
plt.rcParams["animation.ffmpeg_path"] = 'c:\\ffmpeg\\ffmpeg-2023-11-27-git-0ea9e26636-full_build\\bin\\ffmpeg.exe'
plt.rcParams.update({'text.usetex' : True})

def main(method='RK4', bMakeVid=False, bDebug=False):

    assert method in ['RK4', 'FinDiff']
    Li = 1
    DL = 1
    Lf = Li + DL

    c = 1
    dt = 0.00001
    dx = 0.025
    invdx2 = 1/(dx**2)

    # rate = dL/Ldt, omega_n = c*pi*n. omega>>dL/Ldt adiabatic, omega<<dL/Ldt non-adiabatic
    rate = 0.01
    dL = Li*dt*rate
    omega_1 = c*np.pi/Li
    mu = np.power((c*dt/dx), 2.0)

    xs = np.arange(0, Lf, dx)
    Nx, Nt = len(xs), int(DL/dL)
    psi = np.zeros((Nt, Nx), dtype=np.float64)
    Lengths = np.arange(Li, Lf, dL)

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

    if bDebug:
        psi_dbg = np.copy(psi)
    # By Taylor series of order 2: psi(dt,x) = psi(0,x) + psi_t(0,x)dt + 0.5psi_tt(0,x)dt^2
    # By the wave equation psi_tt(t,x) = (c/dx)^2*(psi(t,x+dx) - 2psi(t,x) + psi(t,x-dx))
    psidot = np.zeros((Nt, Nx), dtype=np.float64)
    psidot[0, :] = psidotIC
    for xI in range(1, xBoundaryInd):
        psi[1, xI] = (1-mu**2.0)*psi[0, xI] + 0.5*mu**2.0*(psi[0, xI+1] + psi[0, xI-1])
    # psi[1, 1:xBoundaryInd] = (1-mu**2.0)*psi[0, 1:xBoundaryInd] + 0.5*mu**2.0*(psi[0, 2:xBoundaryInd+1] + psi[0, :xBoundaryInd-1])

    Ndisps = 20
    tDisps = np.round(np.linspace(2, Nt-1, Ndisps, endpoint=True))

    for tI in range(0, Nt-1):
        if tI == 0 and (not bRK4):
            continue
        t = (tI+1)*dt
        L = Li#Lengths[tI+1]
        xBoundaryInd = np.argmin(np.abs(L-xs))

        if (method == 'RK4'):
            k1 = np.zeros((psi.shape[1]))
            k2 = np.zeros((psi.shape[1]))
            k3 = np.zeros((psi.shape[1]))
            k4 = np.zeros((psi.shape[1]))
            psidot1 = np.zeros((psi.shape[1]))
            psidot2 = np.zeros((psi.shape[1]))
            psidot3 = np.zeros((psi.shape[1]))
            psi1 = np.zeros((psi.shape[1]))
            psi2 = np.zeros((psi.shape[1]))
            psi3 = np.zeros((psi.shape[1]))

            if(bDebug):
                for xI in range(1, xBoundaryInd):
                    k1[xI] = c**2*(psi[tI, xI+1]-2*psi[tI, xI]+psi[tI, xI-1])*invdx2
                    psidot1[xI] = psidot[tI, xI] + 0.5*k1[xI]*dt
                    psi1[xI] = psi[tI, xI] + 0.25*(psidot[tI, xI] + psidot1[xI])*dt
                for xI in range(1, xBoundaryInd):
                    k2[xI] = c**2*(psi1[xI+1]-2*psi1[xI]+psi1[xI-1])*invdx2
                    psidot2[xI] = psidot[tI, xI] + 0.5*k2[xI]*dt
                    psi2[xI] = psi[tI, xI] + 0.25*(psidot[tI, xI] + psidot2[xI])*dt
                for xI in range(1, xBoundaryInd):
                    k3[xI] = c**2*(psi2[xI+1]-2*psi2[xI]+psi2[xI-1])*invdx2
                    psidot3[xI] = psidot[tI, xI] + k3[xI]*dt
                    psi3[xI] = psi[tI, xI] + 0.5*(psidot[tI, xI] + psidot3[xI])*dt
                for xI in range(1, xBoundaryInd):
                    k4[xI] = c**2*(psi3[xI+1]-2*psi3[xI]+psi3[xI-1])*invdx2
                    psidot[tI+1, xI] = psidot[tI, xI] + dt*(k1[xI] + 2*k2[xI] + 2*k3[xI] + k4[xI])/6
                    psi_dbg[tI+1, xI] = psi[tI, xI] + dt*(psidot1[xI] + 2*psidot2[xI] + 2*psidot3[xI] + psidot[tI+1, xI])/6

            k1[1:xBoundaryInd] = c**2*(psi[tI, 2:xBoundaryInd+1]-2*psi[tI, 1:xBoundaryInd]+psi[tI, :xBoundaryInd-1])*invdx2
            psidot1[1:xBoundaryInd] = psidot[tI, 1:xBoundaryInd] + 0.5*k1[1:xBoundaryInd]*dt
            psi1[1:xBoundaryInd] = psi[tI, 1:xBoundaryInd] + 0.25*(psidot[tI, 1:xBoundaryInd] + psidot1[1:xBoundaryInd])*dt

            k2[1:xBoundaryInd] = c**2*(psi1[2:xBoundaryInd+1]-2*psi1[1:xBoundaryInd]+psi1[:xBoundaryInd-1])*invdx2
            psidot2[1:xBoundaryInd] = psidot[tI, 1:xBoundaryInd] + 0.5*k2[1:xBoundaryInd]*dt
            psi2[1:xBoundaryInd] = psi[tI, 1:xBoundaryInd] + 0.25*(psidot[tI, 1:xBoundaryInd] + psidot2[1:xBoundaryInd])*dt

            k3[1:xBoundaryInd] = c**2*(psi2[2:xBoundaryInd+1]-2*psi2[1:xBoundaryInd]+psi2[:xBoundaryInd-1])*invdx2
            psidot3[1:xBoundaryInd] = psidot[tI, 1:xBoundaryInd] + k3[1:xBoundaryInd]*dt
            psi3[1:xBoundaryInd] = psi[tI, 1:xBoundaryInd] + 0.5*(psidot[tI, 1:xBoundaryInd] + psidot3[1:xBoundaryInd])*dt

            k4[1:xBoundaryInd] = c**2*(psi3[2:xBoundaryInd+1]-2*psi3[1:xBoundaryInd]+psi3[:xBoundaryInd-1])*invdx2
            psidot[tI+1, 1:xBoundaryInd] = psidot[tI, 1:xBoundaryInd] + dt*(k1[1:xBoundaryInd] + 2*k2[1:xBoundaryInd] + 2*k3[1:xBoundaryInd] + k4[1:xBoundaryInd])/6
            psi[tI+1, 1:xBoundaryInd] = psi[tI, 1:xBoundaryInd] + dt*(psidot1[1:xBoundaryInd] + 2*psidot2[1:xBoundaryInd] + 2*psidot3[1:xBoundaryInd] + psidot[tI+1, 1:xBoundaryInd])/6

            if bDebug:
                assert np.all(psi_dbg[tI+1, :]==psi[tI+1, :])

        elif (method == 'FinDiff'):
            psi[tI + 1, 0] = 2.0 * psi[tI, 0] - psi[tI - 1, 0]
            for xI in range(1, xBoundaryInd):
                psi[tI+1, xI] = 2*(1-mu**2)*psi[tI, xI] + mu**2*(psi[tI, xI+1]+psi[tI, xI-1]) - psi[tI-1, xI]
            # psi[tI+1, 1:xBoundaryInd] = 2*(1-mu**2)*psi[tI, 1:xBoundaryInd] + mu**2*(psi[tI, 2:xBoundaryInd+1]+psi[tI, :xBoundaryInd-1]) - psi[tI-1, 1:xBoundaryInd]
        else:
            assert False
        if False and (tI+1) in tDisps:
            plt.plot(xs, psi[tI+1, :])
            plt.plot(xs[xBoundaryInd], 0, 'or')
            plt.title(f"Adiabatic t={np.round(t)}[sec]")
            plt.ylim([-1.1,1.1])
            plt.show(block=True)
            #plt.savefig(f"c:/Users/eitan/Desktop/String Simulation/adiabatic t={int(t)}.jpg")
            plt.close()
            #plt.show(block=True)

    #fig = plt.figure(figsize=(15, 15))


    # dE = 0.5(T*(dpsi/dx)^2 + rhp*(dpsi/dt)^2) = 0.5rho((c*dpsi/dx)^2 + (dpsi/dt)^2)
    psitag = (psi[:, 2:]-psi[:, :-2])/(2*dx)
    Dt = (DL/dL)*dt
    ts = np.arange(0, Dt, dt)
    Ek = 0.5*(np.sum(psidot[:, 1:-1]**2*dx, axis=1))
    Ep = 0.5*(np.sum((c*psitag)**2*dx, axis=1))
    Energy = 0.5*(np.sum((c*psitag)**2*dx, axis=1) + np.sum(psidot[:, 1:-1]**2*dx, axis=1))

    fig = plt.figure()
    l = plt.plot([], [], 'k-')
    plt.xlim(0, Lf)
    plt.ylim(-1, 1)
    index = iter(np.arange(0, Nt, 100))

    def animate(i):
        tI = next(index)
        plt.clf()
        plt.xlabel('x[m]')
        plt.ylabel('$\psi(x)[m]$')
        time = (tI+1)*dt
        L = Lengths[tI+1]
        xBoundaryInd = np.argmin(np.abs(L-xs))
        plt.plot(xs[xBoundaryInd], 0, 'or')
        plt.title(f"Adiabatic t={np.round(t)}[sec]")
        plt.title(f't={time}sec')
        plt.xlim(0, Lf)
        plt.ylim(-1.2, 1.2)
        plt.plot(xs, psi[tI, :])

    if bMakeVid:
        ani = animation.FuncAnimation(fig, animate, interval=200)
        writevideo = animation.FFMpegWriter(fps=15)
        ani.save('c:/Users/eitan/Desktop/String Simulation/adiabatic.mp4', writer=writevideo)
    #plt.show()

    #metadata = dict(title='StringWave', artist='EitanT')
    #writer = FFMpegWriter(fps=15, metadata=metadata)

    #writer.saving(ani, 'c:/Users/eitan/Desktop/String Simulation/adiabatic.mp4', 100)

    print(1)
    """
    thresh = 0.001
    coeffs_i = np.zeros((1, 20))
    coeffs_i[0] = 1
    coeffs = np.copy(coeffs_i)

    for stepI in range(1, Nsteps+1):
        L = Li + stepI*dL
    """

main()