"""
This is a simulation of quiescent plasma with periodic boundaries
It was modified from the provided lwfa_script.py tutorial

Help
----
All the structures implemented in FBPIC are internally documented.
Enter "print(fbpic_object.__doc__)" to have access to this documentation,
where fbpic_object is any of the objects or function of FBPIC.
"""

# -------
# Imports
# -------
import numpy as np
import math as math
from scipy.constants import c, e, m_e, m_p, Boltzmann, pi, mu_0, epsilon_0
from scipy.special import ellipk
# Import the relevant structures in FBPIC
from fbpic.main import Simulation
from fbpic.lpa_utils.external_fields import ExternalField
from fbpic.lpa_utils.laser import add_laser_pulse
from fbpic.lpa_utils.laser.laser_profiles import GaussianLaser
from fbpic.openpmd_diag import FieldDiagnostic, ParticleDiagnostic, ParticleChargeDensityDiagnostic, \
     set_periodic_checkpoint, restart_from_checkpoint

# ----------
# Parameters
# ----------

# Whether to use the GPU
use_cuda = False

# Order of the stencil for z derivatives in the Maxwell solver.
# Use -1 for infinite order, i.e. for exact dispersion relation in
# all direction (advised for single-GPU/single-CPU simulation).
# Use a positive number (and multiple of 2) for a finite-order stencil
# (required for multi-GPU/multi-CPU with MPI). A large `n_order` leads
# to more overhead in MPI communications, but also to a more accurate
# dispersion relation for electromagnetic waves. (Typically,
# `n_order = 32` is a good trade-off.)
# See https://arxiv.org/abs/1611.05712 for more information.
n_order = -1

# The simulation box
Nz = 400         # Number of gridpoints along z
zmax = 800.0     # Right end of the simulation box (meters)
zmin = 0         # Left end of the simulation box (meters)
Nr = 200         # Number of gridpoints along r
rmax = 400.0     # Length of the box along r (meters)
Nm = 3           # Number of modes used

n_guard = 96      # number of guard cells
n_r_inj = 30      # radial size of radial injection region (number of cells)

# The simulation timestep
dt = 4.0e-8      # Timestep (seconds)
print("Timestep: ", dt)

# The particles
p_zmin = -2.0    # Position of the beginning of the plasma (meters)
p_zmax = 12000.0  # Position of the end of the plasma (meters)
p_rmax = 402.0   # Maximal radial position of the plasma (meters)
n_e = 8e6        # Density (electrons.meters^-3)
n_Nei = 8e6      # Hydrogen ions per cubic meter
n_N = 1e6        # Hydrogen neutrals per cubic meter
p_nz = 2         # Number of particles per cell along z
p_nr = 2         # Number of particles per cell along r
p_nt = 4         # Number of particles per cell along theta
T_e = 15000000     # kelvin
T_n = 7500000      # kelvin (same for ions and neutrals)

ndiff = 1e6      # density added to radial injection region
#ndiff = n_Nei*sqrt(2*pi*m_p/(T_e*Boltzman))*(((Nr+n_r_inj)*(rmax/Nr))**2 - rmax**2)/(4*rmax)
# ^This is what it would be if n_Nei corisponds to ambient density at simulation edge

# The moving window
v_window = 450000         # Speed of the window

# The diagnostics and the checkpoints/restarts
diag_period = 50         # Period of the diagnostics in number of timesteps
save_checkpoints = False # Whether to write checkpoint files
checkpoint_period = 100  # Period for writing the checkpoints
use_restart = False      # Whether to restart from a previous checkpoint
track_electrons = False  # Whether to track and write particle ids

# Temperature calculations (these definition come from Remi Lehe on the FB-PIC slack
uthxyz_e = np.sqrt(Boltzmann * T_e / m_e) / c
uthxyz_i = np.sqrt(Boltzmann * T_n / (m_p)) / c
uthxyz_n = uthxyz_i
uth_3sigma = 3*c*max([uthxyz_e,uthxyz_i,uthxyz_n])
print("3 sigma thermal velocity 1 axis: ", uth_3sigma)
nz_diffusion = np.ceil(uth_3sigma * dt * Nz / (zmax - zmin))
nr_diffusion = np.ceil(uth_3sigma * dt * Nr / rmax)
print("Diffusion grid steps in z, r: ", nz_diffusion, nr_diffusion)

# The density profile
#ramp_start = 30.e-6
#ramp_length = 40.e-6

#def dens_func( z, r ) :
#    """Returns relative density at position z and r"""
#    # Allocate relative density
#    n = np.ones_like(z)
#    n = np.where(z < ramp_start + ramp_length, n/10 , n)
#
#    return(n)

# sample applied field function from https://fbpic.github.io/api_reference/lpa_utilities/external_fields.html
def field_func_z( F, x, y, z, t , amplitude, length_scale ):
    # Magnetic field of a dipole position Z0, moment m0
    m0 = amplitude  # A-m^2
    z0 = 400.0 + 450000*t   # z coordinate of dipole
    rdipole = math.sqrt(x**2 + y**2 + (z-z0)**2)
    rdotm = (z-z0)*m0
    Badd = (mu_0/(4*pi))*((3*(z-z0)*rdotm/(rdipole**5))-(m0/(rdipole**3)))
    return(F + Badd)

def field_func_x( F, x, y, z, t , amplitude, length_scale ):
    # Magnetic field of a dipole position Z0, moment m0
    m0 = amplitude  # A-m^2
    z0 = 400.0 + 450000*t   # z coordinate of dipole
    rdipole = math.sqrt(x**2 + y**2 + (z-z0)**2)
    rdotm = (z-z0)*m0
    Badd = (mu_0/(4*pi))*((3*x*rdotm/(rdipole**5)))
    return(F + Badd)

def field_func_y( F, x, y, z, t , amplitude, length_scale ):
    # Magnetic field of a dipole position Z0, moment m0
    m0 = amplitude  # A-m^2
    z0 = 400.0 + 450000*t   # z coordinate of dipole
    rdipole = math.sqrt(x**2 + y**2 + (z-z0)**2)
    rdotm = (z-z0)*m0
    Badd = (mu_0/(4*pi))*((3*y*rdotm/(rdipole**5)))
    return(F + Badd)


# The interaction length of the simulation (meters)
L_interact = 20.e-1 # increase to simulate longer distance!
# Interaction time (seconds) (to calculate number of PIC iterations)
#T_interact = ( L_interact + (zmax-zmin) ) / (2 * v_window)
# (i.e. the time it takes for the moving window to slide across the plasma)

# ---------------------------
# Carrying out the simulation
# ---------------------------

# NB: The code below is only executed when running the script,
# (`python lwfa_script.py`), but not when importing it (`import lwfa_script`).
if __name__ == '__main__':

    # Initialize the simulation object
    sim = Simulation( Nz, zmax, Nr, rmax, Nm, dt, zmin=zmin,
        n_order=n_order, use_cuda=use_cuda, n_guard=n_guard, n_r_inj=n_r_inj, ndiff=ndiff,  exchange_period=1,
        boundaries={'z':'open', 'r':'reflective', 'rdiff':'yes', 'zdiff':'yes'}, verbose_level=2)
        # 'r': 'open' can also be used, but is more computationally expensive

    # Add the neon ions pre ionized to level 1
    atoms_Nei = sim.add_new_species( q=e, m=1.*m_p, n=n_Nei,
        dens_func=None, p_nz=p_nz, p_nr=p_nr, p_nt=p_nt, p_zmin=p_zmin, p_zmax=p_zmax, p_rmax=p_rmax,
        uz_th = uthxyz_i, ux_th = uthxyz_i, uy_th = uthxyz_i, continuous_injection=True)
    # Add the neutral neon atoms
    atoms_N = sim.add_new_species( q=0, m=1.*m_p, n=n_N,
        dens_func=None, p_nz=p_nz, p_nr=p_nr, p_nt=p_nt, p_zmin=p_zmin, p_zmax=p_zmax, p_rmax=p_rmax,
        uz_th = uthxyz_n, ux_th = uthxyz_n, uy_th = uthxyz_n, continuous_injection=True)
    # Create the plasma electrons
    elec = sim.add_new_species( q=-e, m=m_e, n=n_e,
        dens_func=None, p_nz=p_nz, p_nr=p_nr, p_nt=p_nt, p_zmin=p_zmin, p_zmax=p_zmax, p_rmax=p_rmax,
        uz_th = uthxyz_e, ux_th = uthxyz_e, uy_th = uthxyz_e, continuous_injection=True)

    # Apply "externally generated" aka "external" fields
    sim.external_fields = [ExternalField(field_func_z, 'Bz', 1e18, 0.5)]
    sim.external_fields = [ExternalField(field_func_x, 'Bx', 1e18, 0.5)]
    sim.external_fields = [ExternalField(field_func_y, 'By', 1e18, 0.5)]

    # Load initial fields
    # Create a Gaussian laser profile
    #laser_profile = GaussianLaser(a0, w0, tau, z0)
    # Add the laser to the fields of the simulation
    #add_laser_pulse( sim, laser_profile)

    if use_restart is False:
        # Track electrons if required (species 0 correspond to the electrons)
        if track_electrons:
            elec.track( sim.comm )
    else:
        # Load the fields and particles from the latest checkpoint file
        restart_from_checkpoint( sim )

    # Configure the moving window
    sim.set_moving_window( v=v_window )

    # Add diagnostics
    sim.diags = [ FieldDiagnostic( diag_period, sim.fld, comm=sim.comm ),
                  ParticleDiagnostic( diag_period, {"electrons" : elec, "ions" : atoms_Nei, "neutrals": atoms_N}, comm=sim.comm ),
                  ParticleChargeDensityDiagnostic( diag_period, sim, {"electrons" : elec, "ions" : atoms_Nei, "neutrals": atoms_N} )]

    # Add checkpoints
    if save_checkpoints:
        set_periodic_checkpoint( sim, checkpoint_period )

    # Number of iterations to perform
    N_step = int(201)

    ### Run the simulation
    sim.step( N_step )
    print('')
