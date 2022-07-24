import netCDF4
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

#conda activate sst

# dqdt requires a list of the time derivatives for q, stored 
# in order from present to the past
def ab_blend(dqdt,order):
    if order==1:
        return dqdt[0]
    elif order==2:
        return 1.5*dqdt[0]-.5*dqdt[1]
    elif order==3:
        return (23*dqdt[0]-16*dqdt[1]+5*dqdt[2])/12.
    else:
        print("order", order ," not supported ")

def advect(u,b,dx,order):
# Assumes u=0 at end points (a solid boundary). So no need to calculate
# dbdx[0] and dbdx[-1]
    dbdx=np.zeros(len(b))
    
    if order == 2: # 2nd-order centered scheme
        dbdx[1:-1] =  (-b[0:-2] + b[2:])/(2*dx)

    elif order == 1: # 1st order upwind
        dbdx[1:-1] = np.where( u[1:-1]>0.,
                              -b[0:-2] + b[1:-1] 
                              ,        - b[1:-1] + b[2:] 
                              )/dx
        
    elif order == 3: # 3rd order upwind. But 1st order one point from boundaries  
        dbdx[1] = np.where( u[1]>0.,
                              -b[0] + b[1] 
                              ,     - b[1] + b[2] 
                              )/dx
        dbdx[-2] = np.where( u[-2]>0.,
                              -b[-3] + b[-2] 
                              ,      - b[-2] + b[-1] 
                              )/dx
        dbdx[2:-2] = np.where( u[2:-2]>0., 
                +   b[:-4]  - 6*b[1:-3] + 3*b[2:-2] + 2*b[3:-1] ,
                            - 2*b[1:-3] - 3*b[2:-2] + 6*b[3:-1] - b[4:] 
                               )/(6*dx)
    else:
        print("no advection scheme for",order)
        
    return -u*dbdx

def calc_dist(lat1,lon1,lat2,lon2):
	R = 6371008.8
	lat1 = np.radians([lat1])
	lon1 = np.radians([lon1])
	lat2 = np.radians([lat2])
	lon2 = np.radians([lon2])

	dlat = lat2 - lat1
	dlon = lon2 - lon1
	a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2) **2
	c = 2 * np.arctan2(a**0.5, (1-a)**0.5)
	d = R * c
	return d

#----------------------------
RES_X = 64 #LONS
RES_Y = 32 #LATS
DEPTH = 10 #m
ITERS = 24*1000
DT = 3600
S_0 = 1371.0
AXIS = 23.5*np.pi/180
SB = 5.67*10**-8 #W m^-2 K^-4
EMISSIVITY = 0.8
R = 6371008.8 #m
DAY = 86400 #s
YEAR = 365*DAY
RHO_0 = 1000. #kg/m^3
f = 10**-4

#equation of state
T_0 = 283. #K
ALPHA = 1.7*10**-4 #K^-1
#----------------------------

lon_bs = np.linspace(0, 360, RES_X+1)
lat_bs = np.linspace(90, -90, RES_Y+1)

lons = (lon_bs[1:] + lon_bs[:-1])/2
lats = (lat_bs[1:] + lat_bs[:-1])/2

#area calcs
ns_dist = calc_dist(lat_bs[0], lon_bs[0], lat_bs[1], lon_bs[0])

ew_dists = []
for lat_b in lat_bs:
	ew_dist = calc_dist(lat_b, lon_bs[0], lat_b, lon_bs[1])
	ew_dists.append(float(ew_dist))

ew_dists = np.array(ew_dists)

print(ew_dists)

trpz_areas = ns_dist*(ew_dists[1:] + ew_dists[:-1])/2

areas = np.rot90(np.tile(trpz_areas, 64).reshape((RES_Y, RES_X)))
masses = RHO_0*DEPTH*areas

print(areas)

#convert lats/lons to meshgrid and radians

lats, lons = np.meshgrid(lats, lons)

lats = lats*np.pi/180
lons = lons*np.pi/180

#initialize sst
ssts = np.zeros((RES_X, RES_Y))+288.
#initial atmospheric temperature
atm_temp = 255. #K
#spring equinox
phi = 0

mean_ssts = []
atm_temps = []
us = 0.*ssts
vs = 0.*ssts
dudt = 0.*us
dvdt = 0.*vs
dTdt = 0.*ssts
dudta = [0.,0.,0.] # items will be arrays of dudt for Adams-Bashforth
dvdta = [0.,0.,0.] # for v...
dTdta = [0.,0.,0.] # for T...
advord = 3
aborder = 3
for x in range(0, ITERS):
	phi += 2*np.pi/YEAR * DT #year angle
	#CALCULATE RADIATION
	#sfc in (insolation)
	E_sol_p_sqm = 0.70*S_0/np.pi * (np.sin(lats)*np.sin(AXIS*np.sin(phi))+np.cos(lats)*np.cos(AXIS*np.sin(phi))) * DT
	E_sol_p_sqm = np.where(E_sol_p_sqm > 0., E_sol_p_sqm, 0.)
	ssts += E_sol_p_sqm*areas/(masses*4184.)
	
	#sfc out (to space and atm)
	E_sfcout_p_sqm = SB*ssts**4 * DT
	ssts -= E_sfcout_p_sqm*areas/(masses*4184.)

	#atmosphere layer
	#simply solve for the temperature of the slab atmosphere
	sfc_temp = np.sum(ssts*areas)/np.sum(areas) #area-weighted mean surface temperature
	atm_temp = 255 #(0.5*sfc_temp**4)**0.25
	
	#sfc in (atmosphere)
	E_atm2sfc_p_sqm = EMISSIVITY*SB*atm_temp**4 * DT
	ssts += E_atm2sfc_p_sqm*areas/(masses*4184.)

	print(sfc_temp)
	mean_ssts.append(sfc_temp)
	atm_temps.append(atm_temp)

	#CALCULATE DENSITY
	rhos = RHO_0*(1-ALPHA*(ssts-T_0))

	#MOMENTUM EQUATIONS
	dudt = f*vs #-1/RHO_0*dpdx
	dvdt = - f*us #-1/RHO_0*dpdy
	
	#CALCULATE ADVECTION
	'''
	dudt += advect(us,us,ew_dists,advord) #nonlinear advection term
	dvdt += advect(vs,vs,ns_dist,advord) #nonlinear advection term

	dTdt += advect(us,ssts,ew_dists,advord)
	dTdt += advect(vs,ssts,ns_dist,advord)

	# Adams-Bashforth time-step:
	abnow = min(x,aborder)
	dudta = [dudt.copy()] + dudta[:-1]
	dvdta = [dvdt.copy()] + dvdta[:-1]
	dTdta = [dTdt.copy()] + dTdta[:-1]
	us += DT*ab_blend(dudta,abnow)
	vs += DT*ab_blend(dvdta,abnow)
	ssts += DT*ab_blend(dTdta,abnow)

	#advect() only acts on one dimension
	#fix ew_dists so advect() only gets one value
	#fix u, v grid sizes
	#make f a proper grid
	'''

	#plot
	if x % 2400 == 0:
		taslevels = np.arange(-40, 48, 3)

		fig = plt.figure(figsize=(18.6, 10.5))
		ax = plt.axes(projection=ccrs.PlateCarree())
		ax.set_axis_off()

		#m.drawcoastlines()

		# plot temperature contours
		plt.contourf(lons, lats, ssts-273.15, levels=taslevels, linewidths=1, cmap='bwr', zorder=0)
		con_temp = plt.contour(lons, lats, ssts-273.15, levels=taslevels, linewidths=1, colors='k', zorder=1)
		plt.clabel(con_temp, con_temp.levels, fmt='%d')

		#plt.colorbar(fraction=0.025, pad=0.01)
	
		plt.savefig("/home/kalassak/sst/sst_" + str(x/24) + ".png", bbox_inches='tight', pad_inches=0, dpi=100)
		plt.close()

		print("plot for " + str(x/24) + " generated")

#plot diagnostics
fig = plt.figure(figsize=(18.6, 10.5))
ax = fig.add_axes((0,0,1,1))

plt.plot(np.arange(0, 24000, 1), mean_ssts, 'b-')
plt.plot(np.arange(0, 24000, 1), atm_temps, 'r-')

plt.savefig("/home/kalassak/sst/diag_mean_sst.png", bbox_inches='tight', pad_inches=0, dpi=100)
plt.close()

#todo:
#
#- S_0/4?
#- slab atmosphere
#- advection scheme
#- wind stress (from plasim)
#- write monthly means to .sra format
#- water should freeze

