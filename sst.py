import netCDF4
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

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

RES_X = 64 #LONS
RES_Y = 32 #LATS
DEPTH = 300 #m
ITERS = 24*1000
DT = 3600
S_0 = 1371.0
AXIS = 0*np.pi/180
SB = 5.67*10**-8 #W m^-2 K^-4

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

print ew_dists

trpz_areas = ns_dist*(ew_dists[1:] + ew_dists[:-1])/2

areas = np.rot90(np.tile(trpz_areas, 64).reshape((RES_Y, RES_X)))
masses = 1000*DEPTH*areas

print areas

#convert lats/lons to meshgrid and radians

lats, lons = np.meshgrid(lats, lons)

lats = lats*np.pi/180
lons = lons*np.pi/180

#initialize sst
ssts = np.zeros((RES_X, RES_Y))+288.

mean_ssts = []
for x in xrange(0, ITERS):
	phi = 0 #year angle
	#calculate radiation
	E_sol_p_sqm = S_0/2 * (np.sin(lats)*np.sin(AXIS*np.sin(phi))+np.cos(lats)*np.cos(AXIS*np.sin(phi))) * DT
	E_sol_p_sqm = np.where(E_sol_p_sqm > 0., E_sol_p_sqm, 0.)
	ssts += E_sol_p_sqm*areas/(masses*4184.)
	
	E_out_p_sqm = SB*ssts**4 * DT
	ssts -= E_out_p_sqm*areas/(masses*4184.)

	print np.mean(ssts)
	mean_ssts.append(np.mean(ssts))

	#plot
	if x % 2400 == 0:
		taslevels = np.arange(-40, 48, 3)

		m = Basemap(projection='cyl', llcrnrlat=-90,urcrnrlat=90,llcrnrlon=0,urcrnrlon=360,resolution='l')

		fig = plt.figure(figsize=(18.6, 10.5))
		ax = fig.add_axes((0,0,1,1))
		ax.set_axis_off()

		#m.drawcoastlines()

		# plot temperature contours
		plt.contourf(lons, lats, ssts-273.15, levels=taslevels, linewidths=1, cmap='bwr', zorder=0)
		con_temp = plt.contour(lons, lats, ssts-273.15, levels=taslevels, linewidths=1, colors='k', zorder=1)
		plt.clabel(con_temp, con_temp.levels, fmt='%d')

		plt.colorbar(fraction=0.025, pad=0.01)
	
		plt.savefig("/home/kalassak/sst/sst_" + str(x/24) + ".png", bbox_inches='tight', pad_inches=0, dpi=100)
		plt.close()

		print "plot for " + str(x/24) + " generated"

#plot diagnostics
fig = plt.figure(figsize=(18.6, 10.5))
ax = fig.add_axes((0,0,1,1))

plt.plot(np.arange(0, 24000, 1), mean_ssts, 'b-')

plt.savefig("/home/kalassak/sst/diag_mean_sst.png", bbox_inches='tight', pad_inches=0, dpi=100)
plt.close()

#todo:
#
#- S_0/4?
#- slab atmosphere
#- advection scheme
#- wind stress (from plasim)
#- write monthly means to .sra format

