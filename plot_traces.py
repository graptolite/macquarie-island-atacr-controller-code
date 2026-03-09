#!/usr/bin/env python3

''' Plot collections of traces of different denoise types cut to the same relative time about their predicted arrival times.

Specific to the Macquarie OBS network.
'''

'''
ATaCR controller code | wrapper scripts for ATaCR as implemented by OBSTools
    Copyright (C) 2026 Yingbo Li

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
'''

from identify_seismem import SeisMem
import os
from obspy import *
import matplotlib.pyplot as plt
import pandas as pd
from obspy.taup import TauPyModel
model = TauPyModel(model="ak135")
import sys

# Load events and station dataframes.
ev_df = pd.read_csv("catalogue.csv")
sta_df = pd.read_csv("3F.csv",index_col=0)

# atacr_pre and atacr_post must have the same values as pre and post in 12_process_event.py
atacr_pre = 100 # s
atacr_post = 200 # s

# Relative start and end time to cut the plotted waveforms to.
pre = 20 # s
post = 40 # s
# Minimum and maximum frequencies to filter the waveforms by before plotting.
freqmin = 0.3 # Hz
freqmax = 2 # Hz

# Denoise types to analyse, mapping to their axis index in the combined plot.
trace_type_idx = {"og":0,
                  "day.ZP-21":1,
                  "sta.ZP-21":2,
                  "day.ZP-H":3,}

# Folder storing denoised traces.
traces_dir = "denoised-data"

# Aliased/derived times.
d_arrtime = pre # s
maxlength = pre+post # s

# Add timestamps to events dataframe, and use as index.
ev_df["TIMESTAMPS"] = ev_df.apply(lambda row : str(UTCDateTime(row["DATE"] + "T" + row["TIME"])).replace("Z",""),axis=1)
ev_df.index = ev_df["TIMESTAMPS"]

# Sensor mappings to identify sensors from in the trace plots.
sensors = {"MRO01":"GUR",
           "MRO03":"CAS",
           "MRO06":"CAS",
           "MRO07":"CAS",
           "MRO10":"GUR",
           "MRO11":"CAS",
           "MRO12":"CAS",
           "MRO13":"CAS",
           "MRO16":"GUR",
           "MRO18":"GUR",
           "MRO20":"CAS",
           "MRO22":"CAS",
           "MRO23":"CAS",
           "MRO24":"CAS",
           "MRO27":"GUR",
           }

# Create directory to save figures to.
fig_dir = os.path.join(traces_dir,"figures")
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir,exist_ok=True)
# Identify SeisMem jsons (denoised events) to plot.
fs = [f for f in os.listdir(traces_dir) if f.endswith(".json")]
# Identify earthquake names.
evs = sorted(set(["-".join(f.split("-")[:3]).replace(".json","") for f in fs]))
for ev in evs:
    print(ev)
    # Load event json.
    SM = SeisMem(os.path.join(traces_dir,ev+".json"))
    # Identify event data (location).
    ev_data = ev_df.loc[".".join(ev.split(".")[:-1])]
    # Initialise figure.
    ncols = len(trace_type_idx)
    fig,axs = plt.subplots(1,ncols,figsize=(ncols*5,6),sharey=True)
    # Get list of station names.
    stas = sorted(set([k.split("/")[1] for k in SM.storage.keys()]))
    # Map station names to trace index (based on alphabetical station ordering).
    sta_idx_dict = {s:i for i,s in enumerate(stas)}
    try:
        # Iterate through all streams in SeisMem collection.
        for k,st in SM.storage.items():
            tr = st[0]
            # Filter trace as requested.
            tr.filter("bandpass",freqmin=freqmin,freqmax=freqmax,zerophase=True)
            # Trim trace as requested.
            tr.trim(tr.stats["starttime"]+(atacr_pre-pre),tr.stats["endtime"]-(atacr_post-post))
            # Identify trace properties.
            trace_props = k.split("/")
            # Identify the denoise type of the trace.
            trace_type = trace_props[0]
            if trace_type in trace_type_idx:
                # Identify axes index to plot the denoise type on.
                type_idx = trace_type_idx[trace_type]
                # Identify station and corresponding trace index.
                sta = trace_props[1]
                sta_idx = j = sta_idx_dict[sta]
                # Normalize trimmed trace.
                tr.normalize()
                # Add text with the station name (and character to distinguish sensor type) to left of undenoised traces plot.
                if type_idx == 0:
                    axs[0].text(-30,j*2,sta + " " + sensors[sta][0],verticalalignment="center")
                x = tr.times()
                y = tr.data
                # Plot trace.
                axs[type_idx].plot(x,y+j*2,c="k")
                axs[type_idx].fill_between(x,j*2,y+j*2,where=y > 0,color="blue",interpolate=True)
                axs[type_idx].fill_between(x,y+j*2,j*2,where=y < 0,color="red",interpolate=True)
                # Add predicted relative arrival time line.
                axs[type_idx].axvline(d_arrtime,zorder=1e2,linewidth=0.1)
                # Add title for denoise type.
                axs[type_idx].set_title(trace_type)
        phase = ev.split(".")[-1]
        # Iterate through the stations and plot all converted P arrivals.
        for sta in stas:
            sta_data = sta_df.loc[sta]
            sta_idx = j = sta_idx_dict[sta]
            # Identify all arrivals.
            maybe_arrs = model.get_travel_times_geo(source_depth_in_km=ev_data["DEPTH"],
                                           source_latitude_in_deg=ev_data["LAT"],
                                           source_longitude_in_deg=ev_data["LON"],
                                           receiver_latitude_in_deg=sta_data["lat"],
                                           receiver_longitude_in_deg=sta_data["lon"])
            # Identify reference arrival, which arrives 20 s after the start of the trace.
            arr_ev = [arr.time for arr in maybe_arrs if arr.phase.name==phase][0]
            for arr in maybe_arrs:
                if arr.phase.name.endswith("P"):
                    # Compute relative phase arrival time.
                    t = arr.time-arr_ev+d_arrtime
                    # Check if relative arrival time is within range of the trimmed trace.
                    if t > 0 and t < maxlength:
                        # Plot predicted relative phase arrival time on all trace collection axes.
                        for i in range(ncols):
                            axs[i].plot([t,t],[j*2-0.5,j*2+0.5],c="m",zorder=-1e5)
                            axs[i].text(t+0.5,j*2-0.6,arr.phase.name,c="m",zorder=-1e5)
        # Save figure with relatively low resolution.
        plt.savefig(os.path.join(fig_dir,ev+".png"),dpi=72)
        plt.close("all")
    except (KeyError,ValueError):
        print("pass")
        pass
