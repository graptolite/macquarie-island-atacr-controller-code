#!/usr/bin/env python3

'''
To be run after 00_combine_days.py, making sure that SDS_parent is set to the SDS output of that script. Parameters are hardcoded and should be changed within this script where desired.

More involved data preprocessing designed to broadly mimic that of ATaCR. Parallelisable.
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

from obspy import *
import os
import sys
import multiprocessing

# Highly simplistic way of parsing arguments. argparser is not used here to avoid overcomplicating the code.
try:
    ncores = sys.argv[1]
except:
    raise IndexError("Usage: python preprocess_all.py <ncores>")

##########################
# DATA PROCESSING PARAMS #
##########################
# Bandpass params suitable for teleseismic arrivals.
freqmin = 0.03
freqmax = 2
# Whether to apply a prior bandpass of 0.03 to the waveforms. ***THIS IS HIGHLY SPECIFIC TO THIS STUDY***.
## Note: a freqmin of 0.03 is applied as a highpass before demeaning (due to instrument-specific properties), detrending (linear) and then bandpassing. Disable this by setting the following to false
instrument_specific_prehighpass = True
# Resampling frequency for the data. This is applied to avoid using up too much disk space (limited on the machines used by the author).
resample_freq = 4 # Hz
# Instrument-specific gains. Direction important, value not (in this study where waveform alignments are the concern, not the amplitude). These gains are just to match the orientation of the Z and H channels.
gains = {"Z":-1,
         "H":300,
         "1":1,
         "2":1,
         }

if freqmax * 2 > resample_freq:
    raise ValueError("Maximum signal frequencies exceed the Nyquist frequency under the requested sampling rate")

########################
# FILE LOCATION PARAMS #
########################
# Location of SDS directory, where the `SDS_parent`'s position within the SDS structure (https://docs.obspy.org/packages/autogen/obspy.clients.filesystem.sds.html) is:
# <SDS_parent>/
#             |- <year>
#             |- <year>
# And an example path location is <SDS_parent>/2020/3F/MRO01/HDH.D/3F.MRO01..HDH.D.2020.366
SDS_parent = "SDS"
# Where to write the processed data in a format suitable for ATaCR.
out_path = "DATA/"
# Extension for output files.
out_format = "mseed"

# Name for the "location" part of the SDS filename to put when saving the processed data into `out_path`.
# Value not particularly important.
loc = "--"

# Ensure the folder for outputs is present.
if not os.path.exists(out_path):
    os.mkdir(out_path)

# Wrapper function for purely for parallelisation.
# NOTE: this function uses GLOBAL variables defined above.
def process_sta(sta_f):
    '''
    sta_f | <str> | folder containing channel folders for one station.
    '''
    # Identify station name from station folder path.
    sta = os.path.basename(sta_f)
    # Identify network from folder path.
    network = os.path.basename(os.path.dirname(sta_f))
    # Identify year.
    year = os.path.basename(os.path.dirname(os.path.dirname(sta_f)))
    print(sta_f,year,network,sta)
    # Iterate through available channels for the station.
    # Where channel includes the type term (i.e. D)
    for channel in os.listdir(sta_f):
        # Construct path to the channel folder.
        channel_f = os.path.join(sta_f,channel)
        # Iterate through the miniseeds (or relevant format) in the channel folder.
        for mseed in sorted(os.listdir(channel_f)):
            mseed_f = os.path.join(channel_f,mseed)
            st = read(mseed_f)
            if instrument_specific_prehighpass:
                # Initial highpass to use as base data. THIS IS HIGHLY SPECIFIC TO THIS STUDY. Zerophase is important.
                st.filter("highpass",freq=0.03,corners=2,zerophase=True)
            tr = st[0]
            # Detrend (demean and linear)
            tr.detrend("demean")
            tr.detrend("linear")
            # Apply bandpass filter. Zerophase is important.
            tr.filter("bandpass",freqmin=freqmin,freqmax=freqmax,corners=2,zerophase=True)
            # Taper.
            tr.taper(max_percentage=0.02,type="cosine")
            # Apply necessary gain for the relevant channel id (last character of channel).
            tr.data = tr.data * gains[channel.split(".")[0][-1]]
            # Resample.
            tr.resample(resample_freq)
            day = mseed.split(".")[-1]
            # Write output.
            outdir = os.path.join(out_path,".".join([network,sta]))
            if not os.path.exists(outdir):
                os.mkdir(outdir)
            out = os.path.join(outdir,".".join([year,day,"",channel.split(".")[0],out_format]))
            tr.write(out)
    return


# Iterate through file items (testing for year folder) in the SDS folder structure.
for year in os.listdir(SDS_parent):
    # Check that the file item is a folder (in which case treat as a year folder).
    if os.path.isdir(os.path.join(SDS_parent,year)):
        # Construct year folder path.
        year_f = os.path.join(SDS_parent,year)
        # Iterate through network folders in the year folder.
        for network in os.listdir(year_f):
            network_f = os.path.join(year_f,network)
            stas = os.listdir(network_f)
            # Identify station folders.
            sta_fs = [os.path.join(network_f,sta) for sta in stas if "MRO" in sta]
            # Initialise a multiprocessing Pool instance for parallelisation. Not the best location to be initiating as reinit will occur for each year (and network) folder, but as there are few year folders (and only one network folder), this is kept as is.
            p = multiprocessing.Pool(int(ncores))
            p.map(process_sta,sta_fs)
