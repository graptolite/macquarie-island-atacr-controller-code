#!/usr/bin/env python3

'''
To be run after downloading (e.g. from AUSPASS) or obtaining mseed data into the structure:
  <parent dir>/<year>/<3 figure julday>/<year and 3 figure julday>_<time>_<sta>_<channel>.mseed
e.g. <parent dir>/2020/366/2020366_000000_MRO27_HDH.mseed
where parent dir is hardcoded to be "AUSPASS-MSEEDS".

Comment out or modify the file of inv = read_inventory("AUSPASS_dataless.xml") as necessary.

Writes to SDS format.
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

# Hardcoded values specific to this study to avoid needing to repeatedly provide the same variables.
out_dir = "USABLE"
years = ["2020","2021"]
freq_resample = 20 # To prevent processed waveforms from taking too much storage.
parent_dir = "AUSPASS-MSEEDS"
inv = read_inventory("AUSPASS_dataless.xml")
# Highly simplistic way of parsing arguments. argparser is not used here to avoid overcomplicating the code.
try:
    channel = sys.argv[1]
    ncores = int(sys.argv[2])
except:
    raise IndexError("Usage: python 01_combine_days.py <last two (important!) characters of channel> <number of cores to use>")

def process_day(day_dir_in,channel,sds_dest="SDS",net="3F",dtype="D",loc="",inv=None,freq_resample=20):
    ''' Join and process all waveform segments in one day (reading from the data structure format specified at the top of this script). Wrapped into a function for easier parallelisation.

    day_dir_in    | <str>                                              | path to the directory containing the waveforms of one day.
    channel       | <str>                                              | last (two) character(s) of the channel of interest. For this study, the last two characters were suitable (instead of just the last one).
    sds_dest      | <str>                                              | where to save the processed files in SDS structure.
    net           | <str>                                              | network code used for naming the output file.
    dtype         | <str>                                              | SDS datatype used for naming the output file (not particularly important here).
    loc           | <str>                                              | SDS location used for naming the output file (not particularly important here).
    inv           | None or <obspy.core.inventory.inventory.Inventory> | inventory to use to remove instrument response. Set to None for no response removal.
    freq_resample | <float>                                            | frequency to resample waveforms to.
    '''
    print(day_dir_in)
    # Identify year and date from the input folder structure.
    day = os.path.basename(day_dir_in)
    year = os.path.basename(os.path.dirname(day_dir_in))
    # Identify all mseeds with the requested channel identification in the day folder. The same station may be represented by multiple files (recording segments) under the AUSPASS data download.
    mseeds = [f for f in os.listdir(day_dir_in) if f.endswith(".mseed") and ((channel+".mseed") in f)]
    # List to store processed stations to avoid repeatedly processing the same station.
    stas_considered = list()
    # Iterate through the mseeds.
    for mseed in sorted(mseeds):
        print(mseed)
        # Identify the station name from the mseed.
        info = mseed.split("_")
        sta = info[2]
        # Check that the station hasn't been processed already.
        if sta not in stas_considered:
            # Identify the output paths and make sure the relevant output folders exist.
            dst_folder = os.path.join(sds_dest,year,net,sta,channel+"."+dtype)
            if not os.path.exists(dst_folder):
                os.makedirs(dst_folder,exist_ok=True)
            f_out = ".".join([net,sta,loc,channel,dtype,year,day])
            # Read all recording segments (of which there may just be one) for the active station within the day folder.
            print("*%s*%s.m*" % (sta,channel))
            st = read(os.path.join(day_dir_in,"*%s*%s.m*" % (sta,channel)))
            if inv is not None:
                # Remove instrument response if possible/requested.
                ## The pre_filt is included for consistency with ATaCR's data download script's description for the default value of pre_filt.
                st.remove_response(inventory=inv,pre_filt=[0.001, 0.005, 45., 50.])
            # Normalise the waveform.
            st.normalize()
            # Linear detrend.
            st.detrend("linear")
            # Interpolate missing data.
            st.merge(method=1,fill_value="latest")
            tr = st[0]
            # Align the start date of trace to the start of the day (midnight).
            t0 = tr.stats.starttime
            t_midnight = UTCDateTime(t0.year,t0.month,t0.day)
            if t0 != t_midnight:
                first_value = tr.data[0]
                tr = tr.trim(t_midnight,tr.stats.endtime,pad=True,fill_value=first_value)
            # Resample if requested.
            if freq_resample:
                tr.resample(freq_resample)
            # Save processed waveform.
            tr.write(os.path.join(dst_folder,f_out),format="MSEED")
            # Declare station as processed.
            stas_considered.append(sta)
    return

def execute(day_dir_in):
    ''' Helper function to allow multiprocessing parallelisation to proceed whilst taking in the global variables stated near the top of this script.
    '''
    process_day(day_dir_in,channel,inv=inv,freq_resample=freq_resample)
    return

if __name__=="__main__":
    # Iterate through the requested years (folders).
    for y in years:
        year_dir_in = os.path.join(parent_dir,y)
        if os.path.exists(year_dir_in):
            # Identify all day folders within the year folder.
            days = sorted(os.listdir(year_dir_in))
            day_dirs_in = [os.path.join(year_dir_in,day) for day in days]
            # Initialise a multiprocessing Pool instance for parallelisation. Not the best location to be initiating as reinit will occur for each year folder, but as there are few year folders, this is kept as is.
            p = multiprocessing.Pool(ncores)
            p.map(execute,day_dirs_in)
