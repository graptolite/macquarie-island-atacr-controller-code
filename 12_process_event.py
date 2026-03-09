#!/usr/bin/env python3

'''
To be run after 11_generate_spectra.sh. Many parameters are hardcoded at the start of the script. Check that these are suitable.

Cutting events and executing denoising. Parallelisable with the number of cores as the second argument. Events are cut directly from DATA/ if the arrival doesn't lie too close to a day boundary (to avoid the need to reprocess the data), otherwise it is cut from the SDS data itself and then processed (slower but less effect of the taper) using the same steps as in 10_preprocess_all.py (around lines 87-101).

Not particularly optimised.
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
import pandas as pd
from obspy.taup import TauPyModel
model = TauPyModel(model="ak135")
import sys
from identify_seismem import SeisMem
import shutil
import subprocess
import numpy as np
import multiprocessing

# Time to start processing events from.
start_processing_from = "2020-10-15T00:00:00.000000" # process all

# Highly simplistic way of parsing arguments. argparser is not used here to avoid overcomplicating the code.
try:
    phase = sys.argv[1]
except IndexError:
    print("No phase supplied, defaulting to P")
    phase = "P"
try:
    n_cores = sys.argv[2]
except IndexError:
    n_cores = 1

print("Cores",str(n_cores))

# Event trimming params (relative to predicted arrival time at the station).
pre = 100 #s
post = 200 #s
# Whether to reprocess and overwrite denoised files if they already exist.
overwrite_evs = True
# +/- days to compute "average" station spectra over.
pm_days = 2
# Minimum number of stations to accept denoising for. Set to 4 in the study (to avoid needing to review events recorded across too few events to be meaningful for teleseismic tomography), but 0 here for testing.
min_stas = 0
# Location of stations csv, with station names as the index column (first column) containing at least the columns: lat lon
stas_f = "3F.csv"
# Location of the full earthquakes csv (of which `evs_to_process_f` must be a subset of).
eq_f = "catalogue.csv"
# Location of SDS data.
SDS_parent = "../SDS"
# Directory within which stations are listed.
data_dir = "DATA"
# List of all channels. These are used for exact matching of miniseed files. The ones listed below are applicable to the Macquarie dataset. This must be 4 long with one vertical, one hydrophone, and two horizontal channels.
all_channels = ["HZ","DH","H1","H2"]

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
# Upsampling frequency after denoising for trace collation. Set to None for no upsampling.
upsample_freq = 20 # Hz

# Whether to avoid reprocessing events that don't lie near midnight (i.e. cut directly from DATA, which has already been processed in the same way as used to generate SPECTRA etc.).
avoid_reprocessing = True
# For checking whether the arrival could lie near enough to midnight to warrant cutting from SDS rather than tapered DATA.
# Approximate centre of study area.
clat,clon = -55,158.5
# Critical +/- time relative to midnight before marking as potential in tapered region.
dt_crit = 2500

def exec_denoise(sta):
    ''' Helper function for executing ATaCR denoising on a single station (to help with parallelisation).
    '''
    subprocess.call(["bash",".denoise_event_run.sh",sta])
    return

def check_taper_crossing(ev,ev_data,phase,clat=clat,clon=clon,dt_crit=dt_crit):
    ''' Determine whether a phase arrival may be at risk of being within a taper around midnight

    ev      | <str>       | event id in the form of a UTC timestamp without the Z at the end (Z is assumed).
    ev_data | <pd.Series> | row of data for the event as read from the events dataframe. Must contain entries for "LAT", "LON" and "DEPTH" of the event.
    phase   | <str>       | phase to check the (first) arrival time of. Must be recognisable by TauPy.
    clat    | <float>     | centre latitude to predict the arrival time to. Does not need to be exact is dt_crit is large.
    clon    | <float>     | centre longitude to predict the arrival time to. Does not need to be exact is dt_crit is large.
    dt_crit | <float>     | critical time within which there's a risk of being within the a taper at the end/start of a day.

    Returns: <bool> | whether the phase arrival is within dt_crit of midnight.
    '''
    t0 = UTCDateTime(ev)
    # Predict arrival time.
    maybe_arr = model.get_travel_times_geo(source_depth_in_km=ev_data["DEPTH"],
                                           source_latitude_in_deg=ev_data["LAT"],
                                           source_longitude_in_deg=ev_data["LON"],
                                           receiver_latitude_in_deg=clat,
                                           receiver_longitude_in_deg=clon,
                                           phase_list=[phase])
    taper_crossing = False
    if len(maybe_arr):
        arr = maybe_arr[0]
        p_arr = arr.time
        # Compute absolute predicted arrival time.
        t = t0 + p_arr
        # Time diff to start of day
        dt0 = t - UTCDateTime(t0.year,t0.month,t0.day)
        # Time diff to end of day
        dt1 = UTCDateTime(t0.year,t0.month,t0.day)+3600*24 - t
        # Check whether either time diff may be within the critical value.
        if abs(dt0) < dt_crit or abs(dt1) < dt_crit:
            taper_crossing = True
    return taper_crossing

def mean_arrival_time(ev,stas,ev_data,phase,stas_df):
    ''' Compute mean arrival time across multiple stations

    ev      | <str>          | event id in the form of a UTC timestamp without the Z at the end (Z is assumed).
    stas    | <list> [<str>] | stations to compute arrival times at. Must be keys of stas_df.
    ev_data | <pd.Series>    | row of data for the event as read from the events dataframe. Must contain entries for "LAT", "LON" and "DEPTH" of the event.
    stas_df | <pd.DataFrame> | dataframe containing station location data ("lat" and "lon" as columns).
    phase   | <str>          | phase of interest.

    Returns: <float> or None | mean arrival time or None if no predicted arrivals.
    '''
    arrs = []
    for sta in stas:
        maybe_arr = model.get_travel_times_geo(source_depth_in_km=ev_data["DEPTH"],
                                               source_latitude_in_deg=ev_data["LAT"],
                                               source_longitude_in_deg=ev_data["LON"],
                                               receiver_latitude_in_deg=stas_df.loc[sta,"lat"],
                                               receiver_longitude_in_deg=stas_df.loc[sta,"lon"],
                                               phase_list=[phase])
        if len(maybe_arr):
            arr = maybe_arr[0]
            arrs.append(arr.time)
    if len(arrs)==0:
        return
    else:
        return np.mean(arrs)

def process_arrival_at_sta(sta,taper_crossing,arrtime_abs,data_dir=data_dir,SDS_parent=SDS_parent,all_channels=all_channels,avoid_reprocessing=avoid_reprocessing):
    ''' Cut waveforms +/- 1 hour about the arrival time for a station.

    sta                | <str>                                | name of station to cut the waveform for.
    taper_crossing     | <bool>                               | whether the event may be close enough to midnight to potentially be within the taper around the start or end of day.
    arrtime_abs        | <obspy.core.utcdatetime.UTCDateTime> | phase arrival time.
    data_dir           | <str>                                | path to the ATaCR data directory (used to generate SPECTRA etc.).
    SDS_parent         | <str>                                | path to the SDS data directory.
    all_channels       | <list> [<str>]                       | list of channels to search over for waveforms. Currently uses exact matching of supplied names. This must be 4 long with one vertical, one hydrophone, and two horizontal channels.
    avoid_reprocessing | <bool>                               | whether to cut from the ATaCR data directory in cases of taper_crossing == False or to always cut from the SDS data directory (with the same processing steps used to generate the ATaCR data directory). The choice of having this set to True or False in the Macquarie study depended on computational cost considerations (though it was found that setting this to False and reprocessing from the SDS directory did not introduce *too* much additional computational load).

    Returns: <dict> or None | channel:waveform data if all four channels are present, or two channels are present and one of them is vertical.
    '''
    # Identify relevant path construction for the taper crossing status of the event.
    if not taper_crossing and avoid_reprocessing:
        get_src = lambda sta,year,julday,cha,net="3F" : os.path.join(data_dir,"%s/%u.%03d..%s.mseed" % (sta,year,julday,cha))
    else:
        get_src = lambda sta,year,julday,cha,net="3F" : os.path.join(SDS_parent,str(year),net,sta.replace(net+".",""),cha+".D",".".join([net,sta.replace(net+".",""),"",cha,"D",str(year),"%03d" % julday]))
    chas = dict()
    # Iterate through channels.
    for cha in all_channels:
        # Determine path to the waveform.
        src = get_src(sta,arrtime_abs.year,arrtime_abs.julday,cha)
        try:
            if os.path.exists(src):
                st = read(src)
                # Identify sampling frequency to ensure the trace end cut is aligned.
                freq = st[0].stats.sampling_rate
                # Wide start and end to cut the trace by (as required for ATaCR).
                trace_start = arrtime_abs-3600
                trace_end = arrtime_abs-3600+7200-1/freq
                # Identify the waveform files representing the start and end of the wide trim and load those if not already covered.
                src_pre = get_src(sta,trace_start.year,trace_start.julday,cha)
                src_end = get_src(sta,trace_end.year,trace_end.julday,cha)
                if src_pre != src:
                    st += read(src_pre)
                if src_end != src:
                    st += read(src_end)
                # Combine traces if multiple are present.
                st.merge(fill_value="interpolate")
                # Perform wide trim.
                st.trim(trace_start,trace_end)
                # Make sure the trace still exists after the trim.
                if len(st) and len(st[0].data):
                    if not taper_crossing and avoid_reprocessing:
                        # Detrend (demean and linear)
                        st.detrend('demean')
                        st.detrend('linear')
                        # Taper.
                        st.taper(max_percentage=0.05,type="cosine")
                        # Store processed waveform.
                        chas[cha] = st
                    else:
                        # The following processing steps are the same as those in 10_preprocess_all.py (around lines 87-101).
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
                        tr.data = tr.data * gains[cha[-1]]
                        # Resample.
                        tr.resample(resample_freq)
                        # Store processed waveform.
                        chas[cha] = tr
        except:
            pass
    # Only commit to writing if 4 (Z and hydro, H2 and H1) or 2 (Z and another - if this other is not hydrophone, ATaCR may encounter a non-blocking error) channels are present.
    if len(chas)==4 or (len(chas)==2 and any([("Z" in k) for k in chas.keys()])):
        return chas
    else:
        return

def exec_atacr_denoise(accepted_stas,t0,t1,n_cores=1):
    ''' Execute atacr denoising via a copy of the `denoise_event.sh` script with the start and end times for station spectra calculation populated.

    accepted_stas | <list> [<str>]                       | list of station names to execute denoising for.
    t0            | <obspy.core.utcdatetime.UTCDateTime> | start time for computing station spectra.
    t1            | <obspy.core.utcdatetime.UTCDateTime> | end time for computing station spectra.
    n_cores       | <int>                                | number of cores to distribute execution over.
    '''
    # Add the start and end days as parameters to the ATaCR
    with open("denoise_event.sh") as infile:
        sh = infile.read()
    t0_str = "%u-%02d-%02d" % (t0.year,t0.month,t0.day)
    t1_str = "%u-%02d-%02d" % (t1.year,t1.month,t1.day)
    params = {"t0":t0_str,
              "t1":t1_str}
    params_sh = ""
    for k,v in params.items():
        params_sh += "%s=\"%s\"\n" % (k.strip(),v.strip())
    with open(".denoise_event_run.sh","w") as outfile:
        outfile.write(params_sh+"\n"+sh)
    print("***Running***")
    # Parallelise the execution of denoising across the station collection.
    with multiprocessing.Pool(int(n_cores)) as p:
        p.map(exec_denoise,accepted_stas)
    print("Finished")
    return

def collate_denoised_waveforms(ev_data,stas_df,phase,min_stas,pre,post,upsample_freq,analyse_denoise_types=["day.ZP-21","day.ZP-H","sta.ZP-21","day.Z2-1","day.ZH","sta.Z2-1"]):
    ''' Collate the denoised waveforms of the requested types (in addition to the undenoised waveforms) and cut them to predicted phase arrival time alignment.

    ev_data               | <pd.Series>    | row of data for the event as read from the events dataframe. Must contain entries for "LAT", "LON" and "DEPTH" of the event.
    stas_df               | <pd.DataFrame> | dataframe containing station location data ("lat" and "lon" as columns).
    phase                 | <str>          | phase of interest.
    min_stas              | <int>          | minimum number of stations to accept denoising for. Set to 4 in the study (to avoid needing to review events recorded across too few events to be meaningful for teleseismic tomography), but 0 here for testing.
    pre                   | <float>        | seconds before the predicted arrival time of the phase to each station to start the wavefrom trim.
    post                  | <float>        | seconds after the predicted arrival time of the phase to each station to end the wavefrom trim.
    analyse_denoise_types | <list> [<str>] | identification strings for the types of denoising to store. Must match ATaCR format.

    Returns: <seismem.SeisMem> | collection of trimmed waveforms.
    '''
    SM = SeisMem()
    # Identify stations for which an ATaCR event was successfully generated (whether denoising was successful or not is not checked at this stage).
    stas = sorted([f for f in os.listdir("EVENTS") if os.path.isdir(os.path.join("EVENTS",f)) and "3F" in f])
    # Exit if the requested minimum number of stations is not reached.
    if len(stas) <= min_stas:
        return None
    # Iterate through stations (as folders/paths).
    for p0 in stas:
        p = p0
        # Get path to station folder.
        p = os.path.join("EVENTS",p)
        # Identify station name (without network code).
        sta = p0.split(".")[-1]
        # Predict arrival time of phase.
        maybe_arr = model.get_travel_times_geo(source_depth_in_km=ev_data["DEPTH"],
                                               source_latitude_in_deg=ev_data["LAT"],
                                               source_longitude_in_deg=ev_data["LON"],
                                               receiver_latitude_in_deg=stas_df.loc[sta,"lat"],
                                               receiver_longitude_in_deg=stas_df.loc[sta,"lon"],
                                               phase_list=[phase])
        if len(maybe_arr):
            # Identify first arrival.
            arr = maybe_arr[0]
            p_arr = arr.time
            print(sta,"predicted arrival",phase,arr.time)
            # Determine absolute time of first arrival.
            t = UTCDateTime(ev) + p_arr
            # Get path to denoised waveforms folder.
            p_corr = os.path.join(p,"CORRECTED")
            # Identify undenoised vertical channel waveform file.
            f_uncorr = [f for f in os.listdir(p) if f.endswith("Z.mseed")][0]
            st_uncorr = read(os.path.join(p,f_uncorr))
            tr_uncorr = st_uncorr[0]
            # Trim waveform to final width requested about the predicted arrival time.
            tr_uncorr.trim(t-pre,t+post)
            print("\t",t-pre,t+post)
            # Check if the undenoised waveform remains when trimmed to final width.
            if len(tr_uncorr.data):
                # Detrend (demean and linear)
                tr_uncorr.detrend("constant")
                tr_uncorr.detrend("linear")
                # Normalise.
                tr_uncorr.normalize()
                if upsample_freq:
                    # Upsample.
                    tr_uncorr.resample(upsample_freq)
                # Store undenoised waveform.
                SM.add(tr_uncorr,"/".join(["og",sta,ev+"."+phase,".".join([ev,sta,"Z",phase,"SAC"])]))
                if os.path.exists(p_corr):
                    # Identify denoised waveforms if they exist.
                    fs_corr = [f for f in os.listdir(p_corr)]
                    # Iterate through the types of denoised waveforms requested.
                    for i,denoise_type in enumerate(analyse_denoise_types):
                        # Identify relevant denoised waveforms for active denoise type.
                        f = [f for f in fs_corr if denoise_type+"." in f]
                        if len(f):
                            # Load denoised waveform.
                            st = read(os.path.join(p_corr,f[0]))
                            tr = st[0]
                            # Trim waveform to final width requested about the predicted arrival time.
                            tr.trim(t-pre,t+post)
                            print("\t",t-pre,t+post)
                            # Check if waveform remains when trimmed to final width.
                            if len(tr.data):
                                # Detrend (demean and linear)
                                tr.detrend("constant")
                                tr.detrend("linear")
                                # Normalise.
                                tr.normalize()
                                if upsample_freq:
                                    # Upsample.
                                    tr.resample(upsample_freq)
                                # Store denoised waveform.
                                SM.add(tr,"/".join([denoise_type,sta,ev+"."+phase,".".join([ev,sta,"Z",phase,"SAC"])]))
            else:
                print("No trace for this time of day")
        else:
            print("No %s arrival" % phase,ev,sta)
    return SM



# Wrapper function for handling the various aspects of earthquake denoising.
# NOTE: this function uses GLOBAL variables defined above, some of which are restated in the inputs as the default for clarity.
def process_ev(ev,ev_data,stas_df,phase=phase,pre=pre,post=post,n_cores=n_cores,pm_days=pm_days,upsample_freq=upsample_freq):
    '''
    ev      | <str>          | event id in the form of a UTC timestamp without the Z at the end (Z is assumed).
    ev_data | <pd.Series>    | row of data for the event as read from the events dataframe. Must contain entries for "LAT", "LON" and "DEPTH" of the event.
    stas_df | <pd.DataFrame> | dataframe containing station location data ("lat" and "lon" as columns).
    phase   | <str>          | phase of interest.
    pre     | <float>        | start time to cut the denoised waveform for saving.
    post    | <float>        | end time to cut the denoised waveform for saving.
    n_cores | <int>          | number of cores to use in parallelisation.
    pm_days | <int>          | +/- days to compute "average" station spectra over.
    upsample_freq |
    '''
    print("\n\n***********************")
    print(ev,"with mag",ev_data["MAG"])
    # Determine whether the phase arrival is at risk of being within the tapered region, in which case read earthquake data from the SDS and preprocess (rather than DATA without much additional processing). This prevents the need to reprocess data if a suitable version already exists (i.e. not within the tapered region).
    taper_crossing = check_taper_crossing(ev,ev_data,phase)
    if taper_crossing:
        print(ev,"may cross taper - will re-preprocess data from SDS")
    # Scan stas for phase arrival.
    stas = [f for f in os.listdir(data_dir) if "3F" in f and not f.startswith(".")]
    # Isolation station names (without network name).
    sta_names = [net_sta.split(".")[1] for net_sta in stas]
    # Predict mean arrival time across the available stations.
    p_arr = mean_arrival_time(ev,sta_names,ev_data,phase,stas_df)
    # Exit this function if no predicted arrivals.
    if p_arr is None:
        print("No %s phase, skipping" % phase)
        return
    # Print relative arrival time.
    print("\tMEAN Predicted arrival %s" % phase,p_arr)
    # Compute absolute time of arrival.
    arrtime_abs = UTCDateTime(ev) + p_arr
    # Clean EVENTS directory.
    if os.path.exists("EVENTS"):
        shutil.rmtree("EVENTS")
    # List to store accepted stations.
    accepted_stas = list()
    # Iterate through available stations.
    for sta in stas:
        # Process waveforms for each channel.
        chas = process_arrival_at_sta(sta,taper_crossing,arrtime_abs)
        if chas is not None:
            if not os.path.exists("EVENTS/%s" % sta):
                os.makedirs("EVENTS/%s" % sta)
            for cha,tr in chas.items():
                tr.write("EVENTS/%s/%u.%03d.%02d.%02d.%s.mseed" % (sta,arrtime_abs.year,arrtime_abs.julday,arrtime_abs.hour,arrtime_abs.minute,cha))
            accepted_stas.append(sta)
    # Exit if there are no accepted stations.
    if not len(accepted_stas):
        print("Event outside of data recording time, skipping")
        return
    # Identify start and end of times to station "average" spectra over.
    seconds_per_day = 3600 * 24
    seconds_margin = pm_days * seconds_per_day
    t0 = arrtime_abs - seconds_margin
    t1 = arrtime_abs + seconds_margin
    # Execute atacr denoising, which will also attempt to perform station spectra denoising.
    exec_atacr_denoise(accepted_stas,t0,t1,n_cores)
    # Combine all waveforms of interest (which includes the original waveform) into one SM object.
    SM = collate_denoised_waveforms(ev_data,stas_df,phase,min_stas,pre,post,upsample_freq)
    if SM is not None:
        # Clear any file that may be called denoised-data.
        if os.path.exists("denoised-data") and not os.path.isdir("denoised-data"):
            os.remove("denoised-data")
        # Ensure output folder for storing waveforms exists.
        if not os.path.exists("denoised-data"):
            os.mkdir("denoised-data")
        # Save waveform collection for this event.
        SM.dump("denoised-data/%s.%s.json" % (ev,phase))
    return

def clean_range_files():
    ''' Remove multi-day pickle files to prevent their consideration when denoising for quicker execution of the denoising script.
    '''
    to_rm = []
    p = "TF_STA"
    if os.path.exists(p):
        for f in os.listdir(p):
            _p = os.path.join(p,f)
            if os.path.isdir(_p):
                fs = os.listdir(_p)
                # Day range spectra, which are different for each event (and so need regenerating).
                to_rm.extend([os.path.join(_p,f) for f in fs if "-" in f])
    p = "AVG_STA"
    if os.path.exists(p):
        for f in os.listdir(p):
            _p = os.path.join(p,f)
            if os.path.isdir(_p):
                fs = os.listdir(_p)
                # Old station averaged computed from day-range spectra.
                to_rm.extend([os.path.join(_p,f) for f in fs])
    for f in to_rm:
        os.remove(f)
    return

if __name__=="__main__":
    # Identify events to process.
    evs_to_process_f = "accepted-%s.csv" % phase
    evs_to_process_df = pd.read_csv(evs_to_process_f)
    evs = [".".join(os.path.basename(e).split(".")[:2]).replace("Z","") for e in evs_to_process_df["UTC"].to_list()]
    # Load full earthquake dataframe.
    df = pd.read_csv(eq_f)
    # Load stations dataframe.
    stas_df = pd.read_csv(stas_f,index_col=0)
    # Assign timestamps to the earthquake dataframe.
    df["UTC"] = df.apply(lambda row : UTCDateTime(row["DATE"] + "T" + row["TIME"]),axis=1)
    df["timestamp"] = df.apply(lambda row : str(row["UTC"])[:-1],axis=1)
    cwd = os.getcwd()
    # Iterate through events to process.
    for ev in evs:
        e0 = UTCDateTime(ev)
        # Check that event exists after the requested start time.
        if e0 > UTCDateTime(start_processing_from):
            # Make sure there are no residual files in TF_STA or AVG_STA that could cause slowdown of ATaCR denoising.
            clean_range_files()
            if os.path.exists(os.path.join(cwd,"denoised-data","%s.%s.json" % (ev,phase))) and not overwrite_evs:
                # Ignore the event if a processed version of it already exists and overwriting is not requested.
                print("Event already handled, skipping")
                pass
            else:
                # Identify event data for active event.
                ev_data = df[df["timestamp"] == ev].iloc[0]
                # Execute processing/denoising.
                process_ev(ev,ev_data,stas_df)
    # Ensure the output folder is a folder, even if there are no denoised waveforms saved to it.
    if not os.path.exists("denoised-data"):
        os.mkdir("denoised-data")
    # Store auxiliary information in output folder.
    shutil.copy(evs_to_process_f,"denoised-data")
    shutil.copy("12_process_event.py","denoised-data")
